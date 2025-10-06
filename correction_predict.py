"""
pose correction prediction pipeline.
"""

import torch
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from correction_model import CorrModel
from utils import (
    cal_error,
    correction_angles_convert,
    equal_rows,
    structure_data,
    update_body_pose_landmarks,
)

warnings.filterwarnings("ignore")


def scale_data(data_input, scalers):
    """
    scale the input data using the provided scalers.

    Parameters
    ----------
    data_input : pd.DataFrame
        input data to be scaled
    scalers : list
        list of fitted scalers

    Returns
    -------
    pd.DataFrame
        scaled input data
    """
    for i, scaler in enumerate(scalers):
        data_input.iloc[:, i] = scaler.transform(
            data_input.iloc[:, i].values.reshape(-1, 1)
        )
    return data_input


def prepare_data(data_input, seq_length, desired_frame_count=25):
    """
    prepare the data by selecting frames or using all frames.

    Parameters
    ----------
    data_input : pd.DataFrame
        input data
    seq_length : int
        sequence length
    desired_frame_count : int
        number of frames to select (25 for subset, None for all frames)

    Returns
    -------
    tuple
        tuple containing (processed_data, frames_selected)
    """
    if desired_frame_count is None or desired_frame_count >= len(data_input):
        frames_selected = np.arange(len(data_input))
    else:
        frames_selected = np.linspace(
            1, len(data_input[:seq_length]) - 2, num=desired_frame_count, dtype=int
        )

    data_input = data_input.iloc[frames_selected]
    return data_input.reset_index(drop=True), frames_selected


def load_model(device, input_size, hidden_size, num_layers, num_output_features, pose):
    """
    load the correction model and scalers for a specific pose.

    Parameters
    ----------
    device : torch.device
        device to load model on
    input_size : int
        input size for the model
    hidden_size : int
        hidden size for the model
    num_layers : int
        number of layers for the model
    num_output_features : int
        number of output features
    pose : str
        pose name for model file

    Returns
    -------
    tuple
        loaded model and scalers
    """
    model = CorrModel(input_size, hidden_size, num_layers, num_output_features).to(
        device
    )
    model_path = f"models/{pose}_correction_model.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()

    scalers_path = f"models/{pose}_correction_scalers.pkl"
    scalers = pickle.load(open(scalers_path, "rb"))

    return model, scalers


def test(data_input, device, pose):
    """
    test the correction model on input data.

    Parameters
    ----------
    data_input : pd.DataFrame
        input data for testing
    device : torch.device
        device to run inference on
    pose : str
        pose name for model loading
    """
    data_input = equal_rows(data_input, 25)
    data_input, frames_selected = prepare_data(data_input, seq_length=25)

    data_original = data_input.copy()

    input_size = 9
    hidden_size = 256
    num_layers = 1
    num_output_features = 9

    columns_to_drop = ["label", "error"]
    data_input = data_input.drop(columns=columns_to_drop, axis=1)

    model, scalers = load_model(
        device, input_size, hidden_size, num_layers, num_output_features, pose
    )

    data_input = scale_data(data_input, scalers)
    data_input_tensor = torch.tensor(data_input.values, dtype=torch.float32).to(device)

    outputs = model(data_input_tensor[0:1])
    outputs = outputs.view(-1, 9)

    outputs = outputs.cpu().detach().numpy()
    for i in range(len(scalers)):
        outputs[:, i] = (
            scalers[i].inverse_transform(outputs[:, i].reshape(-1, 1)).reshape(-1)
        )

    transformed_data_input = data_input_tensor.cpu().detach().numpy()
    for i in range(len(scalers)):
        transformed_data_input[:, i] = (
            scalers[i]
            .inverse_transform(transformed_data_input[:, i].reshape(-1, 1))
            .reshape(-1)
        )

    plot_results(data_original, transformed_data_input, outputs, frames_selected)


def plot_results(data_original, data_input, outputs, frames_selected):
    """
    plot the original data, input data, and model outputs for comparison.

    Parameters
    ----------
    data_original : pd.DataFrame
        original data
    data_input : np.ndarray
        input data
    outputs : np.ndarray
        model outputs
    frames_selected : np.ndarray
        selected frame indices
    """
    feature_label = [
        "Left Elbow",
        "Right Elbow",
        "Left Hip",
        "Right Hip",
        "Left Knee",
        "Right Knee",
        "Neck",
        "Left Shoulder",
        "Right Shoulder",
    ]
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))

    for i in range(3):
        for j in range(3):
            axs[i, j].set_xlim([0, len(data_original)])

            axs[i, j].plot(
                data_original.iloc[:, 3 * i + j], label="Original", linestyle="--"
            )
            axs[i, j].plot(
                frames_selected,
                data_input[:, 3 * i + j],
                label="Input",
                linestyle="",
                marker="o",
            )
            axs[i, j].plot(
                frames_selected, outputs[:, 3 * i + j], label="Output", linestyle="-"
            )

            lower_bound_1std = outputs[:, 3 * i + j] - np.std(outputs[:, 3 * i + j])
            upper_bound_1std = outputs[:, 3 * i + j] + np.std(outputs[:, 3 * i + j])
            axs[i, j].fill_between(
                frames_selected,
                lower_bound_1std,
                upper_bound_1std,
                alpha=0.2,
                color="grey",
                label="1-std band",
            )

            lower_bound_2std = outputs[:, 3 * i + j] - 2 * np.std(outputs[:, 3 * i + j])
            upper_bound_2std = outputs[:, 3 * i + j] + 2 * np.std(outputs[:, 3 * i + j])
            axs[i, j].fill_between(
                frames_selected,
                lower_bound_2std,
                upper_bound_2std,
                alpha=0.1,
                color="grey",
                label="1.5-std band",
            )

            axs[i, j].set_title(
                "Feature " + str(3 * i + j + 1) + " " + f"({feature_label[3 * i + j]})",
                fontsize=15,
            )

            for k in range(len(frames_selected)):
                if data_input[k, 3 * i + j] < lower_bound_1std[k]:
                    axs[i, j].plot(
                        frames_selected[k],
                        data_input[k, 3 * i + j],
                        "ro",
                        label="incorrect points",
                    )
                    axs[i, j].arrow(
                        frames_selected[k],
                        data_input[k, 3 * i + j],
                        0,
                        lower_bound_1std[k] - data_input[k, 3 * i + j],
                        head_width=2,
                        head_length=0.1,
                        fc="red",
                        ec="red",
                        label="correction vector",
                    )
                if data_input[k, 3 * i + j] > upper_bound_1std[k]:
                    axs[i, j].plot(
                        frames_selected[k],
                        data_input[k, 3 * i + j],
                        "ro",
                        label="incorrect points",
                    )
                    axs[i, j].arrow(
                        frames_selected[k],
                        data_input[k, 3 * i + j],
                        0,
                        upper_bound_1std[k] - data_input[k, 3 * i + j],
                        head_width=2,
                        head_length=0.1,
                        fc="red",
                        ec="red",
                        label="correction vector",
                    )

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    unique_labels = sorted(list(set(labels)))
    selected_lines = [lines[labels.index(label)] for label in unique_labels]

    fig.subplots_adjust(top=0.2)
    fig.legend(
        selected_lines,
        unique_labels,
        loc="upper center",
        framealpha=1,
        fontsize=13,
        ncol=7,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle("PosePilot", fontsize=20, y=1.05)
    fig.supxlabel("Frames (n)", fontsize=15, x=0.5, y=0.004)
    fig.supylabel("Degrees (°)", fontsize=15, y=0.5, x=0.004)

    plt.tight_layout()
    plt.savefig("correction_output.png")
    plt.show()


def corr_predict(pose, data):
    """
    predict pose corrections for input data.

    Parameters
    ----------
    pose : str
        pose name
    data : pd.DataFrame
        input pose data

    Returns
    -------
    None
    """
    data, _ = structure_data(data)
    data, _ = update_body_pose_landmarks(data, _)
    data = correction_angles_convert(data)
    data = cal_error(data)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    test(data, device=device, pose=pose)


def predict_correction_from_dataframe(data, pose, use_all_frames=True):
    """
    predict pose corrections from a DataFrame directly.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing pose landmarks
    pose : str
        pose name for correction
    use_all_frames : bool
        whether to use all frames or select 25 frames

    Returns
    -------
    dict or None
        dictionary containing correction data for plotting or None if error
    """
    try:
        processed_data, _ = structure_data(data)
        processed_data, _ = update_body_pose_landmarks(processed_data, _)
        processed_data = correction_angles_convert(processed_data)
        processed_data = cal_error(processed_data)

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        correction_data = get_correction_data_for_plotting(
            processed_data, device, pose, use_all_frames
        )

        return correction_data

    except Exception as e:
        print(f"Error in correction prediction: {e}")
        return None


def predict_correction_from_csv(csv_path, pose):
    """
    predict pose corrections from a CSV file.

    Parameters
    ----------
    csv_path : str
        path to CSV file containing pose landmarks
    pose : str
        pose name for correction

    Returns
    -------
    dict or None
        dictionary containing correction data for plotting or None if error
    """
    try:
        data = pd.read_csv(csv_path)
        return predict_correction_from_dataframe(data, pose)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None


def get_correction_data_for_plotting(data_input, device, pose, use_all_frames=True):
    """
    get correction data needed for plotting graphs.

    Parameters
    ----------
    data_input : pd.DataFrame
        processed input data
    device : str
        device to run model on
    pose : str
        pose name

    Returns
    -------
    dict or None
        dictionary with correction data for plotting
    """
    try:
        model, scalers = load_model(device, 9, 256, 1, 9, pose)

        if use_all_frames:
            data_processed, frames_selected = prepare_data(
                data_input, seq_length=len(data_input), desired_frame_count=None
            )
        else:
            data_input = equal_rows(data_input, 25)
            data_processed, frames_selected = prepare_data(
                data_input, seq_length=25, desired_frame_count=25
            )

        data_original = data_processed.copy()

        feature_columns = [f"f{i}" for i in range(1, 10)]
        data_features = data_processed[feature_columns].copy()

        data_scaled = scale_data(data_features, scalers)
        data_tensor = torch.tensor(data_scaled.values, dtype=torch.float32).to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(data_tensor)
            outputs = outputs.view(-1, 9)

        outputs = outputs.cpu().detach().numpy()
        for i in range(len(scalers)):
            outputs[:, i] = (
                scalers[i].inverse_transform(outputs[:, i].reshape(-1, 1)).reshape(-1)
            )

        transformed_data_input = data_tensor.cpu().detach().numpy()
        for i in range(len(scalers)):
            transformed_data_input[:, i] = (
                scalers[i]
                .inverse_transform(transformed_data_input[:, i].reshape(-1, 1))
                .reshape(-1)
            )

        return {
            "input_data": transformed_data_input,
            "corrected_data": outputs,
            "reference_data": data_original,
            "peak_indices": frames_selected,
        }

    except Exception as e:
        print(f"Error getting correction data: {e}")
        return None
