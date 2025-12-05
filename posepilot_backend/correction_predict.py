"""
pose correction prediction pipeline.
Updated to handle 7 classes and match the training sequence length logic.
FIXED: Now returns proper correction data for visualization.
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
    Scale the input data using the provided scalers.

    Parameters
    ----------
    data_input : pd.DataFrame
        Input data to be scaled (features f1-f9)
    scalers : list
        List of fitted MinMaxScaler objects (one per feature f1-f9)

    Returns
    -------
    pd.DataFrame
        Scaled input data
    """
    feature_columns = [f"f{i}" for i in range(1, 10)]
    for i, scaler in enumerate(scalers):
        col_name = feature_columns[i]
        if col_name in data_input.columns:
            data_input[col_name] = scaler.transform(
                data_input[col_name].values.reshape(-1, 1)
            ).flatten()
        else:
            print(f"Warning: Column {col_name} not found in data_input for scaling.")
    return data_input


def unscale_data(data_output, scalers):
    """
    Inverse scale the output data using the provided scalers.

    Parameters
    ----------
    data_output : np.ndarray
        Output data from the model (features f1-f9)
    scalers : list
        List of fitted MinMaxScaler objects (one per feature f1-f9)

    Returns
    -------
    np.ndarray
        Unscaled output data
    """
    feature_columns = [f"f{i}" for i in range(1, 10)]
    for i, scaler in enumerate(scalers):
        data_output[:, i] = (
            scaler.inverse_transform(data_output[:, i].reshape(-1, 1))
            .flatten()
        )
    return data_output


def load_model_and_scalers(device, pose):
    """
    Load the correction model and scalers for a specific pose.
    Assumes model and scalers are named {pose}_correction_model.pth/pkl.

    Parameters
    ----------
    device : torch.device
        Device to load model on
    pose : str
        Pose name (e.g., 'chair', 'cobra') for model/scaler file names

    Returns
    -------
    tuple
        Loaded model and scalers
    """
    input_size = 9
    hidden_size = 256
    num_layers = 1
    num_output_features = 9

    model = CorrModel(input_size, hidden_size, num_layers, num_output_features).to(
        device
    )
    model_path = f"models/{pose}_correction_model.pth"
    print(f"Loading model from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    scalers_path = f"models/{pose}_correction_scalers.pkl"
    print(f"Loading scalers from: {scalers_path}")
    scalers = pickle.load(open(scalers_path, "rb"))

    return model, scalers


def predict_correction_sequence(data_input, device, pose, scalers, model):
    """
    Run the correction model prediction on the processed input data.

    Parameters
    ----------
    data_input : pd.DataFrame
        Input data (features f1-f9, scaled)
    device : torch.device
        Device to run inference on
    pose : str
        Pose name (used for potential logging)
    scalers : list
        List of scalers used for inverse transform
    model : CorrModel
        Loaded model

    Returns
    -------
    tuple
        (unscaled_model_outputs, unscaled_input_data)
    """
    data_input_tensor = torch.tensor(data_input.values, dtype=torch.float32).unsqueeze(0).to(device)
    print(f"Input tensor shape for model: {data_input_tensor.shape}")

    with torch.no_grad():
        outputs_scaled = model(data_input_tensor)

    outputs_scaled_np = outputs_scaled.squeeze(0).cpu().detach().numpy()
    outputs_unscaled = unscale_data(outputs_scaled_np, scalers)

    input_scaled_np = data_input_tensor.squeeze(0).cpu().detach().numpy()
    input_unscaled = unscale_data(input_scaled_np, scalers)

    return outputs_unscaled, input_unscaled


def plot_correction_results(data_original, input_unscaled, outputs_unscaled, pose_name):
    """
    Plot the original data, scaled input data, and model outputs for comparison.

    Parameters
    ----------
    data_original : pd.DataFrame
        Original angle data (f1-f9) before scaling (length seq_len)
    input_unscaled : np.ndarray
        Unscaled input data fed to the model (length seq_len)
    outputs_unscaled : np.ndarray
        Unscaled model outputs (length seq_len-1)
    pose_name : str
        Name of the pose for title
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

    plot_length_input = len(input_unscaled)
    plot_length_output = len(outputs_unscaled)

    for i in range(3):
        for j in range(3):
            feature_idx = 3 * i + j
            max_len = max(len(data_original), plot_length_input)
            axs[i, j].set_xlim([0, max_len - 1])

            if len(data_original) > 0:
                axs[i, j].plot(
                    data_original.iloc[:, feature_idx], label="Original (Raw Input Angles)", linestyle="--", color='blue'
                )

            axs[i, j].plot(
                range(plot_length_input),
                input_unscaled[:, feature_idx],
                label="Input (t) - Unscaled",
                linestyle="",
                marker="o",
                color='orange',
                markersize=3
            )

            axs[i, j].plot(
                range(1, plot_length_output + 1),
                outputs_unscaled[:, feature_idx], label="Output (t+1) - Predicted", linestyle="-", color='green'
            )

            if plot_length_output > 0:
                lower_bound_1std = outputs_unscaled[:, feature_idx] - np.std(outputs_unscaled[:, feature_idx])
                upper_bound_1std = outputs_unscaled[:, feature_idx] + np.std(outputs_unscaled[:, feature_idx])

                axs[i, j].fill_between(
                    range(1, plot_length_output + 1),
                    lower_bound_1std,
                    upper_bound_1std,
                    alpha=0.2,
                    color="grey",
                    label="1-std band (on output)",
                )

                lower_bound_2std = outputs_unscaled[:, feature_idx] - 1.5 * np.std(outputs_unscaled[:, feature_idx])
                upper_bound_2std = outputs_unscaled[:, feature_idx] + 1.5 * np.std(outputs_unscaled[:, feature_idx])
                axs[i, j].fill_between(
                    range(1, plot_length_output + 1),
                    lower_bound_2std,
                    upper_bound_2std,
                    alpha=0.1,
                    color="grey",
                    label="1.5-std band (on output)",
                )

            axs[i, j].set_title(
                f"Feature {feature_idx + 1} ({feature_label[feature_idx]})",
                fontsize=12,
            )

            for k in range(plot_length_output):
                input_val_at_t = input_unscaled[k, feature_idx]
                predicted_val_at_t_plus_1 = outputs_unscaled[k, feature_idx]
                deviation_from_input = abs(predicted_val_at_t_plus_1 - input_val_at_t)
                std_of_output = np.std(outputs_unscaled[:, feature_idx])
                threshold = 1.5 * std_of_output

                if deviation_from_input > threshold:
                    axs[i, j].plot(
                        k + 1,
                        predicted_val_at_t_plus_1,
                        "ro",
                        label="Predicted Deviation (t->t+1)",
                        markersize=5
                    )

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    unique_labels = sorted(list(set(labels)))
    selected_lines = [lines[labels.index(label)] for label in unique_labels]

    fig.subplots_adjust(top=0.95)
    fig.legend(
        selected_lines,
        unique_labels,
        loc="upper center",
        framealpha=1,
        fontsize=10,
        ncol=min(4, len(unique_labels)),
        bbox_to_anchor=(0.5, 0.98),
    )
    fig.suptitle(f"Pose Correction Prediction: {pose_name}", fontsize=16, y=0.95)
    fig.supxlabel("Frames (n)", fontsize=12, x=0.5)
    fig.supylabel("Degrees (°)", fontsize=12, y=0.5, x=0.005)

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.savefig(f"correction_output_{pose_name}.png")
    plt.show()


def corr_predict(pose, data, return_data=False):
    """
    Predict pose corrections for input data.

    Parameters
    ----------
    pose : str
        Pose name (e.g., 'chair', 'cobra')
    data : pd.DataFrame
        Input pose landmark data (raw landmarks, e.g., 132 columns per row)
    return_data : bool
        If True, return correction data dict instead of just plotting

    Returns
    -------
    dict or None
        If return_data=True, returns dict with correction data, else None
    """
    print(f"Starting correction prediction for pose: {pose}")

    # Step 1: Preprocess Raw Landmarks into Angle Features
    try:
        structured_df, body_pose_landmarks = structure_data(data)
        structured_df, body_pose_landmarks = update_body_pose_landmarks(
            structured_df, body_pose_landmarks
        )
        angle_features_df = correction_angles_convert(structured_df)
        print(f"Extracted angle features. Shape: {angle_features_df.shape}")
    except Exception as e:
        print(f"Error during landmark/angle conversion: {e}")
        return None

    # Step 2: Determine Required Sequence Length
    try:
        average_lengths = {
            'chair': 85,
            'cobra': 144,
            'downdog': 120,
            'goddess': 91,
            'surya_namaskar': 1221,
            'tree': 110,
            'warrior': 103
        }
        target_length = average_lengths.get(pose, 100)
        print(f"Using target length {target_length} for equal_rows based on average for '{pose}'.")

        angle_features_df = equal_rows(angle_features_df, target_length, calculate_error=False)
        print(f"Applied equal_rows (no error calc). New shape: {angle_features_df.shape}")
    except Exception as e:
        print(f"Error during equal_rows: {e}")
        return None

    # Step 3: Load Model and Scalers
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"Using device: {device}")

        model, scalers = load_model_and_scalers(device, pose)
    except FileNotFoundError as e:
        print(f"Error loading model/scalers: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error loading model/scalers: {e}")
        return None

    # Step 4: Scale Features
    try:
        scaled_angle_features_df = scale_data(angle_features_df.copy(), scalers)
        print(f"Features scaled. Shape: {scaled_angle_features_df.shape}")
    except Exception as e:
        print(f"Error during scaling: {e}")
        return None

    # Step 5: Run Prediction
    try:
        outputs_unscaled, input_unscaled = predict_correction_sequence(
            scaled_angle_features_df, device, pose, scalers, model
        )
        print(f"Prediction completed. Output shape: {outputs_unscaled.shape}, Input shape: {input_unscaled.shape}")
    except Exception as e:
        print(f"Error during model prediction: {e}")
        return None

    # Step 6: Return data if requested or plot
    if return_data:
        # Return the data in the format expected by gradio_app.py
        return {
            "status": "success",
            "pose": pose,
            "input_data": input_unscaled,  # Unscaled input (seq_len, 9)
            "corrected_data": outputs_unscaled,  # Unscaled outputs (seq_len-1, 9)
            "reference_data": angle_features_df,  # Original angle features before scaling
        }
    else:
        # Step 7: Plot Results
        try:
            plot_correction_results(angle_features_df, input_unscaled, outputs_unscaled, pose)
            print("Plot saved and displayed.")
        except Exception as e:
            print(f"Error during plotting: {e}")
            return None

        print(f"Correction prediction for pose '{pose}' completed successfully.")
        return None


def predict_correction_from_dataframe(data, pose):
    """
    Predict pose corrections from a DataFrame directly.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing pose landmarks (e.g., raw landmark columns)
    pose : str
        Pose name for correction (e.g., 'chair', 'cobra')

    Returns
    -------
    dict or None
        Dictionary containing correction data for plotting or None if error
    """
    print(f"Predicting correction for pose: {pose} from DataFrame")
    try:
        # Use return_data=True to get the correction data instead of plotting
        correction_data = corr_predict(pose, data, return_data=True)
        return correction_data
    except Exception as e:
        print(f"Error in correction prediction from DataFrame: {e}")
        return {"status": "error", "pose": pose, "error": str(e)}


def predict_correction_from_csv(csv_path, pose):
    """
    Predict pose corrections from a CSV file.

    Parameters
    ----------
    csv_path : str
        Path to CSV file containing pose landmarks (e.g., raw landmark columns)
    pose : str
        Pose name for correction (e.g., 'chair', 'cobra')

    Returns
    -------
    dict or None
        Dictionary containing correction data for plotting or None if error
    """
    print(f"Predicting correction for pose: {pose} from CSV: {csv_path}")
    try:
        data = pd.read_csv(csv_path)
        return predict_correction_from_dataframe(data, pose)
    except FileNotFoundError:
        print(f"Error: CSV file not found: {csv_path}")
        return {"status": "error", "pose": pose, "error": f"File not found: {csv_path}"}
    except Exception as e:
        print(f"Error reading CSV file or predicting: {e}")
        return {"status": "error", "pose": pose, "error": str(e)}


# Example usage (uncomment if running this script directly):
# if __name__ == "__main__":
#     csv_file_path = "path/to/your/chair_pose_landmarks.csv"
#     pose_name = "chair"
#     result = predict_correction_from_csv(csv_file_path, pose_name)
#     print(result)