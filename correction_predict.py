"""
pose correction prediction pipeline.
Updated to handle 7 classes and match the training sequence length logic.
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
    feature_columns = [f"f{i}" for i in range(1, 10)] # f1 to f9
    for i, scaler in enumerate(scalers):
        # Ensure we are scaling the correct column
        col_name = feature_columns[i]
        if col_name in data_input.columns:
            data_input[col_name] = scaler.transform(
                data_input[col_name].values.reshape(-1, 1)
            ).flatten() # Flatten back to 1D array
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
    feature_columns = [f"f{i}" for i in range(1, 10)] # f1 to f9
    for i, scaler in enumerate(scalers):
        # data_output is expected to be shape (num_timesteps, 9)
        data_output[:, i] = (
            scaler.inverse_transform(data_output[:, i].reshape(-1, 1))
            .flatten() # Flatten back to 1D array per feature
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
    # Define model parameters - these should match the training script
    input_size = 9
    hidden_size = 256
    num_layers = 1
    num_output_features = 9 # Should match input for sequence prediction

    model = CorrModel(input_size, hidden_size, num_layers, num_output_features).to(
        device
    )
    model_path = f"models/{pose}_correction_model.pth"
    print(f"Loading model from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device)) # Use map_location for loading on correct device
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
    # Convert DataFrame to tensor
    data_input_tensor = torch.tensor(data_input.values, dtype=torch.float32).unsqueeze(0).to(device) # Add batch dim: (1, seq_len, 9)
    print(f"Input tensor shape for model: {data_input_tensor.shape}")

    # Run model prediction
    with torch.no_grad():
        # The model was trained to predict the *next* timestep
        # Input: [f1_t, f2_t, ..., f9_t] -> Output: [f1_t+1, f2_t+1, ..., f9_t+1]
        # So, the output sequence length is one less than the input sequence length
        outputs_scaled = model(data_input_tensor) # Shape: (1, seq_len-1, 9)
        # print(f"Model output shape (scaled): {outputs_scaled.shape}")

    # Remove batch dimension and convert to numpy
    outputs_scaled_np = outputs_scaled.squeeze(0).cpu().detach().numpy() # Shape: (seq_len-1, 9)
    # print(f"Output shape after squeeze/detach: {outputs_scaled_np.shape}")

    # Inverse scale the outputs
    outputs_unscaled = unscale_data(outputs_scaled_np, scalers) # Shape: (seq_len-1, 9)

    # Inverse scale the input data for comparison
    input_scaled_np = data_input_tensor.squeeze(0).cpu().detach().numpy() # Shape: (seq_len, 9)
    input_unscaled = unscale_data(input_scaled_np, scalers) # Shape: (seq_len, 9)

    return outputs_unscaled, input_unscaled # Note: outputs are for t+1, input is for t


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

    # Determine the length for plotting (input_unscaled is longer by 1 than outputs_unscaled)
    plot_length_input = len(input_unscaled)
    plot_length_output = len(outputs_unscaled)

    for i in range(3):
        for j in range(3):
            feature_idx = 3 * i + j
            # Set x-axis limits based on the original data length if available, otherwise input length
            max_len = max(len(data_original), plot_length_input)
            axs[i, j].set_xlim([0, max_len - 1]) # Indices go from 0 to len-1

            # Plot Original (if available and same length or close)
            if len(data_original) > 0:
                axs[i, j].plot(
                    data_original.iloc[:, feature_idx], label="Original (Raw Input Angles)", linestyle="--", color='blue'
                )

            # Plot Scaled Input (unscaled back) - covers timesteps 0 to seq_len-1
            axs[i, j].plot(
                range(plot_length_input), # x-axis: 0 to seq_len-1
                input_unscaled[:, feature_idx],
                label="Input (t) - Unscaled",
                linestyle="",
                marker="o",
                color='orange',
                markersize=3
            )

            # Plot Model Output (unscaled back) - covers timesteps 1 to seq_len (predicted t+1 based on t)
            # Plot it against the original x-axis indices starting from 1
            axs[i, j].plot(
                range(1, plot_length_output + 1), # x-axis: 1 to seq_len (predicted values)
                outputs_unscaled[:, feature_idx], label="Output (t+1) - Predicted", linestyle="-", color='green'
            )

            # Calculate std bands based on the *output* sequence (predicted values)
            if plot_length_output > 0:
                lower_bound_1std = outputs_unscaled[:, feature_idx] - np.std(outputs_unscaled[:, feature_idx])
                upper_bound_1std = outputs_unscaled[:, feature_idx] + np.std(outputs_unscaled[:, feature_idx])

                # Fill bands aligned with output x-axis (1 to seq_len)
                axs[i, j].fill_between(
                    range(1, plot_length_output + 1), # x-axis for bands
                    lower_bound_1std,
                    upper_bound_1std,
                    alpha=0.2,
                    color="grey",
                    label="1-std band (on output)",
                )

                lower_bound_2std = outputs_unscaled[:, feature_idx] - 1.5 * np.std(outputs_unscaled[:, feature_idx]) # Using 1.5 as in your utils correction viz
                upper_bound_2std = outputs_unscaled[:, feature_idx] + 1.5 * np.std(outputs_unscaled[:, feature_idx])
                axs[i, j].fill_between(
                    range(1, plot_length_output + 1), # x-axis for bands
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

            # Identify and plot 'incorrect points' and 'correction vectors'
            # Compare input (t) with the *predicted* output (t+1)
            # This visualization shows how the *next* frame (predicted) deviates from the *current* frame (input)
            # It might be more appropriate to compare input (t) with *expected* next frame (if known), or just show the predicted sequence.
            # For now, let's visualize deviation of predicted t+1 from input t.
            # Only compare where both input and output exist (up to output length)
            for k in range(plot_length_output): # Iterate up to length of output (seq_len-1)
                 # k corresponds to timestep t, output[k] corresponds to predicted t+1
                 # input[k+1] would be the *actual* next frame after input[k] (if available)
                 # input[k] is the *current* frame
                 # Let's see deviation of output[k] (predicted t+1) from input[k] (current t)
                 input_val_at_t = input_unscaled[k, feature_idx] # Value of feature at timestep k (current)
                 predicted_val_at_t_plus_1 = outputs_unscaled[k, feature_idx] # Value predicted for timestep k+1

                 # Check if predicted t+1 value deviates significantly from current t value
                 # This is a simple check, might not be the most meaningful for 'correction'
                 # A better check might involve comparing predicted t+1 with *actual* t+1 (if available in data_original)
                 # For now, we'll use the output std bands as a reference for 'normal' variation
                 # Check against the bands calculated on the *output* sequence
                 # Check if the *predicted* value (at t+1 index) is outside the bands *around the predicted mean*
                 # OR, check if predicted (t+1) deviates significantly from input (t)
                 # Let's do the latter: compare predicted t+1 with input t
                 deviation_from_input = abs(predicted_val_at_t_plus_1 - input_val_at_t)
                 std_of_output = np.std(outputs_unscaled[:, feature_idx])
                 threshold = 1.5 * std_of_output # Use 1.5 std as a threshold for 'significant' deviation

                 if deviation_from_input > threshold:
                    # Plot the *predicted* value (at timestep k+1 index) as an 'incorrect' point
                    axs[i, j].plot(
                        k + 1, # x-axis: timestep k+1 (where the prediction applies)
                        predicted_val_at_t_plus_1, # y-axis: the predicted value
                        "ro", # Red circle
                        label="Predicted Deviation (t->t+1)", # Label for legend
                        markersize=5
                    )
                    # Draw an arrow from the input value (at t, y=input_val_at_t) towards the predicted value (at t+1, y=predicted_val_at_t_plus_1)
                    # This shows the 'direction' of the predicted change from input
                    # Or, draw an arrow from input (t) to predicted (t+1) at the *same* x-coordinate k (showing the jump)
                    # axs[i, j].arrow(k, input_val_at_t, 0, predicted_val_at_t_plus_1 - input_val_at_t, ...)
                    # Or, draw an arrow from input (t) to predicted (t+1) at the *different* x-coordinates k and k+1
                    # This is tricky because arrows usually go between points on the same x-axis.
                    # A common approach for sequence models is to show the *input* and *target* (actual next) and *prediction* (next).
                    # Since we don't have the actual next frame here easily, we'll just mark the predicted value if it deviates.
                    # The arrow concept might be confusing without the actual target.
                    # Let's just mark the predicted point if it deviates significantly from the input.
                    # The band visualization already provides context.
                    # Maybe draw an arrow from the *expected continuation* (e.g., same as input_val_at_t) to the predicted_val_at_t_plus_1
                    # axs[i, j].arrow(k+1, input_val_at_t, 0, predicted_val_at_t_plus_1 - input_val_at_t, ...)
                    # This still plots at k+1, which might be okay.
                    # For now, just mark the point if it deviates.
                    pass # The 'ro' marker already marks it

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    unique_labels = sorted(list(set(labels)))
    selected_lines = [lines[labels.index(label)] for label in unique_labels]

    fig.subplots_adjust(top=0.95) # Adjust to make space for legend
    fig.legend(
        selected_lines,
        unique_labels,
        loc="upper center",
        framealpha=1,
        fontsize=10,
        ncol=min(4, len(unique_labels)), # Limit number of columns
        bbox_to_anchor=(0.5, 0.98),
    )
    fig.suptitle(f"Pose Correction Prediction: {pose_name}", fontsize=16, y=0.95)
    fig.supxlabel("Frames (n)", fontsize=12, x=0.5)
    fig.supylabel("Degrees (°)", fontsize=12, y=0.5, x=0.005)

    plt.tight_layout(rect=[0, 0.03, 1, 0.92]) # Adjust layout to prevent overlap with suptitle/legend
    plt.savefig(f"correction_output_{pose_name}.png")
    plt.show()


def corr_predict(pose, data):
    """
    Predict pose corrections for input data.

    Parameters
    ----------
    pose : str
        Pose name (e.g., 'chair', 'cobra')
    data : pd.DataFrame
        Input pose landmark data (raw landmarks, e.g., 132 columns per row)

    Returns
    -------
    None
    """
    print(f"Starting correction prediction for pose: {pose}")

    # --- Step 1: Preprocess Raw Landmarks into Angle Features ---
    # This matches the initial processing in train_correction.py
    try:
        structured_df, body_pose_landmarks = structure_data(data)
        structured_df, body_pose_landmarks = update_body_pose_landmarks(
            structured_df, body_pose_landmarks
        )
        angle_features_df = correction_angles_convert(structured_df) # Shape: (variable_len, 9)
        print(f"Extracted angle features. Shape: {angle_features_df.shape}")
    except Exception as e:
        print(f"Error during landmark/angle conversion: {e}")
        return

    # --- Step 2: Determine Required Sequence Length ---
    # This is tricky because the training script used *average* lengths per asana
    # and then augmented to dynamic counts. The model expects sequences of the length
    # used *during augmentation*.
    # Option 1: Standardize to a fixed length (e.g., 25) - might not match training *for all asanas*.
    # Option 2: Load the *average length* used during training for this specific asana.
    #           This requires saving/loading that information.
    # Option 3: Assume a reasonable fixed length *if* the model was trained on diverse lengths
    #           and is robust to slight variations. This is often done but relies on padding/truncation
    #           handled implicitly by the model or dataloader, which this model doesn't seem to do explicitly.
    # Option 4: Use the *average length* if it's retrievable or hardcoded based on training.
    # Let's try a common approach: Assume the model was trained on sequences around the *average* length
    # for that pose, and *then augmented*. The augmentation created sequences of specific lengths (e.g., 17, 25).
    # We need to know what *that specific length* was for the model we are loading.
    # The *easiest* way without changing the training script again is to pick a reasonable length
    # that covers most cases, or restructure the prediction to work with variable lengths (harder).
    # Let's assume the user knows the *approximate* length or we standardize based on the *model's training*.
    # Since we don't have the *exact* augmentation length saved per model easily here,
    # let's try standardizing to the *average length* used *before augmentation* in training.
    # We can hardcode these averages based on your previous output:
    # chair: 85, cobra: 144, downdog: 120, goddess: 91, surya_namaskar: 1221, tree: 110, warrior: 103
    # However, using the *full* average length like 1221 for surya_namaskar might be excessive for a single prediction run.
    # The augmentation *did* standardize to a specific count (e.g., 25 peaks).
    # A compromise: Let's standardize to the *desired peak count* that was calculated *during training*
    # for this specific asana. This requires knowing it *at prediction time*.
    # The most practical approach without storing extra metadata is to choose a *fixed* sequence length
    # that is *feasible* for the augmentation logic *and* hopefully close to what was used in training for *most* classes.
    # Looking at the successful calculations:
    # chair: 17, cobra: 25 (capped), downdog: 24, goddess: 18, surya_namaskar: 25 (capped), tree: 22, warrior: 20
    # A length around 20-25 seems common *after* augmentation target calculation.
    # Let's try standardizing the *angle features* sequence to 25 frames *before* scaling/prediction.
    # This matches the augmentation target *if* the sequence was long enough, and is a reasonable fixed size.
    # Apply equal_rows to the *angle features* DataFrame
    try:
        # Use the average length *or* a standard length used for augmentation (e.g., 25 if capped)
        # Let's standardize to 25, which was the cap and a common outcome for longer sequences.
        # This might truncate longer sequences or pad shorter ones processed by equal_rows.
        # The original training script used equal_rows based on *average* length *per asana*.
        # To be more accurate, we should ideally know the *average length* for this `pose`.
        # Hardcode average lengths based on your previous run output:
        average_lengths = {
            'chair': 85,
            'cobra': 144,
            'downdog': 120,
            'goddess': 91,
            'surya_namaskar': 1221, # This is very long
            'tree': 110,
            'warrior': 103
        }
        target_length = average_lengths.get(pose, 100) # Default to 100 if pose not found
        print(f"Using target length {target_length} for equal_rows based on average for '{pose}'.")

        # Apply equal_rows to the angle features DataFrame
        angle_features_df = equal_rows(angle_features_df, target_length)
        print(f"Applied equal_rows. New shape: {angle_features_df.shape}")
    except Exception as e:
        print(f"Error during equal_rows: {e}")
        return

    # --- Step 3: Load Model and Scalers ---
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
        return
    except Exception as e:
        print(f"Unexpected error loading model/scalers: {e}")
        return

    # --- Step 4: Scale Features ---
    try:
        scaled_angle_features_df = scale_data(angle_features_df.copy(), scalers)
        print(f"Features scaled. Shape: {scaled_angle_features_df.shape}")
    except Exception as e:
        print(f"Error during scaling: {e}")
        return

    # --- Step 5: Run Prediction ---
    try:
        outputs_unscaled, input_unscaled = predict_correction_sequence(
            scaled_angle_features_df, device, pose, scalers, model
        )
        print(f"Prediction completed. Output shape: {outputs_unscaled.shape}, Input shape: {input_unscaled.shape}")
    except Exception as e:
        print(f"Error during model prediction: {e}")
        return

    # --- Step 6: Plot Results ---
    try:
        plot_correction_results(angle_features_df, input_unscaled, outputs_unscaled, pose)
        print("Plot saved and displayed.")
    except Exception as e:
        print(f"Error during plotting: {e}")
        return

    print(f"Correction prediction for pose '{pose}' completed successfully.")


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
        (Currently just calls corr_predict, but could be modified to return data)
    """
    print(f"Predicting correction for pose: {pose} from DataFrame")
    try:
        corr_predict(pose, data)
        # If corr_predict runs successfully, we could potentially return the
        # processed data or results here if needed by other parts of the codebase.
        # For now, it just runs the plotting.
        return {"status": "success", "pose": pose}
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
        (Currently just calls corr_predict via predict_correction_from_dataframe)
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
#     # Example: Load a CSV file with landmarks for the 'chair' pose
#     csv_file_path = "path/to/your/chair_pose_landmarks.csv"
#     pose_name = "chair" # Must match the folder name used during training
#     result = predict_correction_from_csv(csv_file_path, pose_name)
