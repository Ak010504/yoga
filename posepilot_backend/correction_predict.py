"""
pose correction prediction pipeline.
Updated to handle 7 classes and match the training sequence length logic.
FIXED: Now uses angle-based velocity gating and returns proper correction data.
"""

import torch
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import label
from correction_model import CorrModel
from utils import (
    cal_error,
    correction_angles_convert,
    equal_rows,
    structure_data,
    update_body_pose_landmarks,
)

warnings.filterwarnings("ignore")


def calculate_angle_velocity(angle_features_df, velocity_threshold=2.0, min_phase_duration=3):
    """
    Calculate velocity on ANGLE FEATURES (f1-f9) instead of raw landmarks.
    This is more meaningful for detecting stable yoga poses.
    FIXES the 132/133 element mismatch by working on angle features directly.
    
    Parameters
    ----------
    angle_features_df : pd.DataFrame
        DataFrame with angle features (f1-f9 columns)
    velocity_threshold : float
        Threshold for angular velocity (degrees/frame) to classify as "stable"
    min_phase_duration : int
        Minimum consecutive frames to consider a phase
    
    Returns
    -------
    tuple
        (velocities_smooth: array, phase_midpoints: array)
    """
    
    # Get only the angle feature columns (f1-f9)
    feature_cols = [col for col in angle_features_df.columns if col.startswith('f') and col[1:].isdigit()]
    
    if len(feature_cols) == 0:
        print("ERROR: No angle feature columns found in DataFrame")
        return np.array([]), np.array([])
    
    angle_data = angle_features_df[feature_cols].values  # Shape: (frames, 9)
    
    print(f"[Angle Velocity] Processing {len(angle_data)} frames with {len(feature_cols)} angle features")
    
    # Initialize velocities
    velocities = np.zeros(len(angle_data))
    
    # Calculate frame-to-frame angular change (velocity)
    for i in range(1, len(angle_data)):
        # Calculate mean absolute angular change across all 9 features
        angular_changes = np.abs(angle_data[i] - angle_data[i-1])
        velocities[i] = np.mean(angular_changes)
    
    # Smooth velocities using Savitzky-Golay filter
    window_length = min(5, max(3, len(velocities) // 10))
    if window_length % 2 == 0:
        window_length += 1
    
    try:
        velocities_smooth = savgol_filter(velocities, window_length=window_length, polyorder=2)
    except Exception as e:
        print(f"Warning: Savgol filter failed, using raw velocities: {e}")
        velocities_smooth = velocities
    
    # Find stable regions (angular velocity < threshold)
    stable_mask = velocities_smooth < velocity_threshold
    
    # Label consecutive stable regions
    labeled_regions, num_phases = label(stable_mask)
    
    # Extract MIDPOINT of each stable phase
    phase_midpoints = []
    for phase_id in range(1, num_phases + 1):
        phase_frames = np.where(labeled_regions == phase_id)[0]
        
        if len(phase_frames) >= min_phase_duration:
            mid_frame = phase_frames[len(phase_frames) // 2]
            phase_midpoints.append(int(mid_frame))
    
    print(f"[Angle Velocity Gating] Detected {len(phase_midpoints)} stable phases")
    print(f"  - Angular velocity threshold: {velocity_threshold}°/frame")
    print(f"  - Min phase duration: {min_phase_duration} frames")
    print(f"  - Total frames in stable regions: {np.sum(stable_mask)}/{len(velocities_smooth)}")
    
    return velocities_smooth, np.array(phase_midpoints)


def pad_with_linear_interpolation(df, target_length):
    """
    Pad by extrapolating linearly based on recent trend.
    """
    current_length = len(df)
    
    if current_length >= target_length:
        return df.iloc[:target_length].reset_index(drop=True)
    
    padding_needed = target_length - current_length
    numeric_cols = [col for col in df.columns if col.startswith('f')]
    
    padded_rows = []
    
    # Calculate trend from last 5 frames
    if current_length >= 5:
        recent_frames = df[numeric_cols].iloc[-5:].values
        # Slope = (last - first) / 4
        slope = (recent_frames[-1] - recent_frames[0]) / 4
        last_values = df[numeric_cols].iloc[-1].values
    else:
        slope = np.zeros(len(numeric_cols))
        last_values = df[numeric_cols].iloc[-1].values
    
    # Extrapolate forward
    for i in range(1, padding_needed + 1):
        new_values = last_values + (slope * i * 0.5)  # 0.5 = damping factor
        padded_rows.append(new_values)
    
    padded_df = pd.DataFrame(padded_rows, columns=numeric_cols)
    
    # Add back other columns
    for col in df.columns:
        if col not in numeric_cols:
            padded_df[col] = df[col].iloc[-1]
    
    padded_df = padded_df[df.columns]
    
    result = pd.concat([df, padded_df], ignore_index=True)
    return result.iloc[:target_length].reset_index(drop=True)


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


import os
import torch
import pickle
from correction_model import CorrModel


def load_model_and_scalers(device, pose):
    # Absolute path to this file's directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Go one level up → PosePilot/
    PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

    model_path = os.path.join(
        PROJECT_ROOT, "models", f"{pose}_correction_model.pth"
    )
    scalers_path = os.path.join(
        PROJECT_ROOT, "models", f"{pose}_correction_scalers.pkl"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not os.path.exists(scalers_path):
        raise FileNotFoundError(f"Scalers not found: {scalers_path}")

    model = CorrModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)

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


def corr_predict(pose, data, return_data=False):
    """
    Predict pose corrections for a SINGLE frame.
    (Velocity gating is done in main.py on the full video)
    """
    print(f"Starting correction prediction for pose: {pose}")

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

    # Pad to target length
    try:
        average_lengths = {
            'chair': 85, 'cobra': 144, 'downdog': 120, 'goddess': 91,
            'surya_namaskar': 122, 'tree': 110, 'warrior': 103
        }
        target_length = average_lengths.get(pose, 100)
        
        if len(angle_features_df) < target_length:
            angle_features_df = pad_with_linear_interpolation(
                angle_features_df, target_length
            )
            print(f"  Padded to {target_length} frames")
        elif len(angle_features_df) > target_length:
            angle_features_df = angle_features_df.iloc[:target_length].reset_index(drop=True)
            print(f"  Truncated to {target_length} frames")
            
    except Exception as e:
        print(f"Error during padding: {e}")
        return None

    # Load model and run prediction
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, scalers = load_model_and_scalers(device, pose)
        
        scaled_df = scale_data(angle_features_df.copy(), scalers)
        outputs_unscaled, input_unscaled = predict_correction_sequence(
            scaled_df, device, pose, scalers, model
        )
        
        if return_data:
            return {
                "status": "success",
                "pose": pose,
                "input_data": input_unscaled,
                "corrected_data": outputs_unscaled,
                "reference_data": angle_features_df,
            }
        return None
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None


def predict_correction_from_dataframe(df, pose):
    """
    Predict correction for a pose from a raw landmark DataFrame.
    Uses last WINDOW_SIZE frames only (TCN-compatible).
    """

    WINDOW_SIZE = 30

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # -----------------------------
        # Load model + scalers
        # -----------------------------
        model, scalers = load_model_and_scalers(device, pose)

        # -----------------------------
        # Extract angle features
        # -----------------------------
        pose_data = df.iloc[:, :132].copy()

        structured_df, body_pose_landmarks = structure_data(pose_data)
        structured_df, body_pose_landmarks = update_body_pose_landmarks(
            structured_df, body_pose_landmarks
        )

        angle_df = correction_angles_convert(structured_df)
        angle_df = angle_df[[f"f{i}" for i in range(1, 10)]]

        print(f"Extracted angle features. Shape: {angle_df.shape}")

        # -----------------------------
        # Sliding window selection
        # -----------------------------
        if len(angle_df) < WINDOW_SIZE:
            return {
                "status": "error",
                "error": f"Not enough frames ({len(angle_df)}) for window size {WINDOW_SIZE}"
            }

        window = angle_df.values[-WINDOW_SIZE:]

        # -----------------------------
        # Normalize
        # -----------------------------
        for i, scaler in enumerate(scalers):
            window[:, i] = scaler.transform(
                window[:, i].reshape(-1, 1)
            ).flatten()

        input_tensor = (
            torch.tensor(window, dtype=torch.float32)
            .unsqueeze(0)
            .to(device)
        )

        # -----------------------------
        # Predict
        # -----------------------------
        with torch.no_grad():
            correction = model(input_tensor).cpu().numpy()[0]

        # -----------------------------
        # Inverse normalize
        # -----------------------------
        for i, scaler in enumerate(scalers):
            correction[i] = scaler.inverse_transform(
                [[correction[i]]]
            )[0][0]

        return {
            "status": "success",
            "pose": pose,
            "correction": {
                f"f{i+1}": float(correction[i]) for i in range(9)
            }
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }




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