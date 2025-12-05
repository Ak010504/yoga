"""
pose classification prediction pipeline.
Modified to handle 7 classes and correct sequence length.
"""

import torch
import pickle
import warnings
import pandas as pd
import numpy as np
from itertools import combinations
from classify_model import ClassifyPose
from utils import (
    cal_angle,
    cal_error,
    select_top_frames,
    structure_data,
    update_body_pose_landmarks,
)

warnings.filterwarnings("ignore")

def feature_classify(data):
    """
    extract the features from the data for classifier.

    Parameters
    ----------
    data : pd.DataFrame
        the DataFrame containing the body pose landmarks

    Returns
    -------
    pd.DataFrame
        the DataFrame containing the features for classification
    """
    feature_df, body_pose_landmarks = structure_data(data)
    feature_df, body_pose_landmarks = update_body_pose_landmarks(
        feature_df, body_pose_landmarks
    )

    mapping = {}
    all_angles = list(combinations(body_pose_landmarks, 3))

    final_df = pd.DataFrame()

    for i in range(len(all_angles)):
        final_df["f" + str(i + 1)] = feature_df.apply(
            lambda x: cal_angle(
                (x[all_angles[i][0] + "_X"], x[all_angles[i][0] + "_Y"]),
                (x[all_angles[i][1] + "_X"], x[all_angles[i][1] + "_Y"]),
                (x[all_angles[i][2] + "_X"], x[all_angles[i][2] + "_Y"]),
            ),
            axis=1,
        )
        mapping["f" + str(i + 1)] = all_angles[i]

    final_df = cal_error(final_df)
    final_df = select_top_frames(final_df) # This should return 10 rows per sequence

    return final_df


def config_model(
    model_path="models/pose_classification_model.pth",
    scaler_path="models/classify_scaler.pkl",
    mapping_path="models/pose_mapping.pkl",
):
    """
    configure the classifier model and load scaler and pose mapping.
    Updated to use dynamic num_classes and sequence_length based on training.

    Parameters
    ----------
    model_path : str
        path to the trained model file
    scaler_path : str
        path to the scaler file
    mapping_path : str
        path to the pose mapping file

    Returns
    -------
    tuple
        tuple containing (model, scaler, pose_mapping)
    """
    # Load the pose_mapping first to get num_classes
    with open(mapping_path, "rb") as f:
        pose_mapping = pickle.load(f)
    
    num_classes = len(pose_mapping) # Dynamically determine number of classes
    print(f"Loaded pose mapping with {num_classes} classes: {pose_mapping}")

    # Use the same input_size, hidden_size, num_layers, sequence_length as in training
    input_size = 680  # Ensure this matches the training script
    hidden_size = 32  # Ensure this matches the training script
    num_layers = 1    # Ensure this matches the training script
    sequence_length = 10 # Ensure this matches the training script

    model = ClassifyPose(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        sequence_length=sequence_length,
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu')) # Load on CPU initially

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # pose_mapping is already loaded above
    return model, scaler, pose_mapping


def predict(data, model, scaler, pose_mapping):
    """
    predict the class of the input data.

    Parameters
    ----------
    data : pd.DataFrame
        the DataFrame containing the pose landmarks
    model : ClassifyPose
        the trained model
    scaler : StandardScaler
        the fitted scaler
    pose_mapping : dict
        the pose name to numeric label mapping

    Returns
    -------
    str
        the predicted pose name
    """
    feature_df = feature_classify(data)
    feature_df.drop(columns=["error"], inplace=True)

    # Ensure feature_df has the expected number of rows (sequence_length)
    if len(feature_df) != 10: # sequence_length from training
        print(f"Warning: Expected 10 frames after select_top_frames, got {len(feature_df)}.")
        # You might want to add logic here to handle variable lengths if necessary
        # For now, we'll proceed assuming it's 10 or handle accordingly
        # If it's less than 10, you might need to pad or select differently
        # If it's more than 10, select_top_frames should handle it
        if len(feature_df) < 10:
             raise ValueError(f"Insufficient frames after select_top_frames: got {len(feature_df)}, expected 10.")
        # If it's more, it's likely an error in select_top_frames or input data structure
        # Assuming select_top_frames correctly returns 10 for a sequence input

    # Prepare features for the model
    features_np = feature_df.values # Shape: (sequence_length, num_features)
    # print(f"Feature shape before scaling: {features_np.shape}") # Debug print

    # Scale features - fit_transform was used during training on the *entire* sequence array,
    # but scaler.transform is used here on the *single sequence* array.
    # The scaler must have been fitted on data with the same feature dimension.
    features_scaled_np = scaler.transform(features_np) # Shape: (sequence_length, num_features)
    # print(f"Feature shape after scaling: {features_scaled_np.shape}") # Debug print

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # print(f"Using device: {device}") # Debug print

    # Prepare input tensor for LSTM: (batch_size, sequence_length, input_size)
    # For a single prediction, batch_size is 1
    input_feature = torch.tensor(features_scaled_np, dtype=torch.float32).unsqueeze(0).to(device) # Shape: (1, sequence_length, num_features)
    # print(f"Input feature shape: {input_feature.shape}") # Debug print

    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(input_feature) # Shape: (1, num_classes)
        # print(f"Model output shape: {outputs.shape}") # Debug print
        _, predicted_tensor = torch.max(outputs.data, 1) # Shape: (1,)
        predicted_class = predicted_tensor.item() # Get the scalar value

    # Map the predicted class index back to the pose name
    # The pose_mapping maps names to indices, so find the name corresponding to the predicted index
    predicted_pose_name = None
    for name, idx in pose_mapping.items():
        if idx == predicted_class:
            predicted_pose_name = name
            break

    if predicted_pose_name is None:
        raise ValueError(f"Could not find pose name for predicted class index {predicted_class}")

    return predicted_pose_name


def predict_from_csv(
    csv_path,
    model_path="models/pose_classification_model.pth",
    scaler_path="models/classify_scaler.pkl",
    mapping_path="models/pose_mapping.pkl",
):
    """
    predict pose from a CSV file.

    Parameters
    ----------
    csv_path : str
        path to the CSV file containing pose landmarks
    model_path : str
        path to the trained model file
    scaler_path : str
        path to the scaler file
    mapping_path : str
        path to the pose mapping file

    Returns
    -------
    str
        the predicted pose name
    """
    data = pd.read_csv(csv_path)
    model, scaler, pose_mapping = config_model(model_path, scaler_path, mapping_path)

    return predict(data, model, scaler, pose_mapping)


def predict_from_dataframe(
    data,
    model_path="models/pose_classification_model.pth",
    scaler_path="models/classify_scaler.pkl",
    mapping_path="models/pose_mapping.pkl",
):
    """
    predict pose from a DataFrame directly.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing pose landmarks
    model_path : str
        path to the trained model file
    scaler_path : str
        path to the scaler file
    mapping_path : str
        path to the pose mapping file

    Returns
    -------
    str
        the predicted pose name
    """
    model, scaler, pose_mapping = config_model(model_path, scaler_path, mapping_path)

    return predict(data, model, scaler, pose_mapping)
