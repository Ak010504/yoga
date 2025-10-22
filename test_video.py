# classify_video.py

import pandas as pd
import torch
import pickle
import warnings
import numpy as np
from itertools import combinations
from classify_model import ClassifyPose
from utils import (
    cal_angle,
    cal_error,
    select_top_frames,
    structure_data,
    update_body_pose_landmarks,
    give_landmarks, # Import the landmark extraction function
)

warnings.filterwarnings("ignore")

def config_model(
    model_path="models/pose_classification_model.pth",
    scaler_path="models/classify_scaler.pkl",
    mapping_path="models/pose_mapping.pkl",
):
    """
    Configure the classifier model and load scaler and pose mapping.
    """
    with open(mapping_path, "rb") as f:
        pose_mapping = pickle.load(f)
    
    num_classes = len(pose_mapping)
    print(f"Loaded pose mapping with {num_classes} classes: {pose_mapping}")

    input_size = 680
    hidden_size = 32
    num_layers = 1
    sequence_length = 10

    model = ClassifyPose(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        sequence_length=sequence_length,
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler, pose_mapping

def feature_classify(data):
    """
    Extract the features from the data for classifier.
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
    final_df = select_top_frames(final_df)

    return final_df

def predict(data, model, scaler, pose_mapping):
    """
    Predict the class of the input data.
    """
    feature_df = feature_classify(data)
    feature_df.drop(columns=["error"], inplace=True)

    if len(feature_df) != 10:
        print(f"Warning: Expected 10 frames after select_top_frames, got {len(feature_df)}.")
        if len(feature_df) < 10:
             raise ValueError(f"Insufficient frames after select_top_frames: got {len(feature_df)}, expected 10.")

    features_np = feature_df.values
    features_scaled_np = scaler.transform(features_np)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    input_feature = torch.tensor(features_scaled_np, dtype=torch.float32).unsqueeze(0).to(device)

    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(input_feature)
        _, predicted_tensor = torch.max(outputs.data, 1)
        predicted_class = predicted_tensor.item()

    predicted_pose_name = None
    for name, idx in pose_mapping.items():
        if idx == predicted_class:
            predicted_pose_name = name
            break

    if predicted_pose_name is None:
        raise ValueError(f"Could not find pose name for predicted class index {predicted_class}")

    return predicted_pose_name

def classify_video(video_path, fps=10):
    """
    Extract landmarks from video and predict the pose.
    """
    print(f"--- Step 1: Extracting Landmarks from {video_path} ---")
    try:
        df_landmarks, landmark_list = give_landmarks(video_path, label="unknown", fps=fps)
        print(f"Extracted {len(df_landmarks)} frames of landmarks.")
    except Exception as e:
        print(f"Error during landmark extraction: {e}")
        return None

    print("\n--- Step 2: Loading Model and Scaler ---")
    try:
        model, scaler, pose_mapping = config_model()
    except FileNotFoundError as e:
        print(f"Error loading model/scaler/mapping files: {e}. Ensure 'models/' directory exists and contains the required files.")
        return None
    except Exception as e:
        print(f"Error during model configuration: {e}")
        return None

    print("\n--- Step 3: Classifying Pose ---")
    try:
        predicted_pose = predict(df_landmarks, model, scaler, pose_mapping)
        print(f"\nPredicted Pose: {predicted_pose}")
        return predicted_pose
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

if __name__ == "__main__":
    # --- Configure these paths and parameters ---
    video_path = "D:\\Amrit\\College\\SSN\\CS\\PosePilot\\assets\\test_vdo4.mp4"  # Replace with your video path
    fps = 10                                      # Frames per second to extract
    # --- End of configuration ---

    print(f"Starting classification for video: {video_path}")
    result = classify_video(video_path, fps)
    if result:
        print(f"Classification completed successfully. Pose: {result}")
    else:
        print("Classification failed.")
