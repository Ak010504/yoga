"""
pose classification prediction pipeline.
"""

import os
import torch
import pickle
import warnings
import pandas as pd
from itertools import combinations
from classify_model import ClassifyPose
from sklearn.preprocessing import StandardScaler
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
    final_df = select_top_frames(final_df)

    return final_df


def config_model(
    model_path="models/pose_classification_model.pth",
    scaler_path="models/classify_scaler.pkl",
    mapping_path="models/pose_mapping.pkl",
):
    """
    configure the classifier model and load scaler and pose mapping.

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
    input_size = 680
    hidden_size = 32
    num_layers = 1
    num_classes = 6
    sequence_length = 10

    model = ClassifyPose(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        sequence_length=sequence_length,
    )
    model.load_state_dict(torch.load(model_path))

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    with open(mapping_path, "rb") as f:
        pose_mapping = pickle.load(f)

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

    feature_df_scaled = scaler.transform(feature_df.values)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_feature = torch.tensor(feature_df_scaled, dtype=torch.float32).to(device)
    input_feature = input_feature.unsqueeze(0)

    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(input_feature)
        _, predicted = torch.max(outputs.data, 1)

    predicted_class = predicted.item()
    pose_name = list(pose_mapping.keys())[
        list(pose_mapping.values()).index(predicted_class)
    ]

    return pose_name


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
