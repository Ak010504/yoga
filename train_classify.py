"""
Training pipeline for classifier model.
"""

import os
import pickle
import warnings
import argparse
from itertools import combinations

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix

from classify_model import ClassifyPose
from utils import (
    cal_angle,
    cal_error,
    plot_confusion_matrix,
    plot_training_history,
    select_top_frames,
    structure_data,
    update_body_pose_landmarks,
)

warnings.filterwarnings("ignore")


def load_and_process_data(data_dir):
    """
    load individual CSV files and process them into sequences.

    Parameters
    ----------
    data_dir : str
        directory containing the dataset

    Returns
    -------
    list
        list of processed sequences (each sequence is a DataFrame)
    """
    all_sequences = []
    pose_classes = ["cobra", "tree", "goddess", "chair", "downdog", "warrior"]

    for pose in pose_classes:
        pose_dir = os.path.join(data_dir, pose)
        for cam in ["cam_0", "cam_1", "cam_2", "cam_3"]:
            cam_dir = os.path.join(pose_dir, cam)
            for file in os.listdir(cam_dir):
                if file.endswith(".csv"):
                    file_path = os.path.join(cam_dir, file)
                    df = pd.read_csv(file_path)

                    sequence = process_csv_to_sequence(
                        df, pose, file.replace(".csv", ""), cam
                    )
                    if sequence is not None:
                        all_sequences.append(sequence)

    return all_sequences


def process_csv_to_sequence(df, pose, person, camera):
    """
    process a single CSV file into a training sequence.

    Parameters
    ----------
    df : pd.DataFrame
        raw CSV data
    pose : str
        pose name
    person : str
        person identifier
    camera : str
        camera identifier

    Returns
    -------
    pd.DataFrame or None
        processed sequence or None if processing failed
    """
    try:
        pose_data = df.iloc[:, :132].copy()
        asana_label = df["asana"].iloc[0]

        df_structured, body_pose_landmarks = structure_data(pose_data)
        df_structured, body_pose_landmarks = update_body_pose_landmarks(
            df_structured, body_pose_landmarks
        )

        feature_df = pd.DataFrame()
        all_angles = list(combinations(body_pose_landmarks, 3))

        for i, angle_combo in enumerate(all_angles):
            feature_name = f"f{i+1}"
            feature_df[feature_name] = df_structured.apply(
                lambda x: cal_angle(
                    (x[angle_combo[0] + "_X"], x[angle_combo[0] + "_Y"]),
                    (x[angle_combo[1] + "_X"], x[angle_combo[1] + "_Y"]),
                    (x[angle_combo[2] + "_X"], x[angle_combo[2] + "_Y"]),
                ),
                axis=1,
            )

        data_with_error = cal_error(feature_df)
        selected_data = select_top_frames(data_with_error, 10)

        selected_data["asana"] = asana_label

        return selected_data

    except Exception as e:
        print(f"Error processing {pose}/{camera}/{person}: {e}")
        return None


def prepare_training_data(sequences, test_size=0.2):
    """
    prepare training and testing data for the LSTM model.

    Parameters
    ----------
    sequences : list
        list of processed sequences (DataFrames)
    test_size : float
        fraction of data to use for testing

    Returns
    -------
    tuple
        (train_loader, test_loader, scaler, pose_mapping)
    """
    pose_classes = ["cobra", "tree", "goddess", "chair", "downdog", "warrior"]
    pose_mapping = {pose: idx for idx, pose in enumerate(pose_classes)}

    all_features = []
    all_labels = []

    for sequence in sequences:
        feature_cols = [
            col for col in sequence.columns if col not in ["asana", "error"]
        ]
        features = sequence[feature_cols].values
        label = pose_mapping[sequence["asana"].iloc[0]]

        all_features.append(features)
        all_labels.extend([label] * len(features))

    # stack all features and labels first
    X = np.vstack(all_features)
    y = np.array(all_labels)

    # reshape to sequences BEFORE train/test split
    sequence_length = 10
    num_sequences = len(X) // sequence_length

    # truncate to ensure we have complete sequences
    X_truncated = X[: num_sequences * sequence_length]
    y_truncated = y[: num_sequences * sequence_length]

    # reshape to sequences
    X_sequences = X_truncated.reshape(num_sequences, sequence_length, -1)
    y_sequences = y_truncated[::sequence_length]  # take every 10th label

    print(f"DEBUG: Total sequences: {num_sequences}")
    print(f"DEBUG: Sequence shape: {X_sequences.shape}")
    print(f"DEBUG: Labels shape: {y_sequences.shape}")
    print(f"DEBUG: Samples per pose: {np.bincount(y_sequences)}")

    # now split sequences into train/test
    train_data, test_data = train_test_split(
        np.column_stack([X_sequences.reshape(num_sequences, -1), y_sequences]),
        test_size=test_size,
        stratify=y_sequences,
    )

    # extract features and labels
    train_features = train_data[:, :-1]
    train_labels = train_data[:, -1]
    test_features = test_data[:, :-1]
    test_labels = test_data[:, -1]

    print(f"DEBUG: Train sequences: {len(train_features)}")
    print(f"DEBUG: Test sequences: {len(test_features)}")

    # reshape back to sequence format
    train_features = train_features.reshape(-1, sequence_length, X_sequences.shape[2])
    test_features = test_features.reshape(-1, sequence_length, X_sequences.shape[2])

    # standardize features
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(
        train_features.reshape(-1, train_features.shape[2])
    )
    test_scaled = scaler.transform(test_features.reshape(-1, test_features.shape[2]))

    # reshape back to sequences
    train_scaled = train_scaled.reshape(train_features.shape)
    test_scaled = test_scaled.reshape(test_features.shape)

    # convert to tensors
    train_tensor = torch.tensor(train_scaled).float()
    test_tensor = torch.tensor(test_scaled).float()
    train_labels_tensor = torch.tensor(train_labels).long()
    test_labels_tensor = torch.tensor(test_labels).long()

    train_loader = DataLoader(
        TensorDataset(train_tensor, train_labels_tensor), batch_size=8, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(test_tensor, test_labels_tensor), batch_size=8, shuffle=False
    )

    return train_loader, test_loader, scaler, pose_mapping


def train_model(model, train_loader, test_loader, num_epochs=50, learning_rate=0.001):
    """
    train the LSTM model.

    Parameters
    ----------
    model : ClassifyPose
        the LSTM model
    train_loader : DataLoader
        training data loader
    test_loader : DataLoader
        test data loader
    num_epochs : int
        number of training epochs
    learning_rate : float
        learning rate for optimizer

    Returns
    -------
    tuple
        (model, training_history)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch_inputs, batch_labels in train_loader:
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(
                device
            )

            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == batch_labels).sum().item()
            total_samples += batch_inputs.size(0)

        avg_loss = total_loss / len(train_loader)
        train_accuracy = correct_predictions / total_samples

        train_losses.append(avg_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                test_correct += (predicted == labels).sum().item()
                test_total += labels.size(0)

        test_accuracy = test_correct / test_total
        test_accuracies.append(test_accuracy)

        print(
            f"Epoch {epoch+1}/{num_epochs}, "
            f"Train Loss: {avg_loss:.4f}, "
            f"Train Acc: {train_accuracy*100:.2f}%, "
            f"Test Acc: {test_accuracy*100:.2f}%"
        )

    training_history = {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "test_accuracies": test_accuracies,
    }

    return model, training_history


def evaluate_model(model, test_loader, pose_mapping):
    """
    evaluate the trained model and generate detailed metrics.

    Parameters
    ----------
    model : ClassifyPose
        trained model
    test_loader : DataLoader
        test data loader
    pose_mapping : dict
        mapping from pose names to indices

    Returns
    -------
    dict
        evaluation metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model = model.to(device)

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = sum(
        pred == label for pred, label in zip(all_predictions, all_labels)
    ) / len(all_labels)
    pose_names = list(pose_mapping.keys())
    report = classification_report(
        all_labels, all_predictions, target_names=pose_names, output_dict=True
    )

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "predictions": all_predictions,
        "labels": all_labels,
    }


def main():
    """main training pipeline."""
    parser = argparse.ArgumentParser(description="Train PosePilot LSTM model")
    parser.add_argument(
        "--data_dir", type=str, default="data", help="Directory containing dataset"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--test_size", type=float, default=0.1, help="Test set size fraction"
    )
    parser.add_argument("--hidden_size", type=int, default=32, help="LSTM hidden size")
    parser.add_argument(
        "--num_layers", type=int, default=1, help="Number of LSTM layers"
    )
    parser.add_argument(
        "--sequence_length", type=int, default=10, help="Sequence length"
    )
    parser.add_argument(
        "--desired_frames", type=int, default=10, help="Number of frames to select"
    )
    parser.add_argument("--no_plots", action="store_true", help="Skip plotting")
    parser.add_argument(
        "--use_cached_features",
        action="store_true",
        help="Load cached features instead of extracting",
    )

    args = parser.parse_args()

    print("Starting PosePilot training pipeline...")
    print(
        f"Configuration: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.learning_rate}"
    )

    data_dir = args.data_dir
    input_size = 680
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    sequence_length = args.sequence_length
    num_classes = 6
    num_epochs = args.epochs
    learning_rate = args.learning_rate
    test_size = args.test_size
    desired_frames = args.desired_frames

    processed_cache_path = os.path.join(data_dir, "cached_classify_features.pkl")

    if args.use_cached_features and os.path.exists(processed_cache_path):
        print("Loading cached processed sequences...")
        with open(processed_cache_path, "rb") as f:
            sequences = pickle.load(f)
        print(f"Loaded cached processed sequences: {len(sequences)} sequences")
    else:
        print("Loading and processing data...")
        sequences = load_and_process_data(data_dir)
        print(f"Processed {len(sequences)} sequences")

        print("Saving processed sequences to cache...")
        with open(processed_cache_path, "wb") as f:
            pickle.dump(sequences, f)
        print("Processed sequences cached successfully!")

    print("Preparing training data...")
    train_loader, test_loader, scaler, pose_mapping = prepare_training_data(sequences)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    print("Creating model...")
    model = ClassifyPose(
        input_size, hidden_size, num_layers, sequence_length, num_classes
    )
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    print("Training model...")
    trained_model, history = train_model(
        model, train_loader, test_loader, num_epochs, learning_rate
    )

    print("Evaluating model...")
    evaluation = evaluate_model(trained_model, test_loader, pose_mapping)

    print(f"\nFinal Test Accuracy: {evaluation['accuracy']*100:.2f}%")
    print("\nClassification Report:")
    print(
        classification_report(
            evaluation["labels"],
            evaluation["predictions"],
            target_names=list(pose_mapping.keys()),
        )
    )

    if not args.no_plots:
        plot_training_history(history)
        plot_confusion_matrix(
            evaluation["labels"], evaluation["predictions"], pose_mapping
        )
    torch.save(trained_model.state_dict(), "models/pose_classification_model.pth")
    with open("models/classify_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("models/pose_mapping.pkl", "wb") as f:
        pickle.dump(pose_mapping, f)

    print("\nModel and scaler saved successfully!")


if __name__ == "__main__":
    main()
