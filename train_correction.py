"""
Training pipeline for pose correction model.
"""

import os
import pickle
import random
import warnings
import argparse

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from correction_model import CorrModel
from utils import (
    cal_error,
    correction_angles_convert,
    equal_rows,
    structure_data,
    update_body_pose_landmarks,
)

warnings.filterwarnings("ignore")


def load_all_asana_data(data_dir):
    """
    load all CSV files from all asanas and camera subdirectories.

    Parameters
    ----------
    data_dir : str
        path to the data directory

    Returns
    -------
    dict
        dictionary with asana names as keys and lists of DataFrames as values
    """
    all_data = {}

    # Get all asana directories
    asana_dirs = [
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    ]

    for asana_name in asana_dirs:
        asana_path = os.path.join(data_dir, asana_name)
        dataframes = []

        # Load from camera subdirectories
        subdirs = [
            d
            for d in os.listdir(asana_path)
            if os.path.isdir(os.path.join(asana_path, d))
        ]

        for subdir in subdirs:
            subdir_path = os.path.join(asana_path, subdir)
            csv_files = [f for f in os.listdir(subdir_path) if f.endswith(".csv")]

            for csv_file in csv_files:
                csv_path = os.path.join(subdir_path, csv_file)
                df = pd.read_csv(csv_path)
                dataframes.append(df)

        all_data[asana_name] = dataframes
        print(f"Total loaded {len(dataframes)} CSV files for asana: {asana_name}")

    return all_data


def calculate_asana_averages(asana_data):
    """
    calculate average frame count for each asana.

    Parameters
    ----------
    asana_data : dict
        dictionary with asana names as keys and lists of DataFrames as values

    Returns
    -------
    dict
        dictionary with asana names as keys and average frame counts as values
    """
    asana_averages = {}

    for asana_name, dataframes in asana_data.items():
        frame_counts = []
        for df in dataframes:
            frame_count = len(df)
            frame_counts.append(frame_count)

        avg_frames = int(np.mean(frame_counts))
        asana_averages[asana_name] = avg_frames
        print(f"{asana_name}: {len(dataframes)} sequences, avg {avg_frames} frames")

    return asana_averages


def augment_data_with_peaks(
    dataframes, asana_name, asana_averages, desired_peak_count=25
):
    """
    augment data using peak selection method (3x augmentation) on each sequence.

    Parameters
    ----------
    dataframes : list
        list of DataFrames, each representing one sequence
    asana_name : str
        name of the asana (used to get the average frame count)
    asana_averages : dict
        dictionary with asana names as keys and average frame counts as values
    desired_peak_count : int
        number of peaks to select per sequence

    Returns
    -------
    list
        list of augmented DataFrames (3x the original size)
    """
    augmented_dataframes = []

    for i, df in enumerate(dataframes):
        df_peaks_0 = pd.DataFrame()
        df_peaks_1 = pd.DataFrame()
        df_peaks_2 = pd.DataFrame()

        peaks_1 = np.linspace(1, len(df) - 2, num=desired_peak_count, dtype=int)
        peaks_2 = [i + 1 for i in peaks_1]
        peaks_0 = [i - 1 for i in peaks_1]

        df_peaks_1 = pd.concat([df_peaks_1, df.iloc[peaks_1]])
        df_peaks_2 = pd.concat([df_peaks_2, df.iloc[peaks_2]])
        df_peaks_0 = pd.concat([df_peaks_0, df.iloc[peaks_0]])

        augmented_dataframes.extend([df_peaks_1, df_peaks_0, df_peaks_2])

    return augmented_dataframes


def collect_all_data_for_global_scaling(asana_data, asana_averages):
    """
    collect all data first to fit global scalers (like notebook approach).

    Parameters
    ----------
    asana_data : dict
        dictionary with asana names as keys and lists of DataFrames as values
    asana_averages : dict
        dictionary with asana names as keys and average frame counts as values

    Returns
    -------
    tuple
        tuple containing (all_angle_data, processed_dataframes)
    """
    all_angle_data = []
    processed_dataframes = {}

    for asana_name, dataframes in asana_data.items():
        asana_processed = []

        for i, df in enumerate(tqdm(dataframes, desc=f"Processing {asana_name}")):
            pose_data = df.iloc[:, :132].copy()
            structured_df, body_pose_landmarks = structure_data(pose_data)
            structured_df, body_pose_landmarks = update_body_pose_landmarks(
                structured_df, body_pose_landmarks
            )
            angle_df = correction_angles_convert(structured_df)
            angle_df = cal_error(angle_df)
            angle_df = equal_rows(angle_df, asana_averages[asana_name])
            all_angle_data.append(angle_df)
            asana_processed.append(angle_df)

        processed_dataframes[asana_name] = asana_processed

    combined_data = pd.concat(all_angle_data, ignore_index=True)
    return combined_data, processed_dataframes


def normalize_with_global_scalers(all_data, processed_dataframes, asana_averages):
    """
    fit global scalers on all data, then apply to individual sequences (like notebook).

    Parameters
    ----------
    all_data : pd.DataFrame
        combined data from all sequences for fitting scalers
    processed_dataframes : dict
        dictionary with processed dataframes per asana
    asana_averages : dict
        dictionary with asana names as keys and average frame counts as values

    Returns
    -------
    tuple
        tuple containing (normalized_data, scalers)
    """
    global_scalers = []
    for i in range(9):
        scaler = MinMaxScaler()
        scaler.fit(all_data.iloc[:, i].values.reshape(-1, 1))
        global_scalers.append(scaler)

    normalized_data = {}

    for asana_name, dataframes in processed_dataframes.items():
        normalized_dataframes = []
        for df in tqdm(dataframes, desc=f"Normalizing {asana_name}"):
            normalized_df = df.copy()
            for i in range(9):
                normalized_df.iloc[:, i] = global_scalers[i].transform(
                    normalized_df.iloc[:, i].values.reshape(-1, 1)
                )
            normalized_dataframes.append(normalized_df)
        augmented_dataframes = augment_data_with_peaks(
            normalized_dataframes, asana_name, asana_averages, desired_peak_count=25
        )
        normalized_data[asana_name] = augmented_dataframes

    return normalized_data, global_scalers


def prepare_correction_data(normalized_data, asana_name, asana_averages):
    """
    prepare data for correction model training using sliding window approach with augmentation.

    Parameters
    ----------
    normalized_data : dict
        dictionary with normalized DataFrames for each asana
    asana_name : str
        name of the asana to prepare data for
    asana_averages : dict
        dictionary with asana names as keys and average frame counts as values

    Returns
    -------
    tuple
        tuple containing (train_tensor, train_labels, test_tensor, test_labels)
    """
    dataframes = normalized_data[asana_name]
    feature_columns = [f"f{i}" for i in range(1, 10)]

    all_sequences = []
    for df in dataframes:
        sequence_data = df[feature_columns].values
        all_sequences.append(sequence_data)
    data_input = np.array(all_sequences, dtype=np.float32)

    num_sequences = len(all_sequences)
    train_test_split = 0.2
    all_sequence_indices = list(range(num_sequences))
    test_indices = random.sample(
        all_sequence_indices, int(train_test_split * num_sequences)
    )
    train_indices = [i for i in all_sequence_indices if i not in test_indices]

    train_data = data_input[train_indices]
    test_data = data_input[test_indices]

    train_tensor = torch.tensor(train_data, dtype=torch.float32)
    test_tensor = torch.tensor(test_data, dtype=torch.float32)
    train_labels = train_tensor[:, 1:, :]
    test_labels = test_tensor[:, 1:, :]
    train_tensor = train_tensor[:, :-1, :]
    test_tensor = test_tensor[:, :-1, :]

    return train_tensor, train_labels, test_tensor, test_labels


def train_correction_model(
    model, train_loader, test_loader, num_epochs=30, learning_rate=0.001
):
    """
    train the correction model.

    Parameters
    ----------
    model : nn.Module
        the correction model to train
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
        tuple containing (train_losses, test_losses)
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    test_losses = []

    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target.view(-1, 9))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Test (matching notebook approach - process only first sample from each batch)
        model.eval()
        test_loss = 0.0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data[0:1])
                loss = criterion(output, target[0:1].view(-1, 9))
                test_loss += loss.item()

        train_loss /= len(train_loader)
        test_loss /= len(test_loader)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(
            f"Epoch {epoch+1:3d}/{num_epochs}: Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}"
        )

    return train_losses, test_losses


def save_model_and_scalers(model, scalers, asana_name, output_dir):
    """
    save trained model and scalers.

    Parameters
    ----------
    model : nn.Module
        trained model
    scalers : list
        list of feature scalers used for normalization (one per feature)
    asana_name : str
        name of the asana
    output_dir : str
        directory to save the model and scalers
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(output_dir, f"{asana_name}_correction_model.pth")
    torch.save(model.state_dict(), model_path)

    # Save scalers
    scalers_path = os.path.join(output_dir, f"{asana_name}_correction_scalers.pkl")
    with open(scalers_path, "wb") as f:
        pickle.dump(scalers, f)

    print(f"Saved model and scalers for {asana_name}")


def main():
    """main training function."""
    parser = argparse.ArgumentParser(description="Train pose correction models")
    parser.add_argument(
        "--data_dir", type=str, default="data", help="Data directory path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size")
    parser.add_argument(
        "--num_layers", type=int, default=1, help="Number of LSTM layers"
    )
    parser.add_argument(
        "--use_cached_features",
        action="store_true",
        help="Use cached processed features",
    )

    args = parser.parse_args()

    print("Starting correction training for all asanas")

    # Check for cached features
    cache_file = os.path.join(args.data_dir, "cached_correction_features.pkl")

    if args.use_cached_features and os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            cached_data = pickle.load(f)
        normalized_data = cached_data["normalized_data"]
        scalers = cached_data["scalers"]
        asana_averages = cached_data["asana_averages"]
    else:
        # Load data for all asanas
        asana_data = load_all_asana_data(args.data_dir)

        # Calculate average frames for each asana
        asana_averages = calculate_asana_averages(asana_data)

        all_data, processed_dataframes = collect_all_data_for_global_scaling(
            asana_data, asana_averages
        )
        normalized_data, scalers = normalize_with_global_scalers(
            all_data, processed_dataframes, asana_averages
        )

        cached_data = {
            "normalized_data": normalized_data,
            "scalers": scalers,
            "asana_averages": asana_averages,
        }
        with open(cache_file, "wb") as f:
            pickle.dump(cached_data, f)

    print("Data preprocessing completed!")

    # Train models for all poses
    input_size = 9
    num_output_features = 9

    for asana_name in normalized_data.keys():
        print(f"\n{'='*50}")
        print(f"Training correction model for: {asana_name}")
        print(f"{'='*50}")

        train_tensor, train_labels, test_tensor, test_labels = prepare_correction_data(
            normalized_data, asana_name, asana_averages
        )
        train_dataset = TensorDataset(train_tensor, train_labels)
        test_dataset = TensorDataset(test_tensor, test_labels)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

        model = CorrModel(
            input_size=input_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_classes=num_output_features,
        )

        train_losses, test_losses = train_correction_model(
            model, train_loader, test_loader, args.epochs, args.learning_rate
        )
        save_model_and_scalers(model, scalers, asana_name, args.output_dir)

        print(f"Completed training for {asana_name}")
        print(f"Final train loss: {train_losses[-1]:.6f}")
        print(f"Final test loss: {test_losses[-1]:.6f}")

    print(f"\n{'='*50}")
    print("All correction models trained successfully!")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
