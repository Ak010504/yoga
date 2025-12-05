"""
Training pipeline for classifier model.
Modified to include Surya Namaskar, updated folder structure,
and optional random-search hyperparameter tuning.
"""

import os
import pickle
import warnings
import argparse
from itertools import combinations
import random  # NEW: for random search

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

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
    # Updated pose classes to include Surya Namaskar (use folder name as it appears)
    pose_classes = ["cobra", "tree", "goddess", "chair", "downdog", "warrior", "surya_namaskar"]

    for pose in tqdm(pose_classes, desc="Processing asanas"):
        pose_dir = os.path.join(data_dir, pose)

        # Check if directory exists
        if not os.path.exists(pose_dir):
            print(f"Warning: Directory {pose_dir} not found. Skipping {pose}...")
            continue

        file_count = 0
        # Process all CSV files directly in the pose directory (no cam subfolders)
        for file in os.listdir(pose_dir):
            if file.endswith(".csv"):
                file_path = os.path.join(pose_dir, file)
                try:
                    df = pd.read_csv(file_path)

                    sequence = process_csv_to_sequence(
                        df, pose, file.replace(".csv", "")
                    )
                    if sequence is not None:
                        all_sequences.append(sequence)
                        file_count += 1
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue

        print(f"  Loaded {file_count} files for {pose}")

    print(f"\nTotal sequences loaded: {len(all_sequences)}")
    return all_sequences


def process_csv_to_sequence(df, pose, person):
    """
    process a single CSV file into a training sequence.

    Parameters
    ----------
    df : pd.DataFrame
        raw CSV data
    pose : str
        pose name (folder name)
    person : str
        person identifier (filename without extension)

    Returns
    -------
    pd.DataFrame or None
        processed sequence or None if processing failed
    """
    try:
        # Extract pose landmark data (first 132 columns: 33 landmarks × 4 values each)
        pose_data = df.iloc[:, :132].copy()

        # Get asana label - check if 'asana' column exists, otherwise use folder name
        if "asana" in df.columns:
            asana_label = df["asana"].iloc[0]
        else:
            # Use the folder name as the label
            asana_label = pose

        # Normalize the label (handle spaces and case)
        asana_label = str(asana_label).lower().strip().replace(" ", "_")

        # Structure the data and get landmark names
        df_structured, body_pose_landmarks = structure_data(pose_data)
        df_structured, body_pose_landmarks = update_body_pose_landmarks(
            df_structured, body_pose_landmarks
        )

        # Calculate all angle features from landmark combinations
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

        # Calculate error and select top frames
        data_with_error = cal_error(feature_df)
        selected_data = select_top_frames(data_with_error, 10)

        # Add the asana label
        selected_data["asana"] = asana_label

        return selected_data

    except Exception as e:
        print(f"Error processing {pose}/{person}: {e}")
        return None


def prepare_training_data(sequences, test_size=0.2):
    """
    prepare training and testing data for the LSTM model.
    (Original version – uses fixed batch size later)

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
    # Updated pose classes to include Surya Namaskar
    pose_classes = ["cobra", "tree", "goddess", "chair", "downdog", "warrior", "surya_namaskar"]
    pose_mapping = {pose: idx for idx, pose in enumerate(pose_classes)}

    print(f"\nPose mapping: {pose_mapping}")

    all_features = []
    all_labels = []

    # Track pose distribution
    pose_counts = {pose: 0 for pose in pose_classes}

    for sequence in sequences:
        feature_cols = [
            col for col in sequence.columns if col not in ["asana", "error"]
        ]
        features = sequence[feature_cols].values

        # Get the pose label and normalize it
        pose_label = str(sequence["asana"].iloc[0]).lower().strip().replace(" ", "_")

        if pose_label not in pose_mapping:
            print(f"Warning: Unknown pose label '{pose_label}'. Skipping sequence.")
            continue

        label = pose_mapping[pose_label]
        pose_counts[pose_label] += 1

        all_features.append(features)
        all_labels.extend([label] * len(features))

    print(f"\nPose distribution:")
    for pose, count in pose_counts.items():
        print(f"  {pose}: {count} sequences")

    if len(all_features) == 0:
        raise ValueError("No valid sequences found! Check your data and pose labels.")

    # Stack all features and labels
    X = np.vstack(all_features)
    y = np.array(all_labels)

    # Reshape to sequences BEFORE train/test split
    sequence_length = 10
    num_sequences = len(X) // sequence_length

    # Truncate to ensure we have complete sequences
    X_truncated = X[: num_sequences * sequence_length]
    y_truncated = y[: num_sequences * sequence_length]

    # Reshape to sequences
    X_sequences = X_truncated.reshape(num_sequences, sequence_length, -1)
    y_sequences = y_truncated[::sequence_length]

    # Check if we have enough samples for stratification
    unique, counts = np.unique(y_sequences, return_counts=True)
    min_samples = min(counts)

    if min_samples < 2:
        print(f"Warning: Some classes have less than 2 samples. Cannot perform stratified split.")
        print(f"Class distribution: {dict(zip(unique, counts))}")
        raise ValueError("Need at least 2 samples per class for train-test split")

    # Train-test split with stratification
    train_data, test_data = train_test_split(
        np.column_stack([X_sequences.reshape(num_sequences, -1), y_sequences]),
        test_size=test_size,
        stratify=y_sequences,
        random_state=42
    )

    train_features = train_data[:, :-1]
    train_labels = train_data[:, -1]
    test_features = test_data[:, :-1]
    test_labels = test_data[:, -1]

    train_features = train_features.reshape(-1, sequence_length, X_sequences.shape[2])
    test_features = test_features.reshape(-1, sequence_length, X_sequences.shape[2])

    # Standardize features
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(
        train_features.reshape(-1, train_features.shape[2])
    )
    test_scaled = scaler.transform(test_features.reshape(-1, test_features.shape[2]))

    # Reshape back to sequences
    train_scaled = train_scaled.reshape(train_features.shape)
    test_scaled = test_scaled.reshape(test_features.shape)

    # Convert to tensors
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


def prepare_dataset_tensors(sequences, test_size=0.2):
    """
    prepare tensors for training and testing (without fixing batch size).
    This is useful for hyperparameter search where batch_size changes.

    Returns
    -------
    tuple
        (train_tensor, test_tensor, train_labels_tensor, test_labels_tensor, scaler, pose_mapping)
    """
    pose_classes = ["cobra", "tree", "goddess", "chair", "downdog", "warrior", "surya_namaskar"]
    pose_mapping = {pose: idx for idx, pose in enumerate(pose_classes)}
    print(f"\nPose mapping: {pose_mapping}")

    all_features = []
    all_labels = []
    pose_counts = {pose: 0 for pose in pose_classes}

    for sequence in sequences:
        feature_cols = [col for col in sequence.columns if col not in ["asana", "error"]]
        features = sequence[feature_cols].values

        pose_label = str(sequence["asana"].iloc[0]).lower().strip().replace(" ", "_")
        if pose_label not in pose_mapping:
            print(f"Warning: Unknown pose label '{pose_label}'. Skipping sequence.")
            continue

        label = pose_mapping[pose_label]
        pose_counts[pose_label] += 1

        all_features.append(features)
        all_labels.extend([label] * len(features))

    print(f"\nPose distribution:")
    for pose, count in pose_counts.items():
        print(f"  {pose}: {count} sequences")

    if len(all_features) == 0:
        raise ValueError("No valid sequences found! Check your data and pose labels.")

    X = np.vstack(all_features)
    y = np.array(all_labels)

    sequence_length = 10
    num_sequences = len(X) // sequence_length

    X_truncated = X[: num_sequences * sequence_length]
    y_truncated = y[: num_sequences * sequence_length]

    X_sequences = X_truncated.reshape(num_sequences, sequence_length, -1)
    y_sequences = y_truncated[::sequence_length]

    unique, counts = np.unique(y_sequences, return_counts=True)
    min_samples = min(counts)
    if min_samples < 2:
        print(f"Warning: Some classes have less than 2 samples. Cannot perform stratified split.")
        print(f"Class distribution: {dict(zip(unique, counts))}")
        raise ValueError("Need at least 2 samples per class for train-test split")

    train_data, test_data = train_test_split(
        np.column_stack([X_sequences.reshape(num_sequences, -1), y_sequences]),
        test_size=test_size,
        stratify=y_sequences,
        random_state=42
    )

    train_features = train_data[:, :-1]
    train_labels = train_data[:, -1]
    test_features = test_data[:, :-1]
    test_labels = test_data[:, -1]

    train_features = train_features.reshape(-1, sequence_length, X_sequences.shape[2])
    test_features = test_features.reshape(-1, sequence_length, X_sequences.shape[2])

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features.reshape(-1, train_features.shape[2]))
    test_scaled = scaler.transform(test_features.reshape(-1, test_features.shape[2]))

    train_scaled = train_scaled.reshape(train_features.shape)
    test_scaled = test_scaled.reshape(test_features.shape)

    train_tensor = torch.tensor(train_scaled).float()
    test_tensor = torch.tensor(test_scaled).float()
    train_labels_tensor = torch.tensor(train_labels).long()
    test_labels_tensor = torch.tensor(test_labels).long()

    return train_tensor, test_tensor, train_labels_tensor, test_labels_tensor, scaler, pose_mapping


def create_data_loaders_from_tensors(
    train_tensor, test_tensor, train_labels_tensor, test_labels_tensor, batch_size=8
):
    """Create DataLoaders from tensors with a given batch size."""
    train_loader = DataLoader(
        TensorDataset(train_tensor, train_labels_tensor), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(test_tensor, test_labels_tensor), batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


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
    print(f"\nUsing device: {device}")
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


def random_search_hyperparams(
    train_tensor,
    test_tensor,
    train_labels_tensor,
    test_labels_tensor,
    input_size,
    sequence_length,
    pose_mapping,
    n_trials=10,
):
    """
    Perform random search over hyperparameters for the LSTM classifier.

    Parameters
    ----------
    n_trials : int
        Number of random configurations to try.

    Returns
    -------
    tuple
        (best_config, results_list)
    """

    # Define search space
    learning_rates = [1e-4, 3e-4, 1e-3, 3e-3]
    hidden_sizes = [32, 64, 128]
    num_layers_list = [1, 2]
    batch_sizes = [8, 16, 32]
    num_epochs_list = [10, 15, 20]  # you can increase for more serious tuning

    best_config = None
    best_accuracy = -1.0
    results = []

    num_classes = len(pose_mapping)

    print("\n==============================")
    print("Starting Random Search")
    print("==============================\n")

    for trial in range(1, n_trials + 1):
        # Sample a random configuration
        lr = random.choice(learning_rates)
        hidden_size = random.choice(hidden_sizes)
        num_layers = random.choice(num_layers_list)
        batch_size = random.choice(batch_sizes)
        num_epochs = random.choice(num_epochs_list)

        print(f"\n--- Trial {trial}/{n_trials} ---")
        print(f"  learning_rate = {lr}")
        print(f"  hidden_size   = {hidden_size}")
        print(f"  num_layers    = {num_layers}")
        print(f"  batch_size    = {batch_size}")
        print(f"  num_epochs    = {num_epochs}")

        # Create DataLoaders for this specific batch size
        train_loader, test_loader = create_data_loaders_from_tensors(
            train_tensor,
            test_tensor,
            train_labels_tensor,
            test_labels_tensor,
            batch_size=batch_size,
        )

        # Create model for this configuration
        model = ClassifyPose(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            sequence_length=sequence_length,
            num_classes=num_classes,
        )

        # Train model
        trained_model, history = train_model(
            model,
            train_loader,
            test_loader,
            num_epochs=num_epochs,
            learning_rate=lr,
        )

        # Evaluate
        evaluation = evaluate_model(trained_model, test_loader, pose_mapping)
        acc = evaluation["accuracy"]
        print(f"  Trial Test Accuracy: {acc*100:.2f}%")

        results.append(
            {
                "trial": trial,
                "learning_rate": lr,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "accuracy": acc,
            }
        )

        # Track best
        if acc > best_accuracy:
            best_accuracy = acc
            best_config = results[-1]

    print("\n==============================")
    print("Random Search Completed")
    print("==============================")
    print(f"Best accuracy: {best_accuracy*100:.2f}%")
    print("Best configuration:")
    for k, v in best_config.items():
        if k != "accuracy":
            print(f"  {k}: {v}")

    return best_config, results


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
        "--batch_size", type=int, default=8, help="Batch size for training (single run)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument("--hidden_size", type=int, default=32, help="LSTM hidden size")
    parser.add_argument(
        "--num_layers", type=int, default=1, help="Number of LSTM layers")
    parser.add_argument(
        "--sequence_length", type=int, default=10, help="Sequence length"
    )
    parser.add_argument("--no_plots", action="store_true", help="Skip plotting")
    parser.add_argument(
        "--use_cached_features",
        action="store_true",
        help="Load cached features instead of extracting",
    )
    parser.add_argument(
        "--random_search",
        action="store_true",
        help="Run random search over hyperparameters instead of single training run",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of random search trials (only used with --random_search)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("PosePilot Training Pipeline - Updated for 7 Classes")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Epochs (single run): {args.epochs}")
    print(f"  Batch size (single run): {args.batch_size}")
    print(f"  Learning rate (single run): {args.learning_rate}")
    print(f"  Hidden size (single run): {args.hidden_size}")
    print(f"  Num layers (single run): {args.num_layers}")
    print(f"  Random search: {args.random_search}")
    if args.random_search:
        print(f"  Random search trials: {args.trials}")
    print("=" * 80)

    data_dir = args.data_dir
    input_size = 680  # This is calculated from combinations of landmarks
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    sequence_length = args.sequence_length
    num_classes = 7  # Updated from 6 to 7 to include Surya Namaskar
    num_epochs = args.epochs
    learning_rate = args.learning_rate

    processed_cache_path = os.path.join(data_dir, "cached_classify_features_v7.pkl")

    if args.use_cached_features and os.path.exists(processed_cache_path):
        print("\nLoading cached processed sequences...")
        with open(processed_cache_path, "rb") as f:
            sequences = pickle.load(f)
        print(f"Loaded {len(sequences)} cached sequences")
    else:
        print("\nLoading and processing data from CSV files...")
        sequences = load_and_process_data(data_dir)

        if len(sequences) == 0:
            raise ValueError("No sequences were loaded! Check your data directory structure.")

        print(f"\nProcessed {len(sequences)} sequences")

        print("Saving processed sequences to cache...")
        with open(processed_cache_path, "wb") as f:
            pickle.dump(sequences, f)
        print("Processed sequences cached successfully!")

    # Branch 1: Random Search Mode
    if args.random_search:
        print("\nPreparing tensors for random search...")
        (
            train_tensor,
            test_tensor,
            train_labels_tensor,
            test_labels_tensor,
            scaler,
            pose_mapping,
        ) = prepare_dataset_tensors(sequences)

        # Derive dimensions from tensors
        input_size = train_tensor.shape[2]
        sequence_length = train_tensor.shape[1]

        best_config, search_results = random_search_hyperparams(
            train_tensor=train_tensor,
            test_tensor=test_tensor,
            train_labels_tensor=train_labels_tensor,
            test_labels_tensor=test_labels_tensor,
            input_size=input_size,
            sequence_length=sequence_length,
            pose_mapping=pose_mapping,
            n_trials=args.trials,
        )

        # Optionally save scaler and pose_mapping after random search
        os.makedirs("models", exist_ok=True)
        with open("models/classify_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        with open("models/pose_mapping.pkl", "wb") as f:
            pickle.dump(pose_mapping, f)

        print("\nRandom search finished. No single best model saved automatically.")
        print("You can now re-run training with the best hyperparameters from the logs.")
        return

    # Branch 2: Normal Single-Run Training
    print("\nPreparing training data...")
    train_loader, test_loader, scaler, pose_mapping = prepare_training_data(sequences)
    print(f"\nDataset split:")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")

    print("\nCreating model...")
    model = ClassifyPose(
        input_size, hidden_size, num_layers, sequence_length, num_classes
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    print("\n" + "=" * 80)
    print("Starting training (single run)...")
    print("=" * 80)
    trained_model, history = train_model(
        model, train_loader, test_loader, num_epochs, learning_rate
    )

    print("\n" + "=" * 80)
    print("Evaluating model...")
    print("=" * 80)
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
        print("\nGenerating plots...")
        plot_training_history(history)
        plot_confusion_matrix(
            evaluation["labels"], evaluation["predictions"], pose_mapping
        )

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    print("\nSaving model and artifacts...")
    torch.save(trained_model.state_dict(), "models/pose_classification_model.pth")
    with open("models/classify_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("models/pose_mapping.pkl", "wb") as f:
        pickle.dump(pose_mapping, f)

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print("Saved files:")
    print("  - models/pose_classification_model.pth")
    print("  - models/classify_scaler.pkl")
    print("  - models/pose_mapping.pkl")
    print("=" * 80)


if __name__ == "__main__":
    main()
