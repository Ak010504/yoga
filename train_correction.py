"""
Training pipeline for pose correction model.
Modified to handle 7 classes based on folder names in data directory.
Assumes CSV files are directly within each pose folder (e.g., data/cobra/file1.csv).
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
    Load all CSV files from all asana directories directly (no camera subdirs).
    Assumes folder names represent the 7 classes: cobra, tree, goddess, chair, downdog, warrior, surya_namaskar.

    Parameters
    ----------
    data_dir : str
        path to the data directory containing pose subdirectories

    Returns
    -------
    dict
        dictionary with asana names as keys and lists of DataFrames as values
    """
    all_data = {}

    # Get all asana directories (folder names should be the class names)
    # Updated to reflect the 7 classes you intend to use
    # You can hardcode them or list the directories - let's list them for flexibility
    asana_dirs = [
        d for d in os.listdir(data_dir) 
        if os.path.isdir(os.path.join(data_dir, d)) 
        # Optional: Filter to known classes if needed
        # and d in ["cobra", "tree", "goddess", "chair", "downdog", "warrior", "surya_namaskar"]
    ]

    print(f"Found potential asana directories: {asana_dirs}")

    for asana_name in asana_dirs:
        asana_path = os.path.join(data_dir, asana_name)
        dataframes = []

        # Load all CSV files directly from the asana directory (no cam subdirs)
        csv_files = [f for f in os.listdir(asana_path) if f.endswith(".csv")]

        for csv_file in csv_files:
            csv_path = os.path.join(asana_path, csv_file)
            try:
                df = pd.read_csv(csv_path)
                dataframes.append(df)
            except Exception as e:
                print(f"Warning: Could not read {csv_path}: {e}")
                continue # Skip problematic files

        all_data[asana_name] = dataframes
        print(f"Total loaded {len(dataframes)} CSV files for asana: {asana_name}")

    if not all_data:
        raise ValueError(f"No valid asana directories with CSV files found in {data_dir}. "
                         f"Expected directories like: cobra, tree, goddess, chair, downdog, warrior, surya_namaskar.")

    return all_data


def calculate_asana_averages(asana_data):
    """
    Calculate average frame count for each asana.

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
            frame_counts.append(len(df))

        avg_frames = int(np.mean(frame_counts))
        asana_averages[asana_name] = avg_frames
        print(f"{asana_name}: {len(dataframes)} sequences, avg {avg_frames} frames")

    return asana_averages


def augment_data_with_peaks(
    dataframes, asana_name, asana_averages, desired_peak_count=25
):
    """
    Augment data using peak selection method (3x augmentation) on each sequence.
    Robustly handles cases where desired_peak_count is too large for sequence length.
    Ensures at least one valid (i-1, i, i+1) triplet is selected if possible.

    Parameters
    ----------
    dataframes : list
        list of DataFrames, each representing one sequence
    asana_name : str
        name of the asana (used to get the average frame count)
    asana_averages : dict
        dictionary with asana names as keys and average frame counts as values
    desired_peak_count : int
        number of peaks to select per sequence (capped based on sequence length)

    Returns
    -------
    list
        list of augmented DataFrames (3x the original size, or original size if augmentation fails)
    """
    augmented_dataframes = []

    for i, df in enumerate(dataframes):
        current_length = len(df)
        # Need at least 3 frames to form one (i-1, i, i+1) triplet
        if current_length < 3:
            print(f"Warning: Sequence {i} of {asana_name} has {current_length} frames, insufficient for augmentation. Using original sequence.")
            # Append the original df 3 times as a fallback if no augmentation is possible, or just once.
            # Let's append once and handle the 3x multiplication elsewhere if needed.
            # For now, just append the original df once if augmentation is impossible.
            # augmented_dataframes.extend([df, df, df]) # This would be fallback
            # Or just add the original df if no augmentation is possible for this sequence
            # The main script expects 3x data, so maybe we should still add 3 empty or original dataframes?
            # No, the script checks for empty dataframes later.
            # The safest is to add the original df once, and let the main script handle if the list is empty after augmentation.
            # The main script already handles empty lists.
            # Let's try adding the original df 3 times as a placeholder if augmentation fails completely.
            # Or, better, add 3 *copies* of the original df if no valid peaks can be found.
            # However, the original script's logic expects augmented_dataframes to be empty if nothing can be augmented.
            # The key is that the function should return *something* that can be used by the next steps.
            # If no augmentation is possible, returning the original data 3 times might be misleading for training.
            # Let's stick to the safer approach: if augmentation is impossible, return an empty list for this df,
            # but since we loop per df, we can't return an empty list for the *entire* function input.
            # We should add the original df 3 times as a fallback if *any* peaks are impossible.
            # Or, add the original df once, and let the downstream logic handle it if needed.
            # Let's add the original df once if augmentation is impossible for this specific sequence.
            # augmented_dataframes.append(df.copy()) # This might break downstream expecting 3x.
            # Let's see what the original script does when augmentation produces no results.
            # It seems like it expects augmented_dataframes list for this asana to potentially be empty or have fewer entries.
            # The prepare_correction_data function processes augmented_dataframes[asana_name].
            # If augmented_dataframes[asana_name] is empty after this function, it fails.
            # So, if we cannot augment, we *must* provide some data.
            # The safest fallback is to return the original sequence 3 times if augmentation fails.
            # This means no augmentation occurred, but the script gets the required 3 entries.
            augmented_dataframes.extend([df.copy(), df.copy(), df.copy()])
            continue # Move to the next sequence

        # Determine the actual number of peaks to select based on sequence length
        # Maximum possible peaks is len(df) - 2 (to allow for i-1 and i+1)
        max_possible_peaks = current_length - 2
        actual_peak_count = min(desired_peak_count, max_possible_peaks)

        if actual_peak_count < 1:
            # This case should theoretically not happen if current_length >= 3,
            # but adding for completeness.
            print(f"Warning: Could not select any peaks for sequence {i} of {asana_name}. Using original sequence.")
            augmented_dataframes.extend([df.copy(), df.copy(), df.copy()])
            continue

        # Generate indices for the 'middle' frame of the triplets (i)
        # Range is from 1 to current_length - 2 (inclusive), so i-1 and i+1 are valid
        # Use linspace to get roughly evenly spaced indices
        peak_indices_i = np.linspace(1, current_length - 2, num=actual_peak_count, dtype=int)

        # Create lists for i-1 and i+1, ensuring they are within bounds
        peak_indices_i_minus_1 = peak_indices_i - 1 # This will be >= 0
        peak_indices_i_plus_1 = peak_indices_i + 1 # This will be <= current_length - 1

        # Verify bounds (should be okay given linspace range and -/+ 1)
        # assert np.all(peak_indices_i_minus_1 >= 0), f"Invalid indices for {asana_name} seq {i}"
        # assert np.all(peak_indices_i_plus_1 < current_length), f"Invalid indices for {asana_name} seq {i}"

        # Select the frames using iloc
        try:
            df_peaks_i_minus_1 = df.iloc[peak_indices_i_minus_1].copy()
            df_peaks_i = df.iloc[peak_indices_i].copy()
            df_peaks_i_plus_1 = df.iloc[peak_indices_i_plus_1].copy()

            # Add the three augmented DataFrames (variations) to the list
            augmented_dataframes.extend([df_peaks_i, df_peaks_i_minus_1, df_peaks_i_plus_1])

        except IndexError as e:
            # This should theoretically not happen with the bounds checking above,
            # but catch it just in case.
            print(f"Warning: IndexError during augmentation for sequence {i} of {asana_name}: {e}. Using original sequence.")
            augmented_dataframes.extend([df.copy(), df.copy(), df.copy()])

    return augmented_dataframes



def collect_all_data_for_global_scaling(asana_data, asana_averages):
    """
    Collect all data first to fit global scalers (like notebook approach).

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
            # Extract pose landmark data (first 132 columns: 33 landmarks × 4 values each)
            pose_data = df.iloc[:, :132].copy()
            structured_df, body_pose_landmarks = structure_data(pose_data)
            structured_df, body_pose_landmarks = update_body_pose_landmarks(
                structured_df, body_pose_landmarks
            )
            angle_df = correction_angles_convert(structured_df)
            angle_df = cal_error(angle_df)
            # Ensure sequence has consistent length based on average for the specific asana
            angle_df = equal_rows(angle_df, asana_averages[asana_name])
            all_angle_data.append(angle_df)
            asana_processed.append(angle_df)

        processed_dataframes[asana_name] = asana_processed

    combined_data = pd.concat(all_angle_data, ignore_index=True)
    return combined_data, processed_dataframes


# ... (other imports and functions remain the same) ...

def normalize_with_global_scalers(all_data, processed_dataframes, asana_averages):
    """
    Fit global scalers on all data, then apply to individual sequences (like notebook).
    Also applies dynamic peak augmentation based on sequence length.
    """
    # Fit scalers on the entire combined dataset for features f1-f9
    global_scalers = []
    feature_columns = [f"f{i}" for i in range(1, 10)] # f1 to f9
    for col in feature_columns:
        scaler = MinMaxScaler()
        # Fit on the specific column across all data
        scaler.fit(all_data[col].values.reshape(-1, 1))
        global_scalers.append(scaler)

    normalized_data = {}

    for asana_name, dataframes in processed_dataframes.items():
        normalized_dataframes = []
        for df in tqdm(dataframes, desc=f"Normalizing {asana_name}"):
            normalized_df = df.copy()
            # Apply the corresponding fitted scaler to each feature column
            for i, col in enumerate(feature_columns):
                 normalized_df[col] = global_scalers[i].transform(
                     df[col].values.reshape(-1, 1)
                 ).flatten() # Flatten back to 1D array
            normalized_dataframes.append(normalized_df)

        # --- Dynamic Peak Count Calculation ---
        avg_len = asana_averages[asana_name]
        # Define a reasonable minimum sequence length needed for augmentation
        # E.g., you need at least 3 frames to have one valid i-1, i, i+1 triplet
        min_len_for_augmentation = 3
        # Define a maximum desired peak count (to avoid too many points even for long sequences)
        max_peak_count = 25
        # Define a target number of frames per peak (e.g., aim for a peak every ~4-10 frames)
        # This is a tunable parameter. 4 means denser sampling, 10 means sparser.
        frames_per_peak_target = 5 # Adjust this value as needed based on your data

        if avg_len < min_len_for_augmentation:
            print(f"Warning: Average length ({avg_len}) for {asana_name} is too short for augmentation. Skipping augmentation for this asana's data.")
            # If augmentation is skipped, you might just pass the normalized_dataframes directly
            # Or handle differently if augmentation is strictly required
            # For now, let's assume we want to try augmentation if possible
            feasible_peak_count = 0 # Or set to 1 if you want at least one point, though augmentation won't work
        else:
            # Calculate desired peak count based on length and target spacing
            calculated_peak_count = max(1, avg_len // frames_per_peak_target)
            # Cap it at the maximum desired count
            feasible_peak_count = min(max_peak_count, calculated_peak_count)

        print(f"Calculating augmentation for {asana_name}: avg_len={avg_len}, target_spacing={frames_per_peak_target}, calculated_count={calculated_peak_count}, final_count={feasible_peak_count}")

        if feasible_peak_count > 0:
            # --- Call Augmentation with Dynamic Count ---
            augmented_dataframes = augment_data_with_peaks(
                normalized_dataframes, asana_name, asana_averages, desired_peak_count=feasible_peak_count
            )
        else:
            # If feasible count is 0 (or too low), return the original normalized dataframes
            # This might mean no 3x augmentation occurs for this asana
            print(f"Insufficient length or calculated peak count for {asana_name}, using original normalized dataframes for training (no augmentation).")
            augmented_dataframes = normalized_dataframes # Or handle differently

        normalized_data[asana_name] = augmented_dataframes

    return normalized_data, global_scalers

# ... (rest of the script remains the same, including the original augment_data_with_peaks function) ...


def prepare_correction_data(normalized_data, asana_name, asana_averages):
    """
    Prepare data for correction model training using sliding window approach with augmentation.

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
    feature_columns = [f"f{i}" for i in range(1, 10)] # f1 to f9

    all_sequences = []
    for df in dataframes:
        sequence_data = df[feature_columns].values # Shape: (seq_len, 9)
        all_sequences.append(sequence_data)
    # all_sequences is now a list of arrays, each with shape (variable_len, 9)
    # Need to stack or pad if sequences have different lengths after augmentation/equal_rows
    # However, equal_rows should have made them consistent per asana based on asana_averages
    # Let's assume they are now consistent within each asana's data after equal_rows and augmentation.
    # The augmentation might create sequences of length desired_peak_count (25).
    # The original equal_rows sets the target length per asana, but augmentation overrides this.
    # The final sequence length per sample will be determined by the augmentation (e.g., 25 peaks = 25 frames).
    # The model expects (batch, seq, features). Here, seq will be 25 after augmentation.
    # The sliding window happens implicitly in the model (input X, target Y where Y is shifted).
    # Input: [f1_t, f2_t, ..., f9_t] -> Output: [f1_t+1, f2_t+1, ..., f9_t+1]
    data_input = np.array(all_sequences, dtype=np.float32) # Shape: (num_samples, 25, 9) or similar after aug
    print(f"Data shape for {asana_name} after augmentation: {data_input.shape}")

    num_sequences = len(all_sequences)
    if num_sequences == 0:
        raise ValueError(f"No data available for asana {asana_name} after processing and augmentation.")

    train_test_split = 0.2
    all_sequence_indices = list(range(num_sequences))
    # Ensure we have enough samples for both train and test
    num_test = int(train_test_split * num_sequences)
    if num_test == 0 and num_sequences > 1:
        num_test = 1 # Ensure at least 1 test sample if possible
    elif num_test >= num_sequences:
        num_test = max(0, num_sequences - 1) # Ensure at least 1 train sample if possible

    test_indices = random.sample(all_sequence_indices, num_test)
    train_indices = [i for i in all_sequence_indices if i not in test_indices]

    train_data = data_input[train_indices] # Shape: (num_train, seq_len, 9)
    test_data = data_input[test_indices]   # Shape: (num_test, seq_len, 9)

    train_tensor = torch.tensor(train_data, dtype=torch.float32) # Shape: (num_train, seq_len, 9)
    test_tensor = torch.tensor(test_data, dtype=torch.float32)   # Shape: (num_test, seq_len, 9)

    # Prepare input (X) and target (Y) for sequence prediction
    # Input: sequence up to t-1 -> Target: sequence from t onwards
    # For each sequence in the batch, X = seq[:-1], Y = seq[1:]
    # This means the model learns to predict the next frame based on the current and previous frames.
    train_labels = train_tensor[:, 1:, :] # Target: frames 1 to end -> Shape: (num_train, seq_len-1, 9)
    test_labels = test_tensor[:, 1:, :]   # Target: frames 1 to end -> Shape: (num_test, seq_len-1, 9)
    train_tensor = train_tensor[:, :-1, :] # Input: frames 0 to second-last -> Shape: (num_train, seq_len-1, 9)
    test_tensor = test_tensor[:, :-1, :]   # Input: frames 0 to second-last -> Shape: (num_test, seq_len-1, 9)

    print(f"Final tensor shapes for {asana_name}:")
    print(f"  Train Input: {train_tensor.shape}, Train Labels: {train_labels.shape}")
    print(f"  Test Input: {test_tensor.shape}, Test Labels: {test_labels.shape}")

    return train_tensor, train_labels, test_tensor, test_labels


def train_correction_model(
    model, train_loader, test_loader, num_epochs=30, learning_rate=0.001
):
    """
    Train the correction model.

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

            output = model(data) # Output shape: (batch_size, seq_len-1, 9)
            target_flat = target.view(-1, 9) # Flatten target: (batch_size * (seq_len-1), 9)
            output_flat = output.view(-1, 9) # Flatten output: (batch_size * (seq_len-1), 9)

            loss = criterion(output_flat, target_flat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Test (matching notebook approach - process only first sample from each batch)
        # Note: This test logic might be less standard for sequence models.
        # It calculates loss only on the *first* sequence of each batch.
        model.eval()
        test_loss = 0.0
        num_test_batches = 0

        with torch.no_grad():
            for data, target in test_loader:
                if len(data) > 0: # Ensure batch is not empty
                    data, target = data.to(device), target.to(device)
                    # Use only the first sequence in the batch
                    output = model(data[0:1]) # Shape: (1, seq_len-1, 9)
                    target_flat = target[0:1].view(-1, 9) # Shape: (1 * (seq_len-1), 9)
                    output_flat = output.view(-1, 9) # Shape: (1 * (seq_len-1), 9)

                    loss = criterion(output_flat, target_flat)
                    test_loss += loss.item()
                    num_test_batches += 1

        if num_test_batches > 0:
            test_loss /= num_test_batches
        else:
            test_loss = float('inf') # Or handle differently if no test batches

        train_loss /= len(train_loader)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(
            f"Epoch {epoch+1:3d}/{num_epochs}: Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}"
        )

    return train_losses, test_losses


def save_model_and_scalers(model, scalers, asana_name, output_dir):
    """
    Save trained model and scalers.

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
    parser = argparse.ArgumentParser(description="Train pose correction models for 7 classes")
    parser.add_argument(
        "--data_dir", type=str, default="data", help="Data directory path containing pose subdirectories"
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
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size") # Keep 1 as per notebook
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

    print("Starting correction training for all asanas (7 classes expected)")
    print(f"Looking for data in: {args.data_dir}")
    print(f"Will save models to: {args.output_dir}")

    # Check for cached features
    cache_file = os.path.join(args.data_dir, "cached_correction_features_v7.pkl") # Updated cache name

    if args.use_cached_features and os.path.exists(cache_file):
        print("Loading cached processed features...")
        with open(cache_file, "rb") as f:
            cached_data = pickle.load(f)
        normalized_data = cached_data["normalized_data"]
        scalers = cached_data["scalers"]
        asana_averages = cached_data["asana_averages"]
        print("Cached features loaded successfully.")
    else:
        print("Loading and processing raw data...")
        # Load data for all asanas (folder names determine classes)
        asana_data = load_all_asana_data(args.data_dir)

        # Calculate average frames for each asana (used by equal_rows)
        asana_averages = calculate_asana_averages(asana_data)

        # Process data and fit global scalers
        all_data, processed_dataframes = collect_all_data_for_global_scaling(
            asana_data, asana_averages
        )
        normalized_data, scalers = normalize_with_global_scalers(
            all_data, processed_dataframes, asana_averages
        )

        # Cache the processed data and scalers
        cached_data = {
            "normalized_data": normalized_data,
            "scalers": scalers,
            "asana_averages": asana_averages,
        }
        print("Saving processed features to cache...")
        with open(cache_file, "wb") as f:
            pickle.dump(cached_data, f)
        print("Cached features saved successfully.")

    print("Data preprocessing completed!")

    # Train models for all found poses (7 classes expected based on folder names)
    input_size = 9  # f1-f9 features
    num_output_features = 9 # Predicting the next 9 features

    print(f"Training correction models for poses found: {list(normalized_data.keys())}")

    for asana_name in normalized_data.keys():
        print(f"\n{'='*60}")
        print(f"Training correction model for: {asana_name}")
        print(f"{'='*60}")

        try:
            train_tensor, train_labels, test_tensor, test_labels = prepare_correction_data(
                normalized_data, asana_name, asana_averages
            )
            
            if train_tensor.size(0) == 0: # Check if training data is empty after split
                 print(f"Warning: Insufficient data for training {asana_name} after train/test split. Skipping.")
                 continue

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
                num_classes=num_output_features, # Output size matches input for sequence prediction
            )

            train_losses, test_losses = train_correction_model(
                model, train_loader, test_loader, args.epochs, args.learning_rate
            )
            save_model_and_scalers(model, scalers, asana_name, args.output_dir)

            print(f"Completed training for {asana_name}")
            print(f"Final train loss: {train_losses[-1]:.6f}")
            print(f"Final test loss: {test_losses[-1]:.6f}")
            
        except ValueError as e:
            print(f"Error during data preparation for {asana_name}: {e}. Skipping this asana.")
            continue # Move to the next asana if preparation fails

    print(f"\n{'='*60}")
    print("All correction models training completed (or attempted)!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
