"""
utility functions for pose analysis and data processing.
"""

import cv2
import warnings
import numpy as np
import pandas as pd
import mediapipe as mp
import scipy.interpolate
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def cal_angle(point1, point2, point3):
    """
    calculate the angle between three points with proper error handling.

    Parameters
    ----------
    point1 : tuple
        (x, y) coordinates of the first point
    point2 : tuple
        (x, y) coordinates of the second point (vertex)
    point3 : tuple
        (x, y) coordinates of the third point

    Returns
    -------
    angle : float
        the angle between the three points in degrees
    """
    try:
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)

        ba = a - b
        bc = c - b
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)

        # Check for zero length vectors (same points)
        if norm_ba == 0 or norm_bc == 0:
            return 0.0  # Return 0 degrees for degenerate cases

        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

        angle_rad = np.arccos(cosine_angle)
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    except Exception as e:
        print(f"Error calculating angle: {e}")
        return 0.0


def cal_error(data):
    """
    calculate the error values for each frame using standard deviation of 5-frame windows.

    Parameters
    ----------
    data : DataFrame
        the DataFrame containing the angle values

    Returns
    -------
    data : DataFrame
        the original DataFrame with an additional column for the error values
    """

    error_list = []
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for idx in range(len(data)):
        temp = []
        for col in numeric_cols:
            window = data[col].iloc[max(idx - 3, 0) : min(idx + 2, len(data))]
            temp.append(np.std(window.values))
        error_list.append(np.mean(temp))

    data["error"] = error_list
    return data


def select_top_frames(data, desired_frames=10):
    """
    select the top number of desired frames based on the error values.

    Parameters
    ----------
    data : DataFrame
        the DataFrame containing the angle values
    desired_frames : int
        the number of frames to select

    Returns
    -------
    data : DataFrame
        the DataFrame containing the selected frames
    """

    min_prominence = 0.1
    error_values = data["error"]
    peaks, properties = find_peaks(error_values, distance=3, prominence=min_prominence)

    if len(peaks) < desired_frames:
        while len(peaks) < 10:
            diff = np.diff(peaks)
            max_diff = np.argmax(diff)
            peaks = np.insert(
                peaks, max_diff + 1, np.mean(peaks[max_diff : max_diff + 2])
            )
        print(
            f"WARNING: Peaks detected at prominence:{min_prominence} were less than desired frames."
        )
    else:
        peaks = peaks[np.argsort(properties["prominences"])[-desired_frames:]]

    data = data.iloc[peaks].reset_index(drop=True)
    return data


def structure_data(data):
    """
    structure the data and add column names, remove unnecessary data.

    Parameters
    ----------
    data : DataFrame
        the input DataFrame with pose landmark data

    Returns
    -------
    data : DataFrame
        the structured DataFrame with proper column names
    """

    body_pose_landmarks = [
        "nose",
        "left eye inner",
        "left eye",
        "left eye outer",
        "right eye inner",
        "right eye",
        "right eye outer",
        "left ear",
        "right ear",
        "mouth left",
        "mouth right",
        "left shoulder",
        "right shoulder",
        "left elbow",
        "right elbow",
        "left wrist",
        "right wrist",
        "left pinky",
        "right pinky",
        "left index",
        "right index",
        "left thumb",
        "right thumb",
        "left hip",
        "right hip",
        "left knee",
        "right knee",
        "left ankle",
        "right ankle",
        "left heel",
        "right heel",
        "left foot index",
        "right foot index",
    ]

    col_name = []
    for i in body_pose_landmarks:
        col_name += [i + "_X", i + "_Y", i + "_Z", i + "_V"]

    data.columns = col_name
    data = data[data.columns[~data.columns.str.contains(" V")]]

    return data, body_pose_landmarks


def update_body_pose_landmarks(data, body_pose_landmarks):
    # remove certain body landmarks
    remove_list = [
        "left eye",
        "left eye inner",
        "left eye outer",
        "left ear",
        "right eye",
        "right eye inner",
        "right eye outer",
        "right ear",
        "mouth left",
        "mouth right",
        "left pinky",
        "right pinky",
        "left thumb",
        "right thumb",
        "left heel",
        "right heel",
    ]

    for i in remove_list:
        body_pose_landmarks.remove(i)
        data = data[data.columns[~data.columns.str.contains(i)]]

    return data, body_pose_landmarks


def correction_angles_convert(final_df):
    """
    convert pose landmarks to 9 angle features for correction analysis.

    Parameters
    ----------
    final_df : DataFrame
        dataframe containing pose landmarks with X,Y coordinates

    Returns
    -------
    DataFrame
        dataframe with 9 angle features (f1-f9)
    """
    feature_df = pd.DataFrame()

    feature_df["f1"] = final_df.apply(
        lambda x: cal_angle(
            (x["left shoulder_X"], x["left shoulder_Y"]),
            (x["left elbow_X"], x["left elbow_Y"]),
            (x["left wrist_X"], x["left wrist_Y"]),
        ),
        axis=1,
    )
    feature_df["f2"] = final_df.apply(
        lambda x: cal_angle(
            (x["right shoulder_X"], x["right shoulder_Y"]),
            (x["right elbow_X"], x["right elbow_Y"]),
            (x["right wrist_X"], x["right wrist_Y"]),
        ),
        axis=1,
    )
    feature_df["f3"] = final_df.apply(
        lambda x: cal_angle(
            (x["left shoulder_X"], x["left shoulder_Y"]),
            (x["left hip_X"], x["left hip_Y"]),
            (x["left knee_X"], x["left knee_Y"]),
        ),
        axis=1,
    )
    feature_df["f4"] = final_df.apply(
        lambda x: cal_angle(
            (x["right shoulder_X"], x["right shoulder_Y"]),
            (x["right hip_X"], x["right hip_Y"]),
            (x["right knee_X"], x["right knee_Y"]),
        ),
        axis=1,
    )
    feature_df["f5"] = final_df.apply(
        lambda x: cal_angle(
            (x["left hip_X"], x["left hip_Y"]),
            (x["left knee_X"], x["left knee_Y"]),
            (x["left ankle_X"], x["left ankle_Y"]),
        ),
        axis=1,
    )
    feature_df["f6"] = final_df.apply(
        lambda x: cal_angle(
            (x["right hip_X"], x["right hip_Y"]),
            (x["right knee_X"], x["right knee_Y"]),
            (x["right ankle_X"], x["right ankle_Y"]),
        ),
        axis=1,
    )
    feature_df["f7"] = final_df.apply(
        lambda x: cal_angle(
            (x["left shoulder_X"], x["left shoulder_Y"]),
            (x["nose_X"], x["nose_Y"]),
            (x["right shoulder_X"], x["right shoulder_Y"]),
        ),
        axis=1,
    )
    feature_df["f8"] = final_df.apply(
        lambda x: cal_angle(
            (x["left elbow_X"], x["left elbow_Y"]),
            (x["left shoulder_X"], x["left shoulder_Y"]),
            (x["left hip_X"], x["left hip_Y"]),
        ),
        axis=1,
    )
    feature_df["f9"] = final_df.apply(
        lambda x: cal_angle(
            (x["right elbow_X"], x["right elbow_Y"]),
            (x["right shoulder_X"], x["right shoulder_Y"]),
            (x["right hip_X"], x["right hip_Y"]),
        ),
        axis=1,
    )

    return feature_df


def reduce_rows(df, limit):
    """
    reduce the number of rows in the DataFrame to the target length by removing rows with the smallest error.

    Parameters
    ----------
    df : DataFrame
        the input DataFrame
    limit : int
        the target number of rows

    Returns
    -------
    DataFrame
        the DataFrame with reduced rows
    """

    target_length = limit

    while len(df) > target_length:
        min_error_index = df["error"].idxmin()
        df = df.drop(min_error_index).reset_index(drop=True)

        df.drop("error", axis=1, inplace=True)
        df = cal_error(df)

    return df


def add_rows(df, limit):
    """
    increase the number of rows in the DataFrame to the target length by interpolating new rows at the highest error points.

    Parameters
    ----------
    df : DataFrame
        the input DataFrame
    limit : int
        the target number of rows

    Returns
    -------
    DataFrame
        the DataFrame with added rows
    """

    target_length = limit

    while len(df) < target_length:
        max_error_index = df["error"].argmax()

        if np.sum(
            np.std(
                df.loc[max_error_index : max_error_index + 1][
                    ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9"]
                ]
            )
        ) < np.sum(
            np.std(
                df.loc[max_error_index - 1 : max_error_index][
                    ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9"]
                ]
            )
        ):
            offset = -0.5
        else:
            offset = 0.5

        features = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9"]
        new_values = []
        for f in features:
            x = range(len(df))
            y = df[f]
            interp_point = max(0, min(max_error_index + offset, max(x)))
            interp_value = scipy.interpolate.interp1d(x, y, kind="linear")(interp_point)
            new_values.append(interp_value)

        new_row = pd.DataFrame([new_values], columns=features)
        max_error_index = int(max_error_index)
        df = pd.concat(
            [df.iloc[:max_error_index], new_row, df.iloc[max_error_index:]]
        ).reset_index(drop=True)

        df.drop("error", axis=1, inplace=True)
        df = cal_error(df)

    return df


# Inside utils.py, replace the equal_rows function
def equal_rows(df, limit, calculate_error=True):
    """
    Adjust the number of rows in the input data to match the target length by either reducing or adding rows.
    Optionally calculates error before adjusting rows.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame (should contain features f1-f9, potentially an 'error' column if calculate_error=False)
    limit : int
        The target number of rows
    calculate_error : bool
        Whether to call cal_error before adjusting rows (default True for backward compatibility)

    Returns
    -------
    DataFrame
        The DataFrame adjusted to the target length
    """
    target_length = limit

    # Calculate error if requested and the 'error' column doesn't already exist
    if calculate_error and 'error' not in df.columns:
        df = cal_error(df) # Add error column based on existing features
    elif calculate_error and 'error' in df.columns:
        # 'error' column already exists, proceed
        pass
    elif not calculate_error:
        # Do not calculate error, assume df is ready for row adjustment based on its existing features
        # Ensure df has features like f1-f9 for add_rows/reduce_rows logic if needed
        # The add_rows/reduce_rows functions might also need 'error' column for their internal logic.
        # If equal_rows is used *without* error calculation, add_rows/reduce_rows might also need changes.
        # For standard sequence length adjustment using simple interpolation/dropping, we might need a different approach.
        # Let's assume for now that if calculate_error=False, we just want to adjust length *without*
        # the complex error-based dropping/adding logic inside add_rows/reduce_rows.
        # The original equal_rows just calls reduce_rows and add_rows which rely on 'error'.
        # We need to potentially bypass the error-dependent logic if calculate_error=False.
        # The safest way is to modify reduce_rows/add_rows too, or create a simpler length adjuster.
        # Let's modify equal_rows to just do simple truncation/padding if calculate_error=False
        # and the complex logic if calculate_error=True.

        # For simple truncation/padding without error dependency:
        current_len = len(df)
        if current_len > target_length:
            # Truncate
            df = df.iloc[:target_length].reset_index(drop=True)
        elif current_len < target_length:
            # Pad by repeating the last row (or use interpolation if needed)
            # This is a basic padding - might need more sophisticated methods
            last_row = df.iloc[[-1]] # Keep it as a DataFrame
            rows_to_add = target_length - current_len
            padding_df = pd.concat([last_row] * rows_to_add, ignore_index=True)
            df = pd.concat([df, padding_df], ignore_index=True)
        # If lengths are equal, df is returned as is
        return df # Return immediately after simple adjust, bypassing reduce/add_rows


    if len(df) < target_length:
        df = add_rows(df, target_length)

    if len(df) > target_length:
        df = reduce_rows(df, target_length)

    return df



def plot_training_history(history):
    """plot training history."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # plot training loss
    ax1.plot(history["train_losses"], label="Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True)

    # plot training and test accuracy
    ax2.plot(history["train_accuracies"], label="Training Accuracy")
    ax2.plot(history["test_accuracies"], label="Test Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Test Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(labels, predictions, pose_mapping):
    """plot confusion matrix."""
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    pose_names = list(pose_mapping.keys())
    cm = confusion_matrix(labels, predictions)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=pose_names,
        yticklabels=pose_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def give_landmarks(video_path, label, fps):
    """
    extract pose landmarks from video frames.

    Parameters
    ----------
    video_path : str
        path to the video file
    label : str
        label for the data
    fps : int
        frames per second to extract

    Returns
    -------
    tuple
        tuple containing (dataframe, landmark_mp_list)
    """
    frame_count = 0
    df_list = []

    cap = cv2.VideoCapture(video_path)
    mp_pose = mp.solutions.pose

    landmark_mp_list = []
    with mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # No more frames in the video

            # Skip frames according to desired FPS
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps > 0 and fps > 0:
                skip_interval = max(1, int(video_fps / fps))
                if frame_count % skip_interval != 0:
                    frame_count += 1
                    continue

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)
            landmark_mp_list.append(results)

            try:
                landmarks = results.pose_landmarks.landmark
            except AttributeError:
                print("No landmarks found")
            frame_count += 1

            if landmarks is not None:
                pose_row = []
                for landmark in landmarks:
                    pose_row += [
                        landmark.x,
                        landmark.y,
                        landmark.z,
                        landmark.visibility,
                    ]

                # append the pose row to the dataframe (no column names)
                df_list.append(pd.DataFrame([pose_row]))

    cap.release()
    df = pd.concat(df_list, ignore_index=True)

    return df, landmark_mp_list


def calculate_sequence_bounds(landmark_mp_list):
    """
    calculate the min/max bounds of landmarks across the entire sequence.

    Parameters
    ----------
    landmark_mp_list : list
        list of MediaPipe pose results

    Returns
    -------
    tuple or None
        (min_x, min_y, max_x, max_y) bounds or None if no landmarks
    """
    if not landmark_mp_list:
        return None

    min_x, min_y = float("inf"), float("inf")
    max_x, max_y = float("-inf"), float("-inf")

    for mp_result in landmark_mp_list:
        if mp_result.pose_landmarks:
            for landmark in mp_result.pose_landmarks.landmark:
                min_x = min(min_x, landmark.x)
                min_y = min(min_y, landmark.y)
                max_x = max(max_x, landmark.x)
                max_y = max(max_y, landmark.y)

    # Check if we found valid bounds
    if min_x == float("inf"):
        return None

    return (min_x, min_y, max_x, max_y)


def render_landmarks(
    image,
    landmarks,
    connections=None,
    correction_data=None,
    frame_idx=0,
    seq_bounds=None,
):
    """
    render pose landmarks on an image with gradient colors based on correction angles.

    Parameters
    ----------
    image : numpy.ndarray
        input image (BGR format)
    landmarks : mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
        detected pose landmarks
    connections : list, optional
        landmark connections to draw, defaults to POSE_CONNECTIONS
    correction_data : dict, optional
        dictionary containing correction data with 'input_data' and 'corrected_data'
    frame_idx : int
        frame index for correction data
    seq_bounds : tuple, optional
        sequence bounds (min_x, min_y, max_x, max_y) for sequence-aware zoom

    Returns
    -------
    numpy.ndarray
        image with rendered landmarks
    """
    if landmarks is None:
        return image

    if connections is None:
        connections = mp.solutions.pose.POSE_CONNECTIONS

    # Convert BGR to RGB for MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a copy for drawing
    annotated_image = rgb_image.copy()

    # If correction data is provided, create white background with gradient circles
    if correction_data is not None:
        try:
            # Create white background
            annotated_image = np.ones_like(image) * 255

            # Apply sequence-aware zoom if bounds are provided
            if seq_bounds is not None:
                min_x, min_y, max_x, max_y = seq_bounds
                seq_width = max_x - min_x
                seq_height = max_y - min_y
                seq_center_x = (min_x + max_x) / 2
                seq_center_y = (min_y + max_y) / 2

                # Calculate zoom factor to fit sequence with padding
                h, w = annotated_image.shape[:2]
                padding = 0.1  # 10% padding
                zoom_x = (1 - 2 * padding) / seq_width if seq_width > 0 else 1
                zoom_y = (1 - 2 * padding) / seq_height if seq_height > 0 else 1
                zoom_factor = min(zoom_x, zoom_y)

                # Create zoomed landmarks
                from mediapipe.framework.formats import landmark_pb2

                zoomed_landmarks = landmark_pb2.NormalizedLandmarkList()
                center_x, center_y = w // 2, h // 2

                for landmark in landmarks.landmark:
                    zoomed_landmark = zoomed_landmarks.landmark.add()
                    # Apply zoom transformation: shift to center, scale, then shift back
                    zoomed_landmark.x = (
                        center_x + (landmark.x - seq_center_x) * w * zoom_factor
                    ) / w
                    zoomed_landmark.y = (
                        center_y + (landmark.y - seq_center_y) * h * zoom_factor
                    ) / h
                    zoomed_landmark.z = landmark.z

                landmarks_to_use = zoomed_landmarks
            else:
                landmarks_to_use = landmarks

            # Get correction angles for the frame
            input_data = correction_data["input_data"]
            corrected_data = correction_data["corrected_data"]
            angle_corrections = corrected_data - input_data  # Shape: (frames, 9)

            if frame_idx >= len(angle_corrections):
                frame_idx = 0

            frame_corrections = angle_corrections[frame_idx]  # 9 angle corrections

            # Use matplotlib's RdYlGn colormap (Red-Yellow-Green)
            import matplotlib.pyplot as plt

            cmap = plt.cm.RdYlGn_r  # Reversed: Green=good, Red=bad

            # Draw thin black skeleton lines (simplified approach)
            mp_drawing = mp.solutions.drawing_utils
            try:
                mp_drawing.draw_landmarks(
                    annotated_image,
                    landmarks_to_use,
                    mp.solutions.pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 0, 0), thickness=0, circle_radius=0  # No landmarks
                    ),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 0, 0), thickness=2  # Thin black lines
                    ),
                )
            except Exception as e:
                print(f"Warning: Could not draw skeleton lines: {e}")
                # Continue without skeleton lines

            # Define feature centers for gradient circles
            feature_centers = [
                ("left elbow", 0),  # f1: left arm
                ("right elbow", 1),  # f2: right arm
                ("left hip", 2),  # f3: left leg
                ("right hip", 3),  # f4: right leg
                ("left knee", 4),  # f5: left lower leg
                ("right knee", 5),  # f6: right lower leg
                ("left shoulder", 6),  # f7: left torso
                ("right shoulder", 7),  # f8: right torso
                ("nose", 8),  # f9: head alignment
            ]

            landmark_names = [
                "nose",
                "left eye inner",
                "left eye",
                "left eye outer",
                "right eye inner",
                "right eye",
                "right eye outer",
                "left ear",
                "right ear",
                "mouth left",
                "mouth right",
                "left shoulder",
                "right shoulder",
                "left elbow",
                "right elbow",
                "left wrist",
                "right wrist",
                "left pinky",
                "right pinky",
                "left index",
                "right index",
                "left thumb",
                "right thumb",
                "left hip",
                "right hip",
                "left knee",
                "right knee",
                "left ankle",
                "right ankle",
                "left heel",
                "right heel",
                "left foot index",
                "right foot index",
            ]

            # Draw gradient circles for each feature
            for landmark_name, feature_idx in feature_centers:
                # Find landmark index
                landmark_idx = None
                for i, name in enumerate(landmark_names):
                    if name == landmark_name:
                        landmark_idx = i
                        break

                if landmark_idx is not None and landmark_idx < len(
                    landmarks_to_use.landmark
                ):
                    point = landmarks_to_use.landmark[landmark_idx]

                    # Get image dimensions
                    h, w = annotated_image.shape[:2]

                    # Convert normalized coordinates to pixel coordinates
                    x = int(point.x * w)
                    y = int(point.y * h)

                    # Get correction angle for this feature
                    correction_angle = abs(frame_corrections[feature_idx])

                    # Map correction angle to color (0-20 degrees -> green to red)
                    if correction_angle > 20:
                        correction_angle = 20
                    color_intensity = correction_angle / 20.0
                    color_rgb = cmap(color_intensity)

                    # Keep RGB format (no BGR conversion)
                    color_rgb_int = (
                        int(color_rgb[0] * 255),
                        int(color_rgb[1] * 255),
                        int(color_rgb[2] * 255),
                    )

                    # Draw gradient circle (reduced size by 20%)
                    cv2.circle(
                        annotated_image, (x, y), 10, color_rgb_int, -1
                    )  # Filled circle
                    cv2.circle(
                        annotated_image, (x, y), 10, (0, 0, 0), 2
                    )  # Black border

        except Exception as e:
            print(f"Error applying correction colors: {e}")
            # Fall back to default rendering
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(
                annotated_image,
                landmarks,
                connections,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=2
                ),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(255, 0, 0), thickness=2
                ),
            )
    else:
        # Default rendering without correction colors
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(
            annotated_image,
            landmarks,
            connections,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(0, 255, 0), thickness=2, circle_radius=2
            ),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(255, 0, 0), thickness=2
            ),
        )

    # Convert back to BGR
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    return annotated_image


def generate_correction_graph(data_input, outputs, data_original, pose_name):
    """
    generate correction graphs showing original, input, output, and correction vectors.
    matches the exact style from corr_model_test.ipynb notebook.

    Parameters
    ----------
    data_input : numpy.ndarray
        input data (incorrect pose) - transformed/scaled
    outputs : numpy.ndarray
        corrected output data
    data_original : pandas.DataFrame
        original reference data
    pose_name : str
        name of the pose for title

    Returns
    -------
    matplotlib.figure.Figure
        figure object with correction graphs
    """
    import numpy as np

    feature_labels = [
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

    # Create 3x3 subplot grid - exact same as notebook
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))

    for i in range(3):
        for j in range(3):
            feature_idx = 3 * i + j

            axs[i, j].set_xlim([0, len(data_original)])

            # Plot original reference data - exact same as notebook
            axs[i, j].plot(
                data_original.iloc[:, feature_idx], label="Original", linestyle="--"
            )

            # Plot input data (transformed/scaled) - exact same as notebook
            axs[i, j].plot(
                data_input[:, feature_idx], label="Input", linestyle="", marker="o"
            )

            # Plot corrected output data - exact same as notebook
            axs[i, j].plot(outputs[:, feature_idx], label="Output", linestyle="-")

            # Add 1-std band - exact same as notebook
            lw_1std = outputs[:, feature_idx] - np.std(outputs[:, feature_idx])
            up_1std = outputs[:, feature_idx] + np.std(outputs[:, feature_idx])
            axs[i, j].fill_between(
                range(len(outputs)),
                lw_1std,
                up_1std,
                alpha=0.2,
                color="grey",
                label="1-std band",
            )

            # Add 2-std band - exact same as notebook (1.5*std, not 2*std)
            lw_2std = outputs[:, feature_idx] - 1.5 * np.std(outputs[:, feature_idx])
            up_2std = outputs[:, feature_idx] + 1.5 * np.std(outputs[:, feature_idx])
            axs[i, j].fill_between(
                range(len(outputs)),
                lw_2std,
                up_2std,
                alpha=0.1,
                color="grey",
                label="2-std band",
            )

            # Subplot title - exact same as notebook
            axs[i, j].set_title(
                "Feature "
                + str(feature_idx + 1)
                + " "
                + f"({feature_labels[feature_idx]})",
                fontsize=15,
            )

            # Add correction arrows and points - exact same logic as notebook
            for k in range(len(outputs)):
                # Show correction arrows from input to nearest 1-std band if outside 2-std band
                if data_input[k, feature_idx] < lw_2std[k]:
                    axs[i, j].plot(
                        k, data_input[k, feature_idx], "ro", label="incorrect points"
                    )
                    axs[i, j].arrow(
                        k,
                        data_input[k, feature_idx],
                        0,
                        lw_1std[k] - data_input[k, feature_idx],
                        head_width=2,
                        head_length=0.1,
                        fc="red",
                        ec="red",
                        label="correction vector",
                    )
                elif data_input[k, feature_idx] > up_2std[k]:
                    axs[i, j].plot(
                        k, data_input[k, feature_idx], "ro", label="incorrect points"
                    )
                    axs[i, j].arrow(
                        k,
                        data_input[k, feature_idx],
                        0,
                        up_1std[k] - data_input[k, feature_idx],
                        head_width=2,
                        head_length=0.1,
                        fc="red",
                        ec="red",
                        label="correction vector",
                    )

    # Create legend - exact same as notebook
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    unique_labels = sorted(list(set(labels)))

    selected_lines = []
    for label in unique_labels:
        selected_lines.append(lines[labels.index(label)])

    fig.subplots_adjust(top=1)
    fig.legend(
        selected_lines,
        unique_labels,
        loc="upper center",
        framealpha=1,
        fontsize=18,
        ncol=7,
        bbox_to_anchor=(0.5, 0),
    )
    fig.supxlabel("Frames (n)", fontsize=15, x=0.5, y=0.004)
    fig.supylabel("Degrees (°)", fontsize=15, y=0.5, x=0.004)
    plt.tight_layout()

    return fig


def create_correction_visualization(landmarks_df, correction_data, frame_idx=0):
    """
    create a correction visualization showing skeleton with gradient colors
    based on correction angles for joints involved in features.

    Parameters
    ----------
    landmarks_df : pd.DataFrame
        dataframe with pose landmarks
    correction_data : dict
        dictionary containing correction data with 'input_data' and 'corrected_data'
    frame_idx : int
        frame index to visualize (default: 0)

    Returns
    -------
    matplotlib.figure.Figure
        correction visualization figure
    """
    try:
        import matplotlib.pyplot as plt

        # Get correction angles for the frame
        input_data = correction_data["input_data"]
        corrected_data = correction_data["corrected_data"]
        angle_corrections = corrected_data - input_data  # Shape: (frames, 9)

        if frame_idx >= len(angle_corrections):
            frame_idx = 0

        frame_corrections = angle_corrections[frame_idx]  # 9 angle corrections

        # Create figure with grey background
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_facecolor("#808080")  # Grey background
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.invert_yaxis()  # Invert Y axis to match image coordinates

        # Get landmark positions for the frame
        frame_data = landmarks_df.iloc[frame_idx]

        # Define landmark positions
        landmarks = {}
        landmark_names = [
            "nose",
            "left eye inner",
            "left eye",
            "left eye outer",
            "right eye inner",
            "right eye",
            "right eye outer",
            "left ear",
            "right ear",
            "mouth left",
            "mouth right",
            "left shoulder",
            "right shoulder",
            "left elbow",
            "right elbow",
            "left wrist",
            "right wrist",
            "left pinky",
            "right pinky",
            "left index",
            "right index",
            "left thumb",
            "right thumb",
            "left hip",
            "right hip",
            "left knee",
            "right knee",
            "left ankle",
            "right ankle",
            "left heel",
            "right heel",
            "left foot index",
            "right foot index",
        ]

        for name in landmark_names:
            x_col = f"{name}_X"
            y_col = f"{name}_Y"
            if x_col in frame_data.index and y_col in frame_data.index:
                landmarks[name] = (frame_data[x_col], frame_data[y_col])

        # Define connections with their corresponding feature indices
        # Based on correction_angles_convert function
        connections_with_features = [
            # Feature f1: left shoulder, left elbow, left wrist
            (("left shoulder", "left elbow"), 0),
            (("left elbow", "left wrist"), 0),
            # Feature f2: right shoulder, right elbow, right wrist
            (("right shoulder", "right elbow"), 1),
            (("right elbow", "right wrist"), 1),
            # Feature f3: left shoulder, left hip, left knee
            (("left shoulder", "left hip"), 2),
            (("left hip", "left knee"), 2),
            # Feature f4: right shoulder, right hip, right knee
            (("right shoulder", "right hip"), 3),
            (("right hip", "right knee"), 3),
            # Feature f5: left hip, left knee, left ankle
            (("left knee", "left ankle"), 4),
            # Feature f6: right hip, right knee, right ankle
            (("right knee", "right ankle"), 5),
            # Feature f7: nose, left shoulder, left hip
            (("nose", "left shoulder"), 6),
            # Feature f8: nose, right shoulder, right hip
            (("nose", "right shoulder"), 7),
            # Feature f9: left shoulder, nose, right shoulder
            (("left shoulder", "nose"), 8),
            (("nose", "right shoulder"), 8),
        ]

        # Additional connections for complete skeleton (no correction)
        additional_connections = [
            ("left ear", "left eye outer"),
            ("right ear", "right eye outer"),
            ("left eye inner", "left eye outer"),
            ("right eye inner", "right eye outer"),
            ("mouth left", "mouth right"),
            ("left wrist", "left pinky"),
            ("left wrist", "left index"),
            ("left wrist", "left thumb"),
            ("right wrist", "right pinky"),
            ("right wrist", "right index"),
            ("right wrist", "right thumb"),
            ("left ankle", "left heel"),
            ("left ankle", "left foot index"),
            ("right ankle", "right heel"),
            ("right ankle", "right foot index"),
        ]

        # Use matplotlib's RdYlGn colormap (Red-Yellow-Green)
        import matplotlib.pyplot as plt

        cmap = plt.cm.RdYlGn_r  # Reversed: Green=good, Red=bad

        # Draw connections with correction-based colors
        for (start_name, end_name), feature_idx in connections_with_features:
            if start_name in landmarks and end_name in landmarks:
                start_pos = landmarks[start_name]
                end_pos = landmarks[end_name]

                # Get correction angle for this feature
                correction_angle = abs(frame_corrections[feature_idx])

                # Map correction angle to color (0-20 degrees -> yellow to red)
                if correction_angle > 20:
                    correction_angle = 20
                color_intensity = correction_angle / 20.0
                color = cmap(color_intensity)

                # Draw thick line
                ax.plot(
                    [start_pos[0], end_pos[0]],
                    [start_pos[1], end_pos[1]],
                    color=color,
                    linewidth=8,
                    alpha=0.8,
                )

        # Draw additional connections in green (no correction)
        for start_name, end_name in additional_connections:
            if start_name in landmarks and end_name in landmarks:
                start_pos = landmarks[start_name]
                end_pos = landmarks[end_name]
                ax.plot(
                    [start_pos[0], end_pos[0]],
                    [start_pos[1], end_pos[1]],
                    color="green",
                    linewidth=6,
                    alpha=0.7,
                )

        # Draw landmarks as circles
        for name, pos in landmarks.items():
            ax.plot(pos[0], pos[1], "o", color="darkgreen", markersize=8, alpha=0.8)

        # Add title and colorbar
        ax.set_title(
            f"Correction Visualization (Frame {frame_idx})\nLight Green=0°, Red=20°+",
            fontsize=14,
            fontweight="bold",
            color="white",
        )

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=20))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label("Correction Angle (degrees)", color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        cbar.ax.tick_params(colors="white")

        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"Error creating correction visualization: {e}")
        return None
