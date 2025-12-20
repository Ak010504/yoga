import cv2
import mediapipe as mp
import pandas as pd
import os
from pathlib import Path

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def process_video_to_csv(video_path, output_csv_path):
    """
    Process a video file and extract pose landmarks to CSV
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    # List to store all frames' data
    all_frames_data = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Extract landmarks
            landmarks = results.pose_landmarks.landmark
            
            # Create a row with all landmark data
            row_data = {}
            for idx, landmark in enumerate(landmarks, start=1):
                row_data[f'x{idx}'] = landmark.x
                row_data[f'y{idx}'] = landmark.y
                row_data[f'z{idx}'] = landmark.z
                row_data[f'v{idx}'] = landmark.visibility
            
            all_frames_data.append(row_data)
        else:
            print(f"Warning: No pose detected in frame {frame_count} of {video_path}")
        
        frame_count += 1
    
    cap.release()
    
    # Create DataFrame and save to CSV
    if all_frames_data:
        df = pd.DataFrame(all_frames_data)
        df.to_csv(output_csv_path, index=False)
        print(f" Processed {video_path.name}: {len(all_frames_data)} frames saved to {output_csv_path}")
    else:
        print(f" No pose data extracted from {video_path}")

def process_all_videos(input_folder, output_folder):
    """
    Process all videos in the input folder and save CSVs to output folder
    """
    # Create output directory if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Get all video files (common video extensions)
    video_extensions = ['.mp4']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(input_folder.glob(f'*{ext}'))
        video_files.extend(input_folder.glob(f'*{ext.upper()}'))
    
    if not video_files:
        print(f"No video files found in {input_folder}")
        return
    
    print(f"Found {len(video_files)} video(s) to process\n")
    
    # Process each video
    for video_path in video_files:
        # Create output CSV filename (same as video but with .csv extension)
        csv_filename = video_path.stem + '.csv'
        output_csv_path = output_folder / csv_filename
        
        print(f"Processing: {video_path.name}")
        process_video_to_csv(video_path, output_csv_path)
        print()

if __name__ == "__main__":
    # Define paths
    input_folder = Path("raw_vids/suryanamaskar")
    output_folder = Path("data/surya_namaskar")
    
    # Check if input folder exists
    if not input_folder.exists():
        print(f"Error: Input folder '{input_folder}' does not exist!")
        print("Please make sure the folder path is correct.")
    else:
        print("=" * 60)
        print("MediaPipe Pose Video to CSV Converter")
        print("=" * 60)
        print(f"Input folder:  {input_folder}")
        print(f"Output folder: {output_folder}")
        print("=" * 60 + "\n")
        
        process_all_videos(input_folder, output_folder)
        
        print("=" * 60)
        print("Processing complete!")
        print("=" * 60)
    
    # Cleanup
    pose.close()