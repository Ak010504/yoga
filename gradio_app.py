"""
gradio web interface for REAL-TIME CORRECTION ONLY (no classification).
SIMPLIFIED AUDIO: Says only the corrections, not the pose name.
Example: "Increase Left Elbow by 74.1 degrees" (not "Your Vrikshasana...")
"""

DISPLAY_NAME_MAPPING = {
    'chair': 'Utkatasana',
    'cobra': 'Bhujangasana',
    'downdog': 'Adho Mukha Svanasana',
    'goddess': 'Utkata Konasana',
    'surya_namaskar': 'Surya Namaskar',
    'tree': 'Vrikshasana',
    'warrior': 'Virabhadrasana',
}

# Ideal angle ranges for each pose
IDEAL_ANGLES = {
    'tree': {
        'left_elbow': (160, 180),
        'right_elbow': (160, 180),
        'left_hip': (140, 180),
        'right_hip': (140, 180),
    },
    'warrior': {
        'left_elbow': (80, 120),
        'right_elbow': (80, 120),
        'left_knee': (70, 100),
        'right_knee': (70, 100),
    },
    'chair': {
        'left_knee': (70, 100),
        'right_knee': (70, 100),
        'left_hip': (70, 110),
        'right_hip': (70, 110),
    },
    'downdog': {
        'left_elbow': (160, 180),
        'right_elbow': (160, 180),
        'left_hip': (160, 180),
        'right_hip': (160, 180),
    },
    'cobra': {
        'left_elbow': (70, 100),
        'right_elbow': (70, 100),
        'left_hip': (140, 180),
        'right_hip': (140, 180),
    },
    'goddess': {
        'left_knee': (70, 100),
        'right_knee': (70, 100),
        'left_hip': (60, 100),
        'right_hip': (60, 100),
    },
    'surya_namaskar': {
        'left_elbow': (80, 160),
        'right_elbow': (80, 160),
        'left_knee': (140, 180),
        'right_knee': (140, 180),
    },
}

import os
import tempfile
import warnings
import cv2
import gradio as gr
import mediapipe as mp
import numpy as np
import os
import tempfile
import subprocess
from collections import deque
from utils import (
    give_landmarks,
    render_landmarks,
    cal_angle,
)

# Import text-to-speech library
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("Warning: pyttsx3 not installed. Install with: pip install pyttsx3")

import pandas as pd
import torch

warnings.filterwarnings("ignore")
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

def merge_audio_with_video(video_path, audio_path, output_path):
    """
    Merge audio track with video file using ffmpeg.
    Creates downloadable video with embedded audio.
    """
    try:
        cmd = [
            'ffmpeg',
            '-i', video_path,        # Input video
            '-i', audio_path,        # Input audio
            '-c:v', 'copy',          # Copy video codec
            '-c:a', 'aac',           # Audio codec
            '-map', '0:v:0',         # Map video
            '-map', '1:a:0',         # Map audio
            '-y',                    # Overwrite
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print(f"✅ Audio merged: {output_path}")
            return output_path
        else:
            print(f"⚠️ FFmpeg error: {result.stderr}")
            return None
    except Exception as e:
        print(f"Error merging audio: {e}")
        return None


# Global state for real-time processing
class RealtimeCorrectionProcessor:
    """
    Real-time yoga correction analyzer without classification.
    Users manually select the pose, app provides corrections.
    """
    
    def __init__(self, max_history=30):
        self.frame_buffer = deque(maxlen=max_history)
        self.landmarks_history = deque(maxlen=max_history)
        self.frame_count = 0
        
    def process_frame(self, frame):
        """Process a single frame and extract landmarks."""
        try:
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.pose_landmarks:
                self.landmarks_history.append(results.pose_landmarks)
                self.frame_count += 1
                return results.pose_landmarks, True
            else:
                return None, False
        except Exception as e:
            print(f"Error processing frame: {e}")
            return None, False
    
    def get_current_angles(self, landmarks):
        """Extract angles from current landmarks."""
        try:
            if not landmarks:
                return None
            
            # landmarks is a NormalizedLandmarkList, access via .landmark
            lm_list = landmarks.landmark
            
            # Check if we have enough landmarks
            if len(lm_list) < 29:
                return None
            
            # Extract angles between key joints
            angles = {}
            
            # Elbow angles
            angles['left_elbow'] = cal_angle(
                (lm_list[11].x, lm_list[11].y),  # Left shoulder
                (lm_list[13].x, lm_list[13].y),  # Left elbow
                (lm_list[15].x, lm_list[15].y),  # Left wrist
            )
            angles['right_elbow'] = cal_angle(
                (lm_list[12].x, lm_list[12].y),
                (lm_list[14].x, lm_list[14].y),
                (lm_list[16].x, lm_list[16].y),
            )
            
            # Hip angles
            angles['left_hip'] = cal_angle(
                (lm_list[24].x, lm_list[24].y),
                (lm_list[26].x, lm_list[26].y),
                (lm_list[28].x, lm_list[28].y),
            )
            angles['right_hip'] = cal_angle(
                (lm_list[23].x, lm_list[23].y),
                (lm_list[25].x, lm_list[25].y),
                (lm_list[27].x, lm_list[27].y),
            )
            
            # Knee angles
            angles['left_knee'] = cal_angle(
                (lm_list[23].x, lm_list[23].y),
                (lm_list[26].x, lm_list[26].y),
                (lm_list[28].x, lm_list[28].y),
            )
            angles['right_knee'] = cal_angle(
                (lm_list[24].x, lm_list[24].y),
                (lm_list[25].x, lm_list[25].y),
                (lm_list[27].x, lm_list[27].y),
            )
            
            return angles
        except Exception as e:
            print(f"Error computing angles: {e}")
            return None

# Initialize processor
processor = RealtimeCorrectionProcessor()

def text_to_speech_realtime(text, output_path=None):
    """
    Convert text feedback to speech audio and return path.
    This audio will be automatically played in the UI.
    """
    if not TTS_AVAILABLE:
        return None
    
    try:
        clean_text = text.replace("**", "").replace("###", "").replace("#", "")
        clean_text = clean_text.replace("•", "").replace("\n", " ").strip()
        
        if not output_path:
            output_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        
        engine = pyttsx3.init()
        engine.setProperty('rate', 180)  # Faster for real-time
        engine.setProperty('volume', 0.95)
        
        try:
            voices = engine.getProperty('voices')
            if len(voices) > 1:
                engine.setProperty('voice', voices[1].id)
        except:
            pass
        
        engine.save_to_file(clean_text, output_path)
        engine.runAndWait()
        
        return output_path
    except Exception as e:
        print(f"Error generating speech: {e}")
        return None

def generate_correction_feedback(angles, selected_pose):
    """
    Generate correction feedback based on angles and selected pose.
    
    AUDIO: Says ONLY the corrections (e.g., "Increase Left Elbow by 74.1 degrees")
           NOT the pose name. For perfect form, says encouraging message.
    
    TEXT: Shows pose name + corrections (for UI display)
    
    Returns: (feedback_text, audio_path, severity, has_corrections)
    """
    if not angles or not selected_pose:
        return "", None, "info", False
    
    display_name = DISPLAY_NAME_MAPPING.get(selected_pose, selected_pose.title())
    pose_ideals = IDEAL_ANGLES.get(selected_pose, {})
    
    feedback_text = ""
    severity = "info"
    has_corrections = False
    
    try:
        corrections = []
        for angle_name, angle_value in angles.items():
            if angle_name in pose_ideals:
                ideal_min, ideal_max = pose_ideals[angle_name]
                
                if angle_value < ideal_min:
                    has_corrections = True
                    diff = ideal_min - angle_value
                    if diff > 20:
                        severity = "alert"
                    elif diff > 10:
                        severity = "warning"
                    
                    readable_name = angle_name.replace('_', ' ').title()
                    corrections.append(
                        f"Increase {readable_name} by {diff:.1f} degrees"
                    )
                elif angle_value > ideal_max:
                    has_corrections = True
                    diff = angle_value - ideal_max
                    if diff > 20:
                        severity = "alert"
                    elif diff > 10:
                        severity = "warning"
                    
                    readable_name = angle_name.replace('_', ' ').title()
                    corrections.append(
                        f"Decrease {readable_name} by {diff:.1f} degrees"
                    )
        
        if corrections:
            # TEXT FEEDBACK: Include pose name (for UI)
            feedback_text = f"**{display_name}** - Adjustments needed:\n"
            feedback_text += "\n".join(corrections)
            
            # AUDIO FEEDBACK: ONLY the corrections (no pose name)
            # Join with period for natural speech
            audio_text = ". ".join(corrections)
        else:
            # PERFECT FORM: Full message for both text and audio
            feedback_text = f"✅ Great! Your **{display_name}** form looks perfect!"
            audio_text = f"Great! Your {display_name} form looks perfect!"
            severity = "success"
        
        # Generate audio with appropriate text
        audio_path = text_to_speech_realtime(audio_text)
        
        return feedback_text, audio_path, severity, has_corrections
    
    except Exception as e:
        print(f"Error generating feedback: {e}")
        return f"Error: {str(e)}", None, "alert", False
def process_frame_live(frame, selected_pose, feedback_state):
    """
    Live webcam processing: single frame → corrected frame + updated feedback + audio.
    frame: HxWx3 (numpy, BGR) from webcam
    selected_pose: key from DISPLAY_NAME_MAPPING
    feedback_state: accumulated feedback text (gr.State)
    """
    if frame is None or not selected_pose:
        return frame, (feedback_state or ""), None

    # Use the same processor & pose graph
    landmarks, success = processor.process_frame(frame)
    if not success or landmarks is None:
        return frame, (feedback_state or ""), None

    angles = processor.get_current_angles(landmarks)
    if not angles:
        return frame, (feedback_state or ""), None

    # Feedback & audio (same logic as video path)
    feedback_text, audio_path, severity, has_corrections = generate_correction_feedback(
        angles, selected_pose
    )

    # Update feedback log (keep last 10 lines)
    if feedback_text:
        if feedback_state is None:
            feedback_state = ""
        feedback_state = feedback_state + f"\n{feedback_text}"
        feedback_state = "\n".join(feedback_state.splitlines()[-10:])

    # OPTIONAL: build simple correction_data so render_landmarks can color joints
    try:
        angle_vec = np.zeros(9, dtype=float)
        angle_vec[0] = angles.get('left_elbow', 0.0)
        angle_vec[1] = angles.get('right_elbow', 0.0)
        angle_vec[2] = angles.get('left_hip', 0.0)
        angle_vec[3] = angles.get('right_hip', 0.0)
        angle_vec[4] = angles.get('left_knee', 0.0)
        angle_vec[5] = angles.get('right_knee', 0.0)
        correction_data = {
            "input_data": angle_vec.reshape(1, -1),
            "corrected_data": angle_vec.reshape(1, -1),
        }

        frame = render_landmarks(
            frame,
            landmarks,
            correction_data=correction_data,
            frameidx=0,
            seqbounds=None
        )
    except Exception as e:
        print(f"Error in render_landmarks (live): {e}")

    return frame, feedback_state, audio_path


def process_video_with_corrections(video_file, selected_pose):
    """
    Process video in real-time with corrections for the selected pose.
    No classification - directly use the user-selected pose.
    Generates audio that will auto-play.
    """
    feedback_log = []
    audio_files = []
    
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Temporary video file for rendered output
    temp_video_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    fourcc = cv2.VideoWriter_fourcc(*"H264")
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (frame_width, frame_height))
    
    if not out.isOpened():
        print("MJPEG not available, trying H264...")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        fourcc = -1
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (frame_width, frame_height))
    processor.frame_count = 0
    frame_idx = 0
    last_audio_path = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        # Process frame
        landmarks, success = processor.process_frame(frame)
        
        if success and landmarks:
            angles = processor.get_current_angles(landmarks)
            
            # Generate feedback every 2 seconds (60 frames at 30fps)
            if frame_idx % 60 == 0 and angles:
                feedback_text, audio_path, severity, has_corrections = generate_correction_feedback(angles, selected_pose)
                
                feedback_log.append({
                    'timestamp': frame_idx / fps,
                    'pose': selected_pose,
                    'feedback': feedback_text,
                    'severity': severity,
                    'has_corrections': has_corrections
                })
                
                if audio_path:
                    audio_files.append(audio_path)
                    last_audio_path = audio_path  # Keep track of latest audio
            
            # Render landmarks
            try:
                frame = render_landmarks(frame, landmarks)
            except:
                pass
        
        # Write frame to output video
        out.write(frame)
        
        # Yield progress (every 60 frames)
        if frame_idx % 10 == 0:
            yield {
                'video_path': temp_video_path,
                'feedback_log': feedback_log,
                'audio_files': audio_files,
                'frame_count': frame_idx,
                'progress': f"Processing... {frame_idx} frames processed",
                'last_audio': last_audio_path  # Latest audio for auto-play
            }
    
    cap.release()
    out.release()
    
    # Final yield with complete results
    yield {
        'video_path': temp_video_path,
        'feedback_log': feedback_log,
        'audio_files': audio_files,
        'frame_count': frame_idx,
        'progress': f"✅ Complete! {frame_idx} frames processed",
        'last_audio': last_audio_path
    }

def create_gradio_interface():
    """Create Gradio dashboard for real-time correction with auto-play audio."""
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        text_size="md",
        font=["Inter", "system-ui", "sans-serif"],
    )

    custom_css = """
    .gradio-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        min-height: 100vh;
    }
    .main-content {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    .live-section {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
    }
    .success { color: #10b981; }
    .warning { color: #f59e0b; }
    .alert { color: #ef4444; }
    
    /* Auto-play audio element */
    audio {
        width: 100%;
        margin-top: 10px;
    }
    """

    with gr.Blocks(title="PosePilot - Real-Time Yoga Correction", theme=theme, css=custom_css) as app:
        
        with gr.Column(elem_classes="main-content"):
            gr.Markdown("# 🧘 Real-Time Yoga Correction System")
            gr.Markdown("Select your asana, upload video **OR** use webcam for **instant audio feedback** on your posture")
            
            # Global audio (auto-plays on all tabs)
            gr.Markdown("**🔔 Audio auto-plays on all tabs**")
            global_audio = gr.Audio(
                label="Global Audio (Auto-plays)",
                interactive=False,
                autoplay=True,
                visible=True
            )
            
            with gr.Row():
                # Input section
                with gr.Column(scale=1):
                    gr.Markdown("### 📋 Setup")
                    
                    # Dropdown to select pose
                    pose_dropdown = gr.Dropdown(
                        choices=list(DISPLAY_NAME_MAPPING.keys()),
                        label="Select Yoga Asana",
                        value="tree",
                        info="Choose the yoga pose you're performing"
                    )
                    
                    gr.Markdown("### 📹 Upload Video (Button Required)")
                    video_input = gr.Video(label="Your Yoga Video", height=250)
                    analyze_btn = gr.Button("▶️ Analyze & Correct", variant="primary", size="lg")
                    
                    gr.Markdown("### 📷 Live Webcam (Instant - No Button)")
                    webcam_input = gr.Image(
                        sources=["webcam"],
                        streaming=True,
                        label="Webcam Input (Auto-starts)",
                        type="numpy",
                        height=250,
                    )
                
                # Results section
                with gr.Column(scale=2, elem_classes="live-section"):
                    gr.Markdown("### 📊 Real-Time Feedback")
                    
                    progress_status = gr.Textbox(
                        label="Processing Status",
                        value="Ready to analyze",
                        interactive=False
                    )
                    
                    with gr.Tabs():
                        with gr.Tab("🎬 Uploaded Video Stream"):
                            live_video = gr.Video(
                                interactive=False,
                                height=300,
                                format="mp4",
                                label="Video with Landmarks & Corrections"
                            )
                        
                        with gr.Tab("📷 Live Webcam Stream"):
                            live_webcam = gr.Image(
                                label="Live Webcam Corrections",
                                interactive=False,
                                height=300,
                            )
                        
                        with gr.Tab("🔊 Live Audio Feedback"):
                            gr.Markdown("**🔔 Audio plays automatically - Just the corrections, no pose name**")
                            audio_stream = gr.Audio(
                                label="Real-Time Corrections (Auto-Play)",
                                interactive=False,
                                autoplay=True
                            )
                        
                        with gr.Tab("📋 Feedback Log"):
                            feedback_display = gr.Textbox(
                                label="Real-Time Corrections Timeline",
                                lines=10,
                                interactive=False,
                                max_lines=20
                            )

        # State for live webcam feedback accumulation
        feedback_state = gr.State("")

        # Define streaming callback (your existing uploaded video logic)
        def analyze_video(video_file, selected_pose):
            if video_file is None or not selected_pose:
                yield (
                    None,
                    "❌ Please select an asana and upload a video",
                    "No input provided",
                    None,
                    None
                )
                return
            
            feedback_lines = []  # Store unique feedback only
            latest_audio = None
            last_timestamp = None  # Track last shown timestamp
            
            for result in process_video_with_corrections(video_file, selected_pose):
                feedback_log = result.get('feedback_log', [])
                
                # Only add NEW feedback (not already shown)
                if feedback_log:
                    latest_feedback = feedback_log[-1]
                    current_timestamp = latest_feedback['timestamp']
                    
                    # Only add if timestamp is DIFFERENT from last one
                    if current_timestamp != last_timestamp:
                        feedback_lines.append(
                            f"[{current_timestamp:.1f}s] {latest_feedback['feedback']}"
                        )
                        last_timestamp = current_timestamp
                
                # Get latest audio file if available (will auto-play)
                if result.get('last_audio'):
                    latest_audio = result['last_audio']
                
                # Build feedback text from unique lines (show last 10)
                feedback_text = "\n".join(feedback_lines[-10:])
                
                yield (
                    result.get('video_path'),
                    result.get('progress'),
                    feedback_text,
                    latest_audio,
                    latest_audio
                )

        # Live webcam processing (NEW - runs automatically)
        def process_frame_live(frame, selected_pose, feedback_state):
            if frame is None or not selected_pose:
                return frame, (feedback_state or ""), None

            # Use the same processor & pose graph
            landmarks, success = processor.process_frame(frame)
            if not success or landmarks is None:
                return frame, (feedback_state or ""), None

            angles = processor.get_current_angles(landmarks)
            if not angles:
                return frame, (feedback_state or ""), None

            # Feedback & audio (same logic as video path)
            feedback_text, audio_path, severity, has_corrections = generate_correction_feedback(
                angles, selected_pose
            )

            # Update feedback log (keep last 10 lines)
            if feedback_text:
                if feedback_state is None:
                    feedback_state = ""
                feedback_state = feedback_state + f"\n{feedback_text}"
                feedback_state = "\n".join(feedback_state.splitlines()[-10:])

            # Render landmarks (your existing function)
            try:
                frame = render_landmarks(frame, landmarks)
            except Exception as e:
                print(f"Error in render_landmarks (live): {e}")

            return frame, feedback_state, audio_path

        # Connect uploaded video button (your existing logic)
        analyze_btn.click(
            fn=analyze_video,
            inputs=[video_input, pose_dropdown],
            outputs=[live_video, progress_status, feedback_display, audio_stream, global_audio]
        )
        
        # Live streaming from webcam: no button, runs automatically
        webcam_input.stream(
            fn=process_frame_live,
            inputs=[webcam_input, pose_dropdown, feedback_state],
            outputs=[live_webcam, feedback_display, global_audio],
            stream_every=0.1,  # seconds between frames
        )
    
    return app

def main():
    """Launch the Gradio app."""
    app = create_gradio_interface()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )

if __name__ == "__main__":
    main()