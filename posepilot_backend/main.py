#!/usr/bin/env python3
# ============================================================================
# PosePilot FastAPI Backend - main.py (FIXED VERSION with Angle Velocity)
# Uses angle-based velocity gating to detect stable phases
# ============================================================================

import os
import cv2
import json
import pickle
import tempfile
import asyncio
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import logging
from scipy.signal import find_peaks

import torch
import torch.nn as nn
from fastapi import FastAPI, WebSocket, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import mediapipe as mp
from correction_predict import predict_correction_from_dataframe, calculate_angle_velocity


# Import your existing model classes and prediction functions
from classify_model import ClassifyPose
from correction_model import CorrModel
from classify_predict import config_model, predict
from correction_predict import predict_correction_from_dataframe, calculate_angle_velocity
from utils import (
    cal_angle, cal_error, 
    structure_data, update_body_pose_landmarks,
    correction_angles_convert, equal_rows
)

# ============================================================================
# CONFIGURATION
# ============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model paths
BASE_DIR = Path(__file__).resolve().parent      # D:\PosePilot\posepilot_backend
PROJECT_ROOT = BASE_DIR.parent                  # D:\PosePilot
MODELS_DIR = PROJECT_ROOT / "models"  
CLASSIFY_MODEL_PATH = MODELS_DIR / "pose_classification_model.pth"
CLASSIFY_SCALER_PATH = MODELS_DIR / "classify_scaler.pkl"
POSE_MAPPING_PATH = MODELS_DIR / "pose_mapping.pkl"

# Correction models directory (one per pose)
CORRECTION_MODELS = {
    "chair": {"model": MODELS_DIR / "chair_correction_model.pth", "scalers": MODELS_DIR / "chair_correction_scalers.pkl"},
    "cobra": {"model": MODELS_DIR / "cobra_correction_model.pth", "scalers": MODELS_DIR / "cobra_correction_scalers.pkl"},
    "downdog": {"model": MODELS_DIR / "downdog_correction_model.pth", "scalers": MODELS_DIR / "downdog_correction_scalers.pkl"},
    "goddess": {"model": MODELS_DIR / "goddess_correction_model.pth", "scalers": MODELS_DIR / "goddess_correction_scalers.pkl"},
    "surya_namaskar": {"model": MODELS_DIR / "surya_namaskar_correction_model.pth", "scalers": MODELS_DIR / "surya_namaskar_correction_scalers.pkl"},
    "tree": {"model": MODELS_DIR / "tree_correction_model.pth", "scalers": MODELS_DIR / "tree_correction_scalers.pkl"},
    "warrior": {"model": MODELS_DIR / "warrior_correction_model.pth", "scalers": MODELS_DIR / "warrior_correction_scalers.pkl"}
}

# ✅ Average lengths for each pose (from training)
AVERAGE_LENGTHS = {
    'chair': 85, 'cobra': 144, 'downdog': 120, 'goddess': 91,
    'surya_namaskar': 122, 'tree': 110, 'warrior': 103
}

# ✅ Pose-specific velocity thresholds (degrees/frame)
POSE_VELOCITY_THRESHOLDS = {
    'tree': 1.5,          # Very stable pose
    'warrior': 2.0,       # Moderately stable
    'chair': 2.5,         # More movement expected
    'downdog': 2.0,
    'cobra': 1.8,
    'goddess': 2.2,
    'surya_namaskar': 3.0  # Flow sequence, more movement
}

# Device selection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

# ============================================================================
# FASTAPI APP SETUP
# ============================================================================

app = FastAPI(title="PosePilot Backend", version="1.0.0")

# Enable CORS for mobile PWA
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# GLOBAL STATE (loaded on startup)
# ============================================================================

class ModelState:
    def __init__(self):
        self.classify_model = None
        self.classify_scaler = None
        self.pose_mapping = None
        self.correction_models = {}
        self.correction_scalers = {}
        self.processor = None  # MediaPipe

model_state = ModelState()

# Initialize MediaPipe Pose (module level for frame processing)
mp_pose = mp.solutions.pose
pose_processor = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ============================================================================
# HELPER FUNCTIONS FOR LANDMARK EXTRACTION (LIKE GRADIO)
# ============================================================================

def generate_text_feedback(corrections_dict):
    """
    Convert ML correction dictionary (f1–f9) into readable feedback.
    Matches Gradio's feedback behavior.
    """

    feature_names = {
        "f1": "Left Elbow",
        "f2": "Right Elbow",
        "f3": "Left Hip",
        "f4": "Right Hip",
        "f5": "Left Knee",
        "f6": "Right Knee",
        "f7": "Neck",
        "f8": "Left Shoulder",
        "f9": "Right Shoulder"
    }

    feedback_items = []

    for key, mean_dev in corrections_dict.items():
        name = feature_names.get(key, key)

        if abs(mean_dev) < 5:
            continue  # ignore minimal deviations

        # Determine action
        if "Elbow" in name:
            action = "Straighten" if mean_dev > 0 else "Bend"

        elif "Hip" in name:
            action = "Open" if mean_dev > 0 else "Tuck"

        elif "Knee" in name:
            action = "Straighten" if mean_dev > 0 else "Bend deeper"

        elif "Neck" in name:
            action = "Lift" if mean_dev > 0 else "Lower"

        elif "Shoulder" in name:
            action = "Raise" if mean_dev > 0 else "Lower"

        else:
            action = "Adjust"

        feedback_items.append(
            f"{action} your **{name.lower()}** by approximately **{abs(mean_dev):.1f}°**"
        )

    if not feedback_items:
        return "Good alignment! Continue holding."

    return " | ".join(feedback_items)


def extract_landmarks_from_video_frames(video_path, fps=30):
    """Extract landmarks frame-by-frame (SAME LOGIC AS GRADIO APP)"""
    landmarks_list = []
    frame_count = 0
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("❌ Failed to open video file")
        return None, False
    
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"📹 Video has {total_frames} total frames")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame with MediaPipe (EXACTLY LIKE GRADIO)
            results = pose_processor.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.pose_landmarks:
                landmarks_row = []
                for lm in results.pose_landmarks.landmark:
                    landmarks_row.extend([lm.x, lm.y, lm.z, lm.visibility])
                
                landmarks_list.append(landmarks_row)
                frame_count += 1
                
                if frame_count % 50 == 0:
                    logger.info(f"  ✅ Processed {frame_count} frames with landmarks detected")
        
        cap.release()
        
        if not landmarks_list:
            logger.error("❌ No frames with landmarks detected in entire video")
            return None, False
        
        # Convert to DataFrame (compatible with your utils)
        columns = []
        for i in range(33):
            columns.extend([f'lm_{i}_x', f'lm_{i}_y', f'lm_{i}_z', f'lm_{i}_vis'])
        
        landmarks_df = pd.DataFrame(landmarks_list, columns=columns)
        total_values = (landmarks_df != 0).sum().sum()
        logger.info(f"✅ Extracted {len(landmarks_df)} frames with {total_values} non-zero landmark values")
        
        return landmarks_df, True
    
    except Exception as e:
        logger.error(f"❌ Error extracting landmarks: {e}", exc_info=True)
        return None, False

# ============================================================================
# HELPER FUNCTIONS FOR FEEDBACK GENERATION
# ============================================================================

def generate_feedback(corrections, pose):
    """Generate human-readable feedback from correction model output"""
    if not corrections:
        return "Good alignment! Continue holding."
    
    feedback_parts = []
    try:
        if isinstance(corrections, dict):
            for joint, correction_val in corrections.items():
                if isinstance(correction_val, (int, float)) and abs(correction_val) > 0.3:
                    if "left" in str(joint).lower():
                        feedback_parts.append(f"Adjust left side")
                    elif "right" in str(joint).lower():
                        feedback_parts.append(f"Adjust right side")
                    else:
                        feedback_parts.append(f"Adjust alignment")
        elif isinstance(corrections, (list, np.ndarray)):
            corrections_list = [c for c in corrections if isinstance(c, (int, float))]
            if corrections_list:
                avg_correction = np.mean([abs(c) for c in corrections_list])
                if avg_correction > 0.5:
                    feedback_parts.append("Significant form adjustments needed")
                elif avg_correction > 0.3:
                    feedback_parts.append("Minor form adjustments needed")
    except Exception as e:
        logger.warning(f"Error generating feedback: {e}")
    
    if feedback_parts:
        return "Corrections needed: " + " | ".join(feedback_parts[:3])
    return "Good alignment! Continue holding."

def determine_severity(corrections):
    """Determine severity level based on correction magnitude"""
    if not corrections:
        return "positive"
    
    try:
        if isinstance(corrections, dict):
            corrections_list = [abs(v) for v in corrections.values() if isinstance(v, (int, float))]
        elif isinstance(corrections, (list, np.ndarray)):
            corrections_list = [abs(c) for c in corrections if isinstance(c, (int, float))]
        else:
            return "positive"
        
        if not corrections_list:
            return "positive"
        
        avg_correction = np.mean(corrections_list)
        if avg_correction > 0.7:
            return "critical"
        elif avg_correction > 0.4:
            return "warning"
        else:
            return "positive"
    except Exception as e:
        logger.warning(f"Error determining severity: {e}")
        return "positive"

# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

def load_classification_model():
    """Load classification model and scaler"""
    logger.info("Loading classification model...")
    try:
        with open(POSE_MAPPING_PATH, 'rb') as f:
            pose_mapping = pickle.load(f)
        logger.info(f"Loaded pose mapping: {pose_mapping}")
        
        with open(CLASSIFY_SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        
        input_size = 680
        hidden_size = 32
        num_layers = 1
        sequence_length = 10
        num_classes = len(pose_mapping)
        
        model = ClassifyPose(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            sequence_length=sequence_length,
            num_classes=num_classes
        )
        
        model.load_state_dict(torch.load(CLASSIFY_MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        
        logger.info("✅ Classification model loaded successfully")
        return model, scaler, pose_mapping
    except Exception as e:
        logger.error(f"❌ Failed to load classification model: {e}")
        raise

def load_correction_models():
    """Load all correction models for each pose"""
    logger.info("Loading correction models...")
    
    correction_models = {}
    correction_scalers = {}
    
    for pose_name, paths in CORRECTION_MODELS.items():
        try:
            logger.info(f"Loading correction model for {pose_name}...")
            
            with open(paths["scalers"], 'rb') as f:
                scalers = pickle.load(f)
            correction_scalers[pose_name] = scalers
            
            input_size = 9
            hidden_size = 256
            num_layers = 1
            num_classes = 9
            
            model = CorrModel(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_classes=num_classes
            )
            
            model.load_state_dict(torch.load(paths["model"], map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            
            correction_models[pose_name] = model
            logger.info(f"✅ Loaded correction model for {pose_name}")
        except Exception as e:
            logger.error(f"❌ Failed to load correction model for {pose_name}: {e}")
    
    return correction_models, correction_scalers

@app.on_event("startup")
async def startup():
    """Load models on startup"""
    logger.info("🚀 Starting PosePilot Backend...")
    
    try:
        model_state.classify_model, model_state.classify_scaler, model_state.pose_mapping = \
            load_classification_model()
        
        model_state.correction_models, model_state.correction_scalers = \
            load_correction_models()
        
        logger.info("✅ All models loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

# ============================================================================
# REST API ENDPOINTS
# ============================================================================

@app.get("/api/poses")
async def get_poses():
    """Get available yoga poses"""
    try:
        poses = list(model_state.pose_mapping.keys())
        return {
            "poses": poses,
            "count": len(poses),
            "display_names": {pose: pose.capitalize() for pose in poses}
        }
    except Exception as e:
        logger.error(f"Error getting poses: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/process-video")
async def process_video(video: UploadFile = File(...), pose: str = Form(...)):
    """
    Process uploaded video file and return corrections for selected pose.
    Uses ANGLE-BASED velocity gating to select stable phases for feedback.
    """
    if pose not in model_state.correction_models:
        return JSONResponse(status_code=400, content={"error": f"Pose '{pose}' not found"})

    temp_video = None
    try:
        # Save uploaded video temporarily
        temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        content = await video.read()
        temp_video.write(content)
        temp_video.close()
        import time

        request_start_time = time.perf_counter()
        logger.info("⏱️ Loading time measurement started")


        logger.info(f"Processing video for pose: {pose}")

        # ✅ STEP 1: Frame-by-frame landmark extraction
        landmarks_df, success = extract_landmarks_from_video_frames(temp_video.name, fps=30)

        if not success or landmarks_df is None or len(landmarks_df) == 0:
            logger.error("❌ Failed to extract landmarks from video")
            return JSONResponse(status_code=400, content={
                "error": "Could not detect body pose in video. Please ensure:",
                "details": ["Your full body is visible", "Lighting is adequate", "Video quality is good"]
            })

        logger.info(f"✅ Extracted {len(landmarks_df)} frames from video")

        # ✅ STEP 2: Calculate error metrics
        try:
            landmarks_df = cal_error(landmarks_df)
            logger.info(f"✅ Calculated error metrics")
        except Exception as e:
            logger.error(f"❌ Error calculating metrics: {e}")
            return JSONResponse(status_code=500, content={"error": f"Failed to calculate pose metrics: {str(e)}"})

        max_error = landmarks_df['error'].max()
        if max_error == 0:
            logger.error("❌ All error values are zero")
            return JSONResponse(status_code=400, content={"error": "No pose variation detected. Try moving more."})

        logger.info(f"✅ Max error value: {max_error:.4f}")
        logger.info(f"✅ Using selected pose: {pose}")

        # ✅ STEP 3: Apply ANGLE-BASED VELOCITY GATING (FIXED VERSION)
        logger.info("📊 Applying angle-based velocity gating to entire video...")

        try:
            # Convert landmarks to angle features FIRST
            # ✅ FORCE exactly 132 landmark columns (33 × 4)
            landmarks_df = landmarks_df.iloc[:, :132].copy()
            landmarks_df.reset_index(drop=True, inplace=True)

            structured_df, body_pose_landmarks = structure_data(landmarks_df)

            structured_df, body_pose_landmarks = update_body_pose_landmarks(
                structured_df, body_pose_landmarks
            )
            angle_features_df = correction_angles_convert(structured_df)
            
            logger.info(f"✅ Converted to angle features: {angle_features_df.shape}")

            # Get pose-specific velocity threshold
            velocity_threshold = POSE_VELOCITY_THRESHOLDS.get(pose, 2.0)
            logger.info(f"📊 Using velocity threshold {velocity_threshold}°/frame for {pose}")

            # Apply angle-based velocity gating
            velocities_smooth, phase_midpoints = calculate_angle_velocity(
                angle_features_df,
                velocity_threshold=velocity_threshold,
                min_phase_duration=3
            )

            logger.info(f"✅ Angle velocity gating: Found {len(phase_midpoints)} stable phases")

            if len(phase_midpoints) == 0:
                logger.warning("⚠️ No stable phases found, using peak detection fallback")
                error_values = landmarks_df['error'].values
                peaks, _ = find_peaks(error_values, distance=3, prominence=0.001)
                selected_frames = peaks
            else:
                selected_frames = phase_midpoints

        except Exception as e:
            logger.warning(f"⚠️ Angle velocity gating failed: {e}, using peak detection instead")
            import traceback
            traceback.print_exc()
            error_values = landmarks_df['error'].values
            peaks, _ = find_peaks(error_values, distance=3, prominence=0.001)
            selected_frames = peaks

        # ✅ STEP 4: Limit feedback frames (cap at 20 for performance)
        if len(selected_frames) > 20:
            error_values = landmarks_df['error'].values
            frame_errors = error_values[selected_frames]
            top_indices = np.argsort(frame_errors)[-20:]
            selected_frames = selected_frames[top_indices]
            selected_frames = np.sort(selected_frames)
            logger.info(f"✅ Limited to top {len(selected_frames)} stable phases by error")

        logger.info(f"✅ Selected {len(selected_frames)} frames for feedback (DYNAMIC)")

        if len(selected_frames) == 0:
            # Fallback: use uniform sampling if no frames detected
            logger.warning("⚠️ No frames selected, using uniform sampling")
            total_frames = len(landmarks_df)
            selected_frames = np.linspace(0, total_frames-1, num=min(10, total_frames), dtype=int)

        # ✅ STEP 5: Generate feedback for each selected frame
        feedback_log = []

        for frame_idx, frame_index in enumerate(selected_frames, 1):
            try:
                frame_row = landmarks_df.iloc[frame_index]
                frame_data = pd.DataFrame([frame_row])

                if 'error' in frame_data.columns:
                    frame_data = frame_data.drop('error', axis=1)

                # Run ML correction model
                corrections = predict_correction_from_dataframe(frame_data, pose)

                if not corrections or "input_data" not in corrections:
                    feedback_text = "Good alignment! Continue holding."
                    severity = "positive"
                    corrections_dict = {}
                else:
                    # Extract last frame's differences
                    last_output = corrections["corrected_data"][-1]
                    last_input = corrections["input_data"][-1]

                    # Build difference dict f1–f9
                    corrections_dict = {
                        f"f{i+1}": float(last_output[i] - last_input[i])
                        for i in range(9)
                    }

                    # Show only >15° corrections
                    high_error = {
                        k: v for k, v in corrections_dict.items()
                        if abs(v) >= 15
                    }

                    if len(high_error) > 0:
                        feedback_text = generate_text_feedback(high_error)
                        severity = "alert"
                        display_dict = high_error
                    else:
                        feedback_text = generate_text_feedback(corrections_dict)
                        severity = "warning"
                        display_dict = corrections_dict

                # Compute timestamp (at 30 fps)
                timestamp_sec = round(frame_index / 30.0, 2)

                logger.info(f"Stable Phase {frame_idx} @ {timestamp_sec}s → {feedback_text}")

                feedback_log.append({
                    "timestamp": timestamp_sec,
                    "frame_index": frame_idx,
                    "feedback": feedback_text,
                    "severity": severity,
                    "angles": display_dict,
                })

            except Exception as e:
                logger.error(f"Error processing frame {frame_idx}: {e}", exc_info=True)
                feedback_log.append({
                    "timestamp": round(frame_index / 30.0, 2),
                    "frame_index": frame_idx,
                    "feedback": "Frame ready",
                    "severity": "positive",
                    "angles": {}
                })

        if not feedback_log:
            return JSONResponse(status_code=500, content={"error": "Failed to generate feedback for any frames"})
        
        request_end_time = time.perf_counter()
        total_loading_time = round((request_end_time - request_start_time) * 1000, 2)

        logger.info(
            f"⏳ TOTAL LOADING TIME (upload → feedback): {total_loading_time} ms"
        )


        return {
            "status": "success",
            "pose": pose,
            "feedback_count": len(feedback_log),
            "feedback_log": feedback_log
        }

    except Exception as e:
        logger.error(f"❌ Error processing video: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"Video processing failed: {str(e)}"})

    finally:
        from metrics import Metrics
        metrics = Metrics(tag="PROCESS_VIDEO")
        sys_metrics = metrics.system()

        logger.info(
            f"✅ Request completed in {metrics.total()} ms | "
            f"CPU={sys_metrics['cpu_percent']}% | "
            f"RAM={sys_metrics['ram_mb']} MB | "
            f"Device={sys_metrics['gpu']}"
        )

        if temp_video and os.path.exists(temp_video.name):
            os.remove(temp_video.name)

# ============================================================================
# HEALTH CHECK ENDPOINT
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": DEVICE,
        "models_loaded": {
            "classification": model_state.classify_model is not None,
            "correction": len(model_state.correction_models) > 0,
            "mediapipe": pose_processor is not None
        }
    }

@app.get("/")
async def root():
    """Root endpoint with API documentation"""
    return {
        "name": "🧘 PosePilot Backend API",
        "version": "1.0.0",
        "endpoints": {
            "poses": "GET /api/poses",
            "process_video": "POST /api/process-video",
            "health": "GET /health"
        },
        "docs": "http://localhost:8000/docs"
    }

# ============================================================================
# MAIN - Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )