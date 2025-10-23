"""
gradio web interface for pose classification and correction analysis.
Updated with modern UI and fixed correction data logic.
"""

import os
import tempfile
import warnings

import cv2
import gradio as gr
import mediapipe as mp
from classify_predict import predict_from_dataframe
from correction_predict import predict_correction_from_dataframe
from utils import (
    give_landmarks,
    render_landmarks,
    generate_correction_graph,
    calculate_sequence_bounds,
)

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


def generate_correction_feedback(correction_data, predicted_pose):
    """
    Generate human-readable correction feedback from correction data.
    
    Parameters
    ----------
    correction_data : dict
        Dictionary containing input_data, corrected_data, and reference_data
    predicted_pose : str
        Name of the predicted pose
    
    Returns
    -------
    str
        Markdown-formatted correction feedback
    """
    if correction_data is None or correction_data.get("status") != "success":
        return "**No corrections available.**"
    
    try:
        import numpy as np
        
        input_data = correction_data["input_data"]  # Current pose angles
        corrected_data = correction_data["corrected_data"]  # Ideal/corrected angles
        
        feature_names = [
            "Left Elbow",
            "Right Elbow", 
            "Left Hip",
            "Right Hip",
            "Left Knee",
            "Right Knee",
            "Neck",
            "Left Shoulder",
            "Right Shoulder"
        ]
        
        # Verify shapes match
        if input_data.shape != corrected_data.shape:
            print(f"Warning: Shape mismatch - input: {input_data.shape}, corrected: {corrected_data.shape}")
            # Handle mismatch by using minimum length
            min_len = min(input_data.shape[0], corrected_data.shape[0])
            input_data = input_data[:min_len]
            corrected_data = corrected_data[:min_len]
        
        # Calculate average deviation for each joint
        feedback_items = []
        
        for i, feature_name in enumerate(feature_names):
            # ✅ FIXED: Use full arrays - they're already aligned
            input_angles = input_data[:, i]  # (110,)
            corrected_angles = corrected_data[:, i]  # (110,)
            
            # Calculate mean deviation
            deviations = corrected_angles - input_angles
            mean_deviation = np.mean(deviations)
            std_deviation = np.std(deviations)
            
            # Only provide feedback if deviation is significant (> 5 degrees average)
            if abs(mean_deviation) > 5:
                # Determine direction and action
                if "Elbow" in feature_name:
                    if mean_deviation > 0:
                        action = f"**Straighten** your {feature_name.lower()} by approximately **{abs(mean_deviation):.1f}°**"
                    else:
                        action = f"**Bend** your {feature_name.lower()} by approximately **{abs(mean_deviation):.1f}°**"
                
                elif "Hip" in feature_name:
                    if mean_deviation > 0:
                        action = f"**Open** your {feature_name.lower()} more by approximately **{abs(mean_deviation):.1f}°**"
                    else:
                        action = f"**Tuck** your {feature_name.lower()} by approximately **{abs(mean_deviation):.1f}°**"
                
                elif "Knee" in feature_name:
                    if mean_deviation > 0:
                        action = f"**Straighten** your {feature_name.lower()} by approximately **{abs(mean_deviation):.1f}°**"
                    else:
                        action = f"**Bend** your {feature_name.lower()} deeper by approximately **{abs(mean_deviation):.1f}°**"
                
                elif "Neck" in feature_name:
                    if mean_deviation > 0:
                        action = f"**Lift** your head/neck by approximately **{abs(mean_deviation):.1f}°**"
                    else:
                        action = f"**Lower** your head/neck by approximately **{abs(mean_deviation):.1f}°**"
                
                elif "Shoulder" in feature_name:
                    if mean_deviation > 0:
                        action = f"**Raise** your {feature_name.lower()} by approximately **{abs(mean_deviation):.1f}°**"
                    else:
                        action = f"**Lower** your {feature_name.lower()} by approximately **{abs(mean_deviation):.1f}°**"
                
                else:
                    action = f"Adjust your {feature_name.lower()} by approximately **{abs(mean_deviation):.1f}°**"
                
                # Add consistency note if high variance
                consistency_note = ""
                if std_deviation > 10:
                    consistency_note = " (⚠️ Try to maintain consistency)"
                
                feedback_items.append(f"• {action}{consistency_note}")
        
        # Build feedback message
        if feedback_items:
            feedback_md = f"### 🎯 Correction Suggestions for **{predicted_pose.title()}** Pose:\n\n"
            feedback_md += "\n".join(feedback_items)
            feedback_md += "\n\n---\n💡 **Tip:** Make adjustments gradually and focus on maintaining proper form throughout the pose."
        else:
            feedback_md = f"### ✅ Great job!\n\nYour **{predicted_pose.title()}** pose looks good! No major corrections needed."
        
        return feedback_md
        
    except Exception as e:
        print(f"Error generating correction feedback: {e}")
        import traceback
        traceback.print_exc()
        return f"**Error generating correction feedback:** {str(e)}"



def classify_pose_from_video(video_file, use_all_frames=True):
    """
    classify pose from uploaded video file and create rendered video with landmarks and correction graphs.
    """
    try:
        print("Step 1: Extracting landmarks from video...")
        landmarks_df, landmark_mp_list = give_landmarks(video_file, "", fps=30)
        
        print("Step 2: Predicting pose...")
        predicted_pose = predict_from_dataframe(landmarks_df)
        print(f"Predicted pose: {predicted_pose}")

        correction_data = None
        correction_graph = None
        correction_feedback = "**No correction feedback available yet.**"  # Initialize

        # Try to get correction data
        try:
            print("Step 3: Getting correction data...")
            correction_data = predict_correction_from_dataframe(
                landmarks_df, predicted_pose
            )
            print(f"Correction data received: {correction_data is not None}")
            
            if correction_data is not None and correction_data.get("status") == "success":
                print("Step 4: Generating correction graph...")
                data_input = correction_data["input_data"]
                outputs = correction_data["corrected_data"]
                data_original = correction_data["reference_data"]
                
                correction_graph = generate_correction_graph(
                    data_input, outputs, data_original, predicted_pose
                )
                print("Correction graph generated successfully")
                
                # **NEW: Generate correction feedback**
                print("Step 4b: Generating correction feedback...")
                correction_feedback = generate_correction_feedback(
                    correction_data, predicted_pose
                )
                print("Correction feedback generated successfully")
            else:
                print("Correction data not available or failed")
                correction_feedback = "**Correction analysis not available for this pose.**"
                
        except Exception as e:
            print(f"Error during correction analysis: {e}")
            correction_data = None
            correction_feedback = f"**Error generating corrections:** {str(e)}"

        print("Step 5: Creating rendered video...")
        rendered_video_path = create_rendered_video_simple(video_file, landmark_mp_list)

        correction_video_path = None
        if correction_data is not None and correction_data.get("status") == "success":
            print("Step 6: Creating correction visualization video...")
            try:
                correction_video_path = create_correction_visualization_video(
                    video_file, landmark_mp_list, correction_data
                )
                print("Correction video created successfully")
            except Exception as e:
                print(f"Error creating correction video: {e}")
                correction_video_path = None

        video_info = f"**Frames processed:** {len(landmarks_df)}"

        # **MODIFIED: Include correction_feedback in return tuple**
        return (
            rendered_video_path,
            predicted_pose,
            video_info,
            correction_graph,
            correction_video_path,
            correction_feedback,  # <-- ADD THIS
        )

    except Exception as e:
        print(f"Error in classify_pose_from_video: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Error: {str(e)}", "", None, None, "**Error during analysis.**"


def create_rendered_video_simple(video_file, landmark_mp_list):
    """
    create a video with rendered pose landmarks (original rendering).

    Parameters
    ----------
    video_file : str
        path to original video file
    landmark_mp_list : list
        list of MediaPipe pose results

    Returns
    -------
    str
        path to rendered video file
    """
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    landmark_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if landmark_idx < len(landmark_mp_list):
            landmarks = landmark_mp_list[landmark_idx].pose_landmarks
            if landmarks is not None:
                try:
                    frame = render_landmarks(frame, landmarks)
                except Exception:
                    pass
            landmark_idx += 1

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    return output_path


def create_correction_visualization_video(
    video_file, landmark_mp_list, correction_data
):
    """
    create a video with rendered pose landmarks showing correction-based gradient colors.

    Parameters
    ----------
    video_file : str
        path to original video file
    landmark_mp_list : list
        list of MediaPipe pose results
    correction_data : dict
        dictionary containing correction data for gradient colors

    Returns
    -------
    str
        path to correction visualization video file
    """
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = tempfile.NamedTemporaryFile(
        suffix="_correction.mp4", delete=False
    ).name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    seq_bounds = calculate_sequence_bounds(landmark_mp_list)

    frame_count = 0
    landmark_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if landmark_idx < len(landmark_mp_list):
            landmarks = landmark_mp_list[landmark_idx].pose_landmarks
            if landmarks is not None:
                try:
                    frame = render_landmarks(
                        frame,
                        landmarks,
                        correction_data=correction_data,
                        frame_idx=landmark_idx,
                        seq_bounds=seq_bounds,
                    )
                except Exception:
                    frame = render_landmarks(frame, landmarks)
            landmark_idx += 1

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    return output_path


def load_sample_video():
    """
    load the sample video from assets folder.

    Returns
    -------
    str
        path to the sample video file
    """
    sample_video_path = "assets/sample_wrong.mp4"
    if os.path.exists(sample_video_path):
        return sample_video_path
    else:
        return None


def create_gradio_interface():
    """
    Create a modern Gradio dashboard for PosePilot with fixed correction logic.
    """
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        text_size="md",
        font=["Inter", "system-ui", "sans-serif"]
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
    backdrop-filter: blur(10px);
}

/* OVERRIDE GRADIO MARKDOWN WHITE TEXT */
.prose * {
    color: #1f2937 !important;
}
.prose strong, .prose span, .prose p, .prose b {
    color: #1f2937 !important;
}

.upload-section, .results-section {
    background: white;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

/* MOST AGGRESSIVE: Force all markdown and tab content to be black */
.markdown, .markdown *, .markdown * *, .tab-content * {
    color: #1f2937 !important;
}

/* Fix section headers */
.upload-section h3, .results-section h3 {
    color: #1f2937 !important;
    font-weight: 600;
}

/* Tab navigation styling */
.tab-nav button, .tab-nav button span {
    color: #1f2937 !important;
    font-weight: 500;
}
.tab-nav button[aria-selected="true"] {
    color: #667eea !important;
    font-weight: 600;
}
.tab-nav button:hover {
    color: #667eea !important;
}
.tabs > .tab-nav > button {
    color: #1f2937 !important;
}
.tabs button span, .tabs button {
    color: #1f2937 !important;
}
button[role="tab"], button[role="tab"] span {
    color: #1f2937 !important;
}

/* Button styling */
button {
    border-radius: 6px;
    font-weight: 500;
}

footer {
    display: none !important;
}
"""



    with gr.Blocks(title="", theme=theme, css=custom_css) as app:
        
        # --- State ---
        analysis_state = gr.State(
            {
                "rendered_video_path": None,
                "predicted_pose": None,
                "video_info": None,
                "correction_graph": None,
                "correction_video_path": None,
                "correction_feedback": None,
            }
        )

        # --- Main layout with container ---
        with gr.Column(elem_classes="main-content"):
            
            with gr.Row():
                # ---------------- LEFT COLUMN ----------------
                with gr.Column(scale=1, elem_classes="upload-section"):
                    gr.Markdown("### 📤 Upload Video")
                    video_input = gr.Video(label="Upload Your Pose Video", height=200)

                    # REMOVED: use_all_frames checkbox
                    # REMOVED: sample_btn button

                    analyze_btn = gr.Button("🚀 Analyze Video", variant="primary", size="lg")
                    save_btn = gr.Button("💾 Save Results", variant="secondary", size="lg")
                    save_status = gr.Markdown(label="Save Status")

                # ---------------- RIGHT COLUMN ----------------
                with gr.Column(scale=2, elem_classes="results-section"):
                    gr.Markdown("### 📊 Analysis Results")

                    with gr.Row():
                        predicted_pose = gr.Textbox(
                            label="Predicted Pose",
                            placeholder="Pose name will appear here...",
                            interactive=False
                        )
                        video_info = gr.Markdown(label="Video Information")

                    with gr.Tab("💡 Correction Feedback"):
                        correction_feedback = gr.Markdown(
                            value="Upload and analyze a video to see correction suggestions...",
                            label="Personalized Corrections"
                        )

                    with gr.Tab("🎬 Rendered Keypoints Video"):
                        rendered_video = gr.Video(interactive=False, height=300)

                    with gr.Tab("🎨 Correction Visualization Video"):
                        correction_video = gr.Video(interactive=False, height=300)

                    with gr.Tab("📈 Correction Graph"):
                        correction_graph = gr.Plot()

        # --- Define button actions ---
        def analyze_video(video_file, state):
            """Analyze video with use_all_frames always set to True"""
            if video_file is None:
                return None, "", "", None, None, "Upload a video first.", state

            # Always use all frames (hardcoded)
            result = classify_pose_from_video(video_file, use_all_frames=True)

            new_state = {
                "rendered_video_path": result[0],
                "predicted_pose": result[1],
                "video_info": result[2],
                "correction_graph": result[3],
                "correction_video_path": result[4],
                "correction_feedback": result[5],
            }

            return result[0], result[1], result[2], result[3], result[4], result[5], new_state

        def save_all_results(state):
            """Save all analysis results from stored state."""
            try:
                import shutil
                from datetime import datetime

                if (
                    not state
                    or not state.get("rendered_video_path")
                    or not state.get("predicted_pose")
                ):
                    return "❌ No results to save. Please analyze a video first."

                rendered_video_path = state["rendered_video_path"]
                predicted_pose = state["predicted_pose"]
                correction_graph = state["correction_graph"]
                correction_video_path = state["correction_video_path"]

                results_dir = "infer_results"
                os.makedirs(results_dir, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pose_dir = os.path.join(results_dir, f"{predicted_pose}_{timestamp}")
                os.makedirs(pose_dir, exist_ok=True)

                saved_files = []

                if correction_graph is not None:
                    fig_path = os.path.join(pose_dir, "correction_analysis.png")
                    correction_graph.savefig(fig_path, dpi=150, bbox_inches="tight")
                    saved_files.append(f"📊 Correction graph: {os.path.basename(fig_path)}")

                if rendered_video_path and os.path.exists(rendered_video_path):
                    video_path = os.path.join(pose_dir, "keypoint_video.mp4")
                    shutil.copy2(rendered_video_path, video_path)
                    saved_files.append(f"🎬 Rendered video: {os.path.basename(video_path)}")

                if correction_video_path and os.path.exists(correction_video_path):
                    correction_video_save_path = os.path.join(
                        pose_dir, "correction_video.mp4"
                    )
                    shutil.copy2(correction_video_path, correction_video_save_path)
                    saved_files.append(f"🎨 Correction video: {os.path.basename(correction_video_save_path)}")

                if saved_files:
                    return (
                        f"✅ **Results saved to:** `{pose_dir}`\n\n**Saved files:**\n"
                        + "\n".join([f"- {file}" for file in saved_files])
                    )
                else:
                    return "❌ No files were saved. Check if analysis was completed successfully."

            except Exception as e:
                return f"❌ Error saving results: {str(e)}"

        # MODIFIED: Removed use_all_frames from inputs
        analyze_btn.click(
            fn=analyze_video,
            inputs=[video_input, analysis_state],
            outputs=[
                rendered_video,
                predicted_pose,
                video_info,
                correction_graph,
                correction_video,
                correction_feedback,
                analysis_state,
            ],
        )

        # REMOVED: sample_btn.click handler

        save_btn.click(fn=save_all_results, inputs=[analysis_state], outputs=[save_status])

    return app



def main():
    """main function to launch the Gradio app."""
    app = create_gradio_interface()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()