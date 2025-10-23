"""
gradio web interface for pose classification and correction analysis.
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


def classify_pose_from_video(video_file, use_all_frames=True):
    """
    classify pose from uploaded video file and create rendered video with landmarks and correction graphs.

    Parameters
    ----------
    video_file : str
        path to uploaded video file
    use_all_frames : bool
        whether to use all frames or 25 selected frames for correction analysis

    Returns
    -------
    tuple
        tuple containing (rendered_video_path, predicted_pose, video_info, correction_graph, comparison_video_path)
    """
    try:
        landmarks_df, landmark_mp_list = give_landmarks(video_file, "", fps=30)
        predicted_pose = predict_from_dataframe(landmarks_df)

        correction_data = None
        correction_graph = None

        try:
            correction_data = predict_correction_from_dataframe(
                landmarks_df, predicted_pose, use_all_frames
            )

            if correction_data is not None:
                correction_graph = generate_correction_graph_for_pose(
                    landmarks_df, predicted_pose
                )

        except Exception:
            correction_data = None

        rendered_video_path = create_rendered_video_simple(video_file, landmark_mp_list)

        correction_video_path = None
        if correction_data is not None:
            correction_video_path = create_correction_visualization_video(
                video_file, landmark_mp_list, correction_data
            )

        video_info = f"**Frames processed:** {len(landmarks_df)}"

        return (
            rendered_video_path,
            predicted_pose,
            video_info,
            correction_graph,
            correction_video_path,
        )

    except Exception as e:
        return None, f"Error: {str(e)}", "", None, None


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


def generate_correction_graph_for_pose(landmarks_df, predicted_pose):
    """
    generate correction graph for the predicted pose.

    Parameters
    ----------
    landmarks_df : pandas.DataFrame
        dataframe with pose landmarks
    predicted_pose : str
        predicted pose name

    Returns
    -------
    matplotlib.figure.Figure or None
        correction graph figure or None if not available
    """
    try:
        import os

        correction_model_path = f"models/{predicted_pose}_correction_model.pth"
        if not os.path.exists(correction_model_path):
            return None

        correction_data = predict_correction_from_dataframe(
            landmarks_df, predicted_pose
        )

        if correction_data is None:
            return None

        data_input = correction_data["input_data"]
        outputs = correction_data["corrected_data"]
        data_original = correction_data["reference_data"]

        fig = generate_correction_graph(
            data_input, outputs, data_original, predicted_pose
        )

        return fig

    except Exception:
        return None


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
    Create a modern Gradio dashboard for PosePilot:
    Left = Upload & Controls, Right = Results & Visualization.
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
    
    .upload-section, .results-section {
        background: white;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Fix text visibility */
    h3, h2, label, .markdown, p {
        color: #1f2937 !important;
    }
    
    /* Fix section headers */
    .upload-section h3, .results-section h3 {
        color: #1f2937 !important;
        font-weight: 600;
    }
    
    footer {
        display: none !important;
    }
    """

    with gr.Blocks(title="PosePilot - Pose Classification", theme=theme, css=custom_css) as app:
        
        # --- State ---
        analysis_state = gr.State(
            {
                "rendered_video_path": None,
                "predicted_pose": None,
                "video_info": None,
                "correction_graph": None,
                "correction_video_path": None,
            }
        )

        # --- Main layout with container ---
        with gr.Column(elem_classes="main-content"):
            with gr.Row():
                # ---------------- LEFT COLUMN ----------------
                with gr.Column(scale=1, elem_classes="upload-section"):
                    gr.Markdown("### 📤 Upload & Settings")
                    video_input = gr.Video(label="Upload Your Pose Video", height=200)

                    use_all_frames = gr.Checkbox(
                        label="Use All Frames",
                        value=True,
                        info="Use all frames for correction analysis (unchecked = 25 sampled frames)"
                    )

                    with gr.Row():
                        analyze_btn = gr.Button("🚀 Analyze Video", variant="primary", size="lg")
                        sample_btn = gr.Button("🎥 Load Sample", variant="secondary")

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

                    with gr.Tab("🎬 Rendered Keypoints Video"):
                        rendered_video = gr.Video(interactive=False, height=240)

                    with gr.Tab("🎨 Correction Visualization Video"):
                        correction_video = gr.Video(interactive=False, height=240)

                    with gr.Tab("📈 Correction Graph"):
                        correction_graph = gr.Plot()

        # --- Define button actions ---
        def analyze_video(video_file, use_all_frames, state):
            if video_file is None:
                return None, "", "", None, None, state

            result = classify_pose_from_video(video_file, use_all_frames)

            new_state = {
                "rendered_video_path": result[0],
                "predicted_pose": result[1],
                "video_info": result[2],
                "correction_graph": result[3],
                "correction_video_path": result[4],
            }

            return result[0], result[1], result[2], result[3], result[4], new_state

        def save_all_results(state):
            try:
                import os, shutil
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
                    saved_files.append(f"📊 {os.path.basename(fig_path)}")

                if rendered_video_path and os.path.exists(rendered_video_path):
                    video_path = os.path.join(pose_dir, "keypoint_video.mp4")
                    shutil.copy2(rendered_video_path, video_path)
                    saved_files.append(f"🎬 {os.path.basename(video_path)}")

                if correction_video_path and os.path.exists(correction_video_path):
                    correction_video_save_path = os.path.join(
                        pose_dir, "correction_video.mp4"
                    )
                    shutil.copy2(correction_video_path, correction_video_save_path)
                    saved_files.append(f"🎨 {os.path.basename(correction_video_save_path)}")

                if saved_files:
                    return f"✅ Results saved to `{pose_dir}`:\n" + "\n".join(saved_files)
                else:
                    return "❌ No files were saved. Please re-run analysis."

            except Exception as e:
                return f"❌ Error saving results: {str(e)}"

        analyze_btn.click(
            fn=analyze_video,
            inputs=[video_input, use_all_frames, analysis_state],
            outputs=[
                rendered_video,
                predicted_pose,
                video_info,
                correction_graph,
                correction_video,
                analysis_state,
            ],
        )

        sample_btn.click(fn=load_sample_video, outputs=[video_input])
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