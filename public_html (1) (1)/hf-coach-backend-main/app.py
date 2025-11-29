import gradio as gr
import numpy as np
import json
import sys
import os
import shutil
from pathlib import Path

# --- CRITICAL FIX: Ensure Parent Directory is on the Path ---

def initialize_paths():
    """
    Ensures the Python environment can find the custom 'fitness_coach' package 
    and the 'demo' folder containing vis.py.
    """
    current_dir = Path(__file__).parent.resolve()
    
    # 1. Add current directory (root) to path to find 'fitness_coach/'
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    # 2. Add 'demo' directory to path to find 'vis' module (required by processors)
    demo_path = current_dir / 'demo'
    if demo_path.is_dir() and str(demo_path) not in sys.path:
        sys.path.insert(0, str(demo_path))
    
# Execute path setup immediately
initialize_paths()

# --- ATTEMPT CRITICAL IMPORT ---
score_exercise = None
try:
    # This import must succeed now that the paths are set
    from fitness_coach.comparison import score_exercise
    print("✓ Successfully loaded fitness_coach.comparison module.")
except ImportError as e:
    # If this fails, the file structure in the Space is wrong.
    print(f"CRITICAL IMPORT FAILURE: Could not import fitness_coach.comparison: {e}")
    score_exercise = None

# --- Configuration ---
REFERENCES_DIR = 'references'
TEMP_DIR = 'temp_uploads'
os.makedirs(REFERENCES_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)


# --- Core Processing Function ---

def run_scoring_pipeline(user_video_file, ref_video_file, exercise_type):
    """
    Runs the full PoseFormer-based scoring pipeline, processing uploaded files.
    """
    
    if score_exercise is None:
        return json.dumps({
            "status": "error",
            "error_message": "Scoring pipeline failed: CRITICAL PYTHON MODULES NOT FOUND. Ensure 'fitness_coach/' folder is uploaded correctly."
        })
    
    if not user_video_file or not ref_video_file:
        return json.dumps({
            "status": "error",
            "error_message": "Both user and reference videos must be uploaded."
        })
    
    user_video_path = user_video_file.name
    ref_video_path = ref_video_file.name
    
    try:
        # --- Run Scoring ---
        results = score_exercise(
            user_video_path=str(user_video_path),
            reference_id=exercise_type,
            references_dir=REFERENCES_DIR,
            force_reprocess=True, 
            scoring_method='statistical'
        )
        
        # --- Format JSON Output (Exact format for Hostinger Frontend) ---
        api_response = {
            "status": "success",
            "exercise": {
                "type": results['exercise_type'],
                "reference": Path(ref_video_path).name,
                "user_video": Path(user_video_path).name
            },
            "scores": {
                "overall": float(round(results['overall_score'], 2)),
                "relevant": float(round(results['relevant_score'], 2)),
                "body_parts": {
                    part: float(round(score, 2)) 
                    for part, score in results['relevant_body_part_scores'].items()
                }
            },
            "metrics": {
                "frames": {
                    "user": int(results['num_frames_user']),
                    "reference": int(results['num_frames_ref']),
                    "aligned": int(results['num_frames_aligned'])
                },
                "alignment_quality": float(round(results['details'].get('alignment_score', 0), 4)) if results['details'].get('alignment_score') is not None else None,
                "body_part_details": {
                    part: {
                        "position_error_avg": float(round(metrics.get('position_error', 0), 4)),
                        "position_error_max": float(round(metrics.get('max_position_error', 0), 4)),
                        "tolerance_threshold": float(round(metrics.get('tolerance_threshold', 0), 4)),
                        "in_tolerance_percentage": float(round(metrics.get('in_tolerance_percentage', 0), 1))
                    }
                    for part, metrics in results['details'].get('body_part_details', {}).items()
                    if part in results['relevant_body_part_scores']
                }
            }
        }
        
        return json.dumps(api_response)

    except Exception as e:
        print(f"CRITICAL BACKEND SCORING ERROR: {e}")
        # The ultimate runtime error from within the comparison/processor logic
        return json.dumps({
            "status": "error",
            "error_message": f"Scoring pipeline failed: {str(e)}. This may be due to a missing file (e.g., demo/vis.py or necessary model weights) or a dependency not installing correctly."
        })


# --- Gradio Interface Setup ---

user_video_input = gr.File(label="Upload User Video (Your Attempt)")
ref_video_input = gr.File(label="Upload Reference Video (Ground Truth)")
exercise_type_input = gr.Dropdown(
    label="Select Exercise Type",
    choices=['pushup', 'squat', 'plank', 'lunge', 'all'],
    value='pushup'
)

output_json = gr.JSON(label="JSON Output for CoachAI Frontend")

iface = gr.Interface(
    fn=run_scoring_pipeline,
    inputs=[user_video_input, ref_video_input, exercise_type_input],
    outputs=output_json,
    title="PoseFormer Fitness Scoring API",
    description="Upload a user video and a reference video. The backend runs the PoseFormer-based comparison and returns the final JSON score for the Gemini Coach.",
    allow_flagging='never',
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
