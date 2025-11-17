"""
Reference Video Processor
Processes reference videos once and saves noisy samples for scoring
"""

import os
import sys
import numpy as np
import json
from pathlib import Path

# Add parent directory and demo directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'demo'))

from fitness_coach.noise_scoring import create_noisy_samples, calculate_statistical_bounds
from fitness_coach.body_parts import calculate_body_scale, get_joints_for_exercise


def process_reference_video(video_path, exercise_type='pushup', output_dir=None, n_samples=100):
    """
    Process a reference video and generate noisy samples for scoring
    
    Args:
        video_path: Path to reference video file
        exercise_type: Type of exercise (e.g., 'pushup', 'squat')
        output_dir: Directory to save processed data (default: references/{exercise_type}/)
        n_samples: Number of noisy samples to generate
    
    Returns:
        Dictionary with paths to saved files and metadata
    """
    # Change to project root for imports to work correctly
    original_cwd = os.getcwd()
    os.chdir(project_root)
    
    try:
        # Import after changing directory
        from demo.vis import get_pose2D, get_pose3D
    finally:
        os.chdir(original_cwd)
    
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Set up output directory
    if output_dir is None:
        output_dir = Path('references') / exercise_type
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing reference video: {video_path.name}")
    print(f"Exercise type: {exercise_type}")
    print(f"Output directory: {output_dir}")
    
    # Create temporary output directory for processing
    temp_output = output_dir / 'temp_processing'
    temp_output.mkdir(exist_ok=True)
    
    # Format output directory string (get_pose3D expects trailing slash)
    # Use absolute path to avoid issues when changing directories
    temp_output_abs = temp_output.resolve()
    output_dir_str = str(temp_output_abs).replace('\\', '/')
    if not output_dir_str.endswith('/'):
        output_dir_str += '/'
    
    video_path_abs = video_path.resolve()
    
    # Change to project root for processing
    os.chdir(project_root)
    
    # Save original argv and temporarily clear it to avoid argparse conflicts
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]]  # Keep only script name
    
    try:
        # Step 1: Extract 2D poses
        print("\n[1/4] Extracting 2D poses...")
        try:
            # get_pose2D expects output_dir with trailing slash
            # It adds 'input_2D/' to it (line 95 in vis.py)
            get_pose2D(str(video_path_abs), output_dir_str)
        except Exception as e:
            print(f"Error in 2D pose extraction: {e}")
            raise
        
        # Step 2: Extract 3D poses
        print("\n[2/4] Extracting 3D poses...")
        try:
            # get_pose3D also expects output_dir with trailing slash
            # It looks for output_dir + 'input_2D/keypoints.npz' (line 190 in vis.py)
            get_pose3D(str(video_path_abs), output_dir_str)
        except Exception as e:
            print(f"Error in 3D pose extraction: {e}")
            raise
    finally:
        sys.argv = original_argv  # Restore original argv
        os.chdir(original_cwd)
    
    # Step 3: Load 3D poses
    # get_pose3D saves to output_dir + 'keypoints_3D.npz' (line 279 in vis.py)
    keypoints_3d_path = temp_output_abs / 'keypoints_3D.npz'
    
    if not keypoints_3d_path.exists():
        # Try alternative locations in case path handling differs
        alt_paths = [
            temp_output_abs / 'keypoints_3D.npz',
            temp_output_abs.parent / 'keypoints_3D.npz',
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                keypoints_3d_path = alt_path
                break
        else:
            # List what files actually exist to help debug
            print(f"\nDebug: Looking for keypoints_3D.npz")
            print(f"Expected location: {keypoints_3d_path}")
            print(f"Files in temp_processing:")
            if temp_output_abs.exists():
                for item in temp_output_abs.rglob('*'):
                    if item.is_file():
                        print(f"  {item}")
            raise FileNotFoundError(f"3D keypoints not found: {keypoints_3d_path}")
    
    keypoints_3d = np.load(str(keypoints_3d_path), allow_pickle=True)['reconstruction']
    print(f"Loaded {len(keypoints_3d)} frames of 3D poses")
    
    # Convert to numpy array if needed
    if isinstance(keypoints_3d, list):
        keypoints_3d = np.array(keypoints_3d)
    
    # Step 4: Generate noisy samples
    print(f"\n[3/4] Generating {n_samples} noisy samples...")
    noisy_samples = create_noisy_samples(keypoints_3d, n_samples=n_samples, per_joint_noise=True)
    print(f"Generated noisy samples shape: {noisy_samples.shape}")
    
    # Step 5: Calculate metadata
    print("\n[4/4] Calculating metadata...")
    body_scale = calculate_body_scale(keypoints_3d)
    relevant_body_parts = get_joints_for_exercise(exercise_type)
    
    # Calculate statistical bounds
    mean_poses, lower_bound, upper_bound, tolerance = calculate_statistical_bounds(
        keypoints_3d, noise_std=0.05
    )
    
    metadata = {
        'exercise_type': exercise_type,
        'video_path': str(video_path),
        'video_name': video_path.stem,
        'num_frames': len(keypoints_3d),
        'body_scale': float(body_scale),
        'relevant_body_parts': relevant_body_parts,
        'n_samples': n_samples,
        'timestamp': str(Path(video_path).stat().st_mtime) if video_path.exists() else None
    }
    
    # Step 6: Save everything
    print("\nSaving processed data...")
    
    # Save 3D poses
    poses_3d_path = output_dir / 'keypoints_3D.npz'
    np.savez_compressed(str(poses_3d_path), reconstruction=keypoints_3d)
    print(f"  Saved 3D poses: {poses_3d_path}")
    
    # Save noisy samples
    noisy_samples_path = output_dir / 'noisy_samples.npz'
    np.savez_compressed(str(noisy_samples_path), samples=noisy_samples)
    print(f"  Saved noisy samples: {noisy_samples_path}")
    
    # Save statistical bounds
    bounds_path = output_dir / 'statistical_bounds.npz'
    np.savez_compressed(
        str(bounds_path),
        mean=mean_poses,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        tolerance=tolerance
    )
    print(f"  Saved statistical bounds: {bounds_path}")
    
    # Save metadata
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata: {metadata_path}")
    
    # Clean up temporary files (optional - keep 2D poses for debugging)
    # import shutil
    # shutil.rmtree(temp_output, ignore_errors=True)
    
    print(f"\n✓ Reference video processed successfully!")
    print(f"  Output directory: {output_dir}")
    
    return {
        'output_dir': str(output_dir),
        'poses_3d_path': str(poses_3d_path),
        'noisy_samples_path': str(noisy_samples_path),
        'bounds_path': str(bounds_path),
        'metadata_path': str(metadata_path),
        'metadata': metadata
    }


def load_reference(exercise_type, references_dir='references'):
    """
    Load a processed reference
    
    Args:
        exercise_type: Type of exercise (e.g., 'pushup')
        references_dir: Directory containing references
    
    Returns:
        Dictionary with loaded data
    """
    ref_dir = Path(references_dir) / exercise_type
    
    if not ref_dir.exists():
        raise FileNotFoundError(f"Reference not found: {ref_dir}")
    
    # Load metadata
    metadata_path = ref_dir / 'metadata.json'
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load 3D poses
    poses_3d_path = ref_dir / 'keypoints_3D.npz'
    if not poses_3d_path.exists():
        raise FileNotFoundError(f"3D poses not found: {poses_3d_path}")
    
    poses_3d = np.load(str(poses_3d_path), allow_pickle=True)['reconstruction']
    if isinstance(poses_3d, list):
        poses_3d = np.array(poses_3d)
    
    # Load noisy samples
    noisy_samples_path = ref_dir / 'noisy_samples.npz'
    noisy_samples = None
    if noisy_samples_path.exists():
        noisy_samples = np.load(str(noisy_samples_path), allow_pickle=True)['samples']
    
    # Load statistical bounds
    bounds_path = ref_dir / 'statistical_bounds.npz'
    bounds = None
    if bounds_path.exists():
        bounds_data = np.load(str(bounds_path), allow_pickle=True)
        bounds = {
            'mean': bounds_data['mean'],
            'lower_bound': bounds_data['lower_bound'],
            'upper_bound': bounds_data['upper_bound'],
            'tolerance': bounds_data['tolerance']
        }
    
    return {
        'poses_3d': poses_3d,
        'noisy_samples': noisy_samples,
        'bounds': bounds,
        'metadata': metadata,
        'ref_dir': str(ref_dir)
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process reference video for scoring')
    parser.add_argument('--video', type=str, required=True, help='Path to reference video')
    parser.add_argument('--exercise', type=str, default='pushup', help='Exercise type')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--samples', type=int, default=100, help='Number of noisy samples')
    
    args = parser.parse_args()
    
    try:
        result = process_reference_video(
            args.video,
            exercise_type=args.exercise,
            output_dir=args.output,
            n_samples=args.samples
        )
        print("\n" + "="*50)
        print("SUCCESS!")
        print("="*50)
        print(f"Reference saved to: {result['output_dir']}")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

