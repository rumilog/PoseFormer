"""
User Video Processor
Processes user videos and extracts 3D poses for scoring
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add parent directory and demo directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'demo'))


def process_user_video(video_path, output_dir=None, cleanup=True):
    """
    Process a user video and extract 3D poses
    
    Args:
        video_path: Path to user video file
        output_dir: Directory to save processed data (default: temp_user_processing/)
        cleanup: If True, remove intermediate files after processing
    
    Returns:
        Dictionary with paths and 3D poses
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
    
    # Set up output directory with caching
    if output_dir is None:
        output_dir = Path('user_videos_cache') / video_path.stem
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already processed (cache hit)
    keypoints_3d_path = output_dir / 'keypoints_3D.npz'
    if keypoints_3d_path.exists():
        print(f"✓ Using cached processing for: {video_path.name}")
        print(f"  Cache location: {output_dir}")
        keypoints_3d = np.load(str(keypoints_3d_path), allow_pickle=True)['reconstruction']
        print(f"  Loaded {len(keypoints_3d)} frames from cache\n")
        
        return {
            'keypoints_3d': keypoints_3d,
            'poses_3d': keypoints_3d,  # Alias for compatibility
            'video_path': video_path,
            'output_dir': output_dir,
            'num_frames': len(keypoints_3d)
        }
    
    print(f"Processing user video: {video_path.name}")
    print(f"Output directory: {output_dir}")
    
    # Format output directory string (both functions expect trailing slash)
    # Use absolute path to avoid issues when changing directories
    output_dir_abs = output_dir.resolve()
    output_dir_str = str(output_dir_abs).replace('\\', '/')
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
        print("\n[1/2] Extracting 2D poses...")
        try:
            # get_pose2D adds 'input_2D/' to output_dir (line 95 in vis.py)
            get_pose2D(str(video_path_abs), output_dir_str)
        except Exception as e:
            print(f"Error in 2D pose extraction: {e}")
            raise
        
        # Step 2: Extract 3D poses
        print("\n[2/2] Extracting 3D poses...")
        try:
            # get_pose3D looks for output_dir + 'input_2D/keypoints.npz' (line 190 in vis.py)
            get_pose3D(str(video_path_abs), output_dir_str)
        except Exception as e:
            print(f"Error in 3D pose extraction: {e}")
            raise
    finally:
        sys.argv = original_argv  # Restore original argv
        os.chdir(original_cwd)
    
    # Step 3: Load 3D poses
    # get_pose3D saves to output_dir + 'keypoints_3D.npz' (line 279 in vis.py)
    keypoints_3d_path = output_dir_abs / 'keypoints_3D.npz'
    if not keypoints_3d_path.exists():
        raise FileNotFoundError(f"3D keypoints not found: {keypoints_3d_path}")
    
    keypoints_3d = np.load(str(keypoints_3d_path), allow_pickle=True)['reconstruction']
    print(f"Loaded {len(keypoints_3d)} frames of 3D poses")
    
    # Convert to numpy array if needed
    if isinstance(keypoints_3d, list):
        keypoints_3d = np.array(keypoints_3d)
    
    result = {
        'poses_3d': keypoints_3d,
        'output_dir': str(output_dir),
        'keypoints_3d_path': str(keypoints_3d_path),
        'num_frames': len(keypoints_3d)
    }
    
    # Cleanup intermediate files if requested
    if cleanup:
        # Keep only the 3D keypoints
        import shutil
        for item in output_dir.iterdir():
            if item.is_dir() and item.name != 'input_2D':  # Keep input_2D for debugging
                shutil.rmtree(item, ignore_errors=True)
            elif item.is_file() and item.name != 'keypoints_3D.npz':
                item.unlink(missing_ok=True)
    
    print(f"\n✓ User video processed successfully!")
    print(f"  Frames: {len(keypoints_3d)}")
    print(f"  Output: {output_dir}")
    
    return result


def load_user_poses(keypoints_path):
    """
    Load user poses from a saved file
    
    Args:
        keypoints_path: Path to keypoints_3D.npz file
    
    Returns:
        poses_3d: Array of shape [frames, 17, 3]
    """
    keypoints_path = Path(keypoints_path)
    if not keypoints_path.exists():
        raise FileNotFoundError(f"Keypoints file not found: {keypoints_path}")
    
    data = np.load(str(keypoints_path), allow_pickle=True)
    poses_3d = data['reconstruction']
    
    if isinstance(poses_3d, list):
        poses_3d = np.array(poses_3d)
    
    return poses_3d


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process user video for scoring')
    parser.add_argument('--video', type=str, required=True, help='Path to user video')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--keep-files', action='store_true', help='Keep intermediate files')
    
    args = parser.parse_args()
    
    try:
        result = process_user_video(
            args.video,
            output_dir=args.output,
            cleanup=not args.keep_files
        )
        print("\n" + "="*50)
        print("SUCCESS!")
        print("="*50)
        print(f"3D poses extracted: {result['num_frames']} frames")
        print(f"Saved to: {result['keypoints_3d_path']}")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

