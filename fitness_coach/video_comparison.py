"""
Generate side-by-side comparison videos of user vs reference 3D poses.
Uses the same visualization as the original pose3D images.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from pathlib import Path
import argparse
import sys
import os

# Import the original show3Dpose function from demo/vis.py
# Add demo directory to path
project_root = Path(__file__).parent.parent
demo_path = str(project_root / 'demo')
if demo_path not in sys.path:
    sys.path.insert(0, demo_path)

from vis import show3Dpose


def load_3d_poses(pose_file):
    """Load 3D poses from npz file."""
    data = np.load(pose_file, allow_pickle=True)
    if 'reconstruction' in data:
        poses = data['reconstruction']
    elif 'poses_3d' in data:
        poses = data['poses_3d']
    else:
        # Try to get the first array
        poses = data[list(data.keys())[0]]
    
    return poses


def plot_pose_3d(ax, pose, title):
    """Plot a single 3D pose using the original show3Dpose function."""
    ax.clear()
    
    # Use the original show3Dpose function (same as pose3D images)
    show3Dpose(pose, ax)
    
    # Add title
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)


def create_comparison_video(user_poses, reference_poses, output_path, 
                            user_video_name="User", reference_name="Reference",
                            fps=30, elev=15, azim=70):
    """
    Create a side-by-side comparison video.
    
    Args:
        user_poses: User 3D poses (N_frames, 17, 3)
        reference_poses: Reference 3D poses (N_frames, 17, 3)
        output_path: Path to save output video
        user_video_name: Display name for user
        reference_name: Display name for reference
        fps: Frames per second for output video
        elev: Elevation angle for 3D view
        azim: Azimuth angle for 3D view
    """
    print(f"\nCreating comparison video...")
    print(f"  User frames: {len(user_poses)}")
    print(f"  Reference frames: {len(reference_poses)}")
    
    # Ensure same number of frames (use minimum)
    n_frames = min(len(user_poses), len(reference_poses))
    user_poses = user_poses[:n_frames]
    reference_poses = reference_poses[:n_frames]
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Add main title
    fig.suptitle('Exercise Form Comparison', fontsize=16, fontweight='bold')
    
    def update(frame):
        """Update function for animation."""
        plot_pose_3d(ax1, reference_poses[frame], 
                    f'{reference_name}\nFrame {frame+1}/{n_frames}')
        plot_pose_3d(ax2, user_poses[frame], 
                    f'{user_video_name}\nFrame {frame+1}/{n_frames}')
        
        if frame % 30 == 0:
            print(f"  Progress: {frame}/{n_frames} frames ({100*frame//n_frames}%)")
        
        return ax1, ax2
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=n_frames, 
                        interval=1000/fps, blit=False)
    
    # Save video - try MP4 first, fall back to GIF if FFmpeg not available
    print(f"  Saving video to: {output_path}")
    
    # Try MP4 first (requires FFmpeg)
    try:
        writer = FFMpegWriter(fps=fps, bitrate=5000, codec='libx264')
        anim.save(str(output_path), writer=writer, dpi=100)
        print(f"✓ Video saved successfully!")
        print(f"  Output: {output_path}")
        print(f"  Duration: {n_frames/fps:.2f} seconds")
        print(f"  Format: MP4")
    except (FileNotFoundError, OSError) as e:
        # FFmpeg not found, try GIF instead
        print(f"  ⚠ FFmpeg not found, saving as GIF instead...")
        gif_path = str(output_path).replace('.mp4', '.gif')
        
        try:
            # Use Pillow writer for GIF (built into matplotlib)
            from matplotlib.animation import PillowWriter
            writer = PillowWriter(fps=fps)
            anim.save(gif_path, writer=writer, dpi=100)
            print(f"✓ GIF saved successfully!")
            print(f"  Output: {gif_path}")
            print(f"  Duration: {n_frames/fps:.2f} seconds")
            print(f"  Format: GIF")
            print(f"\n  Note: For MP4 format, install FFmpeg:")
            print(f"    conda install -c conda-forge ffmpeg")
            print(f"    or: winget install ffmpeg")
        except Exception as gif_error:
            print(f"✗ Error saving GIF: {gif_error}")
            print(f"\nOriginal MP4 error: {e}")
            print("\nTo enable MP4 output, install FFmpeg:")
            print("  conda install -c conda-forge ffmpeg")
            print("  or: winget install ffmpeg")
            raise
    except Exception as e:
        print(f"✗ Error saving video: {e}")
        raise
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Generate side-by-side comparison video of 3D poses'
    )
    parser.add_argument('--user-poses', required=True,
                       help='Path to user 3D poses npz file')
    parser.add_argument('--reference-poses', required=True,
                       help='Path to reference 3D poses npz file')
    parser.add_argument('--output', default='comparison_output.mp4',
                       help='Output video path')
    parser.add_argument('--user-name', default='Your Form',
                       help='Display name for user')
    parser.add_argument('--reference-name', default='Correct Form',
                       help='Display name for reference')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second')
    parser.add_argument('--elev', type=float, default=15,
                       help='Elevation angle for 3D view (degrees)')
    parser.add_argument('--azim', type=float, default=70,
                       help='Azimuth angle for 3D view (degrees)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("3D POSE COMPARISON VIDEO GENERATOR")
    print("="*60)
    
    # Load poses
    print(f"\nLoading user poses from: {args.user_poses}")
    user_poses = load_3d_poses(args.user_poses)
    print(f"  Loaded {len(user_poses)} frames")
    
    print(f"\nLoading reference poses from: {args.reference_poses}")
    reference_poses = load_3d_poses(args.reference_poses)
    print(f"  Loaded {len(reference_poses)} frames")
    
    # Create video
    create_comparison_video(
        user_poses=user_poses,
        reference_poses=reference_poses,
        output_path=args.output,
        user_video_name=args.user_name,
        reference_name=args.reference_name,
        fps=args.fps,
        elev=args.elev,
        azim=args.azim
    )
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == '__main__':
    main()

