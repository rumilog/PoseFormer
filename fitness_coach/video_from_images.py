"""
Generate side-by-side comparison videos from existing pose3D images.
Much simpler - just combines the existing PNG images!
"""
import numpy as np
from PIL import Image
import glob
from pathlib import Path
import argparse
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def load_image_sequence(image_dir):
    """Load all PNG images from a directory, sorted by filename."""
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    # Find all PNG files matching the pattern (e.g., 0000_3D.png, 0001_3D.png)
    image_files = sorted(glob.glob(str(image_dir / '*_3D.png')))
    
    if not image_files:
        raise FileNotFoundError(f"No pose3D images found in {image_dir}")
    
    print(f"  Found {len(image_files)} images in {image_dir}")
    return image_files


def create_comparison_video_from_images(user_image_dir, reference_image_dir, output_path,
                                       user_video_name="Your Form", reference_name="Correct Form",
                                       fps=30):
    """
    Create side-by-side video from existing pose3D images.
    
    Args:
        user_image_dir: Directory containing user pose3D images
        reference_image_dir: Directory containing reference pose3D images
        output_path: Path to save output video
        user_video_name: Display name for user
        reference_name: Display name for reference
        fps: Frames per second
    """
    print(f"\nCreating comparison video from existing images...")
    
    # Load image sequences
    print(f"\nLoading user images from: {user_image_dir}")
    user_images = load_image_sequence(user_image_dir)
    
    print(f"\nLoading reference images from: {reference_image_dir}")
    reference_images = load_image_sequence(reference_image_dir)
    
    # Use minimum length to ensure both sequences are the same
    n_frames = min(len(user_images), len(reference_images))
    user_images = user_images[:n_frames]
    reference_images = reference_images[:n_frames]
    
    print(f"\n  Using {n_frames} frames for comparison")
    
    # Load first images to get dimensions
    user_img = Image.open(user_images[0])
    ref_img = Image.open(reference_images[0])
    
    # Get dimensions (assuming they're similar)
    img_height = max(user_img.height, ref_img.height)
    img_width = max(user_img.width, ref_img.width)
    
    # Create figure for side-by-side display
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    ax1.axis('off')
    ax2.axis('off')
    
    # Add titles
    fig.suptitle('Exercise Form Comparison', fontsize=16, fontweight='bold')
    ax1.set_title(f'{reference_name}', fontsize=14, fontweight='bold', pad=10)
    ax2.set_title(f'{user_video_name}', fontsize=14, fontweight='bold', pad=10)
    
    def update(frame):
        """Update function for animation."""
        # Load images
        ref_img = Image.open(reference_images[frame])
        user_img = Image.open(user_images[frame])
        
        # Display images
        ax1.clear()
        ax1.imshow(ref_img)
        ax1.axis('off')
        ax1.set_title(f'{reference_name}\nFrame {frame+1}/{n_frames}', 
                     fontsize=12, fontweight='bold', pad=10)
        
        ax2.clear()
        ax2.imshow(user_img)
        ax2.axis('off')
        ax2.set_title(f'{user_video_name}\nFrame {frame+1}/{n_frames}', 
                     fontsize=12, fontweight='bold', pad=10)
        
        if frame % 30 == 0:
            print(f"  Progress: {frame}/{n_frames} frames ({100*frame//n_frames}%)")
        
        return ax1, ax2
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=n_frames, 
                        interval=1000/fps, blit=False)
    
    # Save video - try MP4 first, fall back to GIF if FFmpeg not available
    print(f"\n  Saving video to: {output_path}")
    
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
            writer = PillowWriter(fps=fps)
            anim.save(gif_path, writer=writer, dpi=100)
            print(f"✓ GIF saved successfully!")
            print(f"  Output: {gif_path}")
            print(f"  Duration: {n_frames/fps:.2f} seconds")
            print(f"  Format: GIF")
            print(f"\n  Note: For MP4 format, install FFmpeg:")
            print(f"    conda install -c conda-forge ffmpeg")
        except Exception as gif_error:
            print(f"✗ Error saving GIF: {gif_error}")
            raise
    except Exception as e:
        print(f"✗ Error saving video: {e}")
        raise
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Generate side-by-side comparison video from existing pose3D images'
    )
    parser.add_argument('--user-images', required=True,
                       help='Directory containing user pose3D images (e.g., user_videos_cache/user/pose3D)')
    parser.add_argument('--reference-images', required=True,
                       help='Directory containing reference pose3D images')
    parser.add_argument('--output', default='comparison_from_images.mp4',
                       help='Output video path')
    parser.add_argument('--user-name', default='Your Form',
                       help='Display name for user')
    parser.add_argument('--reference-name', default='Correct Form',
                       help='Display name for reference')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second')
    
    args = parser.parse_args()
    
    print("="*60)
    print("3D POSE COMPARISON VIDEO FROM IMAGES")
    print("="*60)
    
    create_comparison_video_from_images(
        user_image_dir=args.user_images,
        reference_image_dir=args.reference_images,
        output_path=args.output,
        user_video_name=args.user_name,
        reference_name=args.reference_name,
        fps=args.fps
    )
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == '__main__':
    main()

