"""
Standalone script to process reference videos
Usage: python demo/process_reference.py --video path/to/video.mp4 --exercise pushup
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fitness_coach.reference_processor import process_reference_video

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process reference video for scoring system')
    parser.add_argument('--video', type=str, required=True, 
                       help='Path to reference video file')
    parser.add_argument('--exercise', type=str, default='pushup',
                       help='Exercise type (pushup, squat, etc.)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: references/{exercise}/)')
    parser.add_argument('--samples', type=int, default=100,
                       help='Number of noisy samples to generate')
    
    args = parser.parse_args()
    
    try:
        result = process_reference_video(
            args.video,
            exercise_type=args.exercise,
            output_dir=args.output,
            n_samples=args.samples
        )
        print("\n" + "="*60)
        print("SUCCESS!")
        print("="*60)
        print(f"Reference processed and saved to:")
        print(f"  {result['output_dir']}")
        print(f"\nFiles created:")
        print(f"  - 3D poses: {result['poses_3d_path']}")
        print(f"  - Noisy samples: {result['noisy_samples_path']}")
        print(f"  - Statistical bounds: {result['bounds_path']}")
        print(f"  - Metadata: {result['metadata_path']}")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

