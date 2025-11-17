"""
Motion Comparison Module
Main module that compares user poses to reference and generates scores
"""

import numpy as np
from .temporal_align import align_poses_sequences, find_phase_alignment
from .noise_scoring import score_with_statistical_bounds, score_with_noisy_reference
from .utils import normalize_body_scale, center_poses, calculate_joint_distances
from .body_parts import get_joints_for_exercise, get_body_part_joints


def compare_motions(user_poses, ref_poses, noisy_samples=None, exercise_type='pushup', 
                    use_dtw=True, scoring_method='statistical'):
    """
    Compare user motion to reference and generate comprehensive scores
    
    Args:
        user_poses: User pose sequence [frames, 17, 3]
        ref_poses: Reference pose sequence [frames, 17, 3]
        noisy_samples: Pre-generated noisy samples [n_samples, frames, 17, 3] (optional)
        exercise_type: Type of exercise for body part focus
        use_dtw: If True, use DTW for temporal alignment (slower but more accurate)
        scoring_method: 'statistical' (faster) or 'noisy_samples' (more accurate)
    
    Returns:
        Dictionary with comprehensive scoring results
    """
    user_poses = np.array(user_poses)
    ref_poses = np.array(ref_poses)
    
    # Convert lists to arrays if needed
    if isinstance(user_poses, list):
        user_poses = np.array(user_poses)
    if isinstance(ref_poses, list):
        ref_poses = np.array(ref_poses)
    
    print(f"Comparing motions:")
    print(f"  User: {len(user_poses)} frames")
    print(f"  Reference: {len(ref_poses)} frames")
    
    # Step 1: Temporal alignment
    alignment_score = None
    if use_dtw:
        print("\n[1/4] Aligning sequences with DTW...")
        try:
            user_aligned, ref_aligned, alignment_score = find_phase_alignment(user_poses, ref_poses)
            print(f"  Alignment score: {alignment_score:.4f}")
        except Exception as e:
            print(f"  DTW failed, using interpolation: {e}")
            from .utils import interpolate_sequence
            target_length = max(len(user_poses), len(ref_poses))
            user_aligned = interpolate_sequence(user_poses, target_length)
            ref_aligned = interpolate_sequence(ref_poses, target_length)
    else:
        print("\n[1/4] Aligning sequences with interpolation...")
        from .utils import interpolate_sequence
        target_length = max(len(user_poses), len(ref_poses))
        user_aligned = interpolate_sequence(user_poses, target_length)
        ref_aligned = interpolate_sequence(ref_poses, target_length)
    
    # Step 2: Spatial normalization
    print("\n[2/4] Normalizing poses...")
    user_norm, user_scale = normalize_body_scale(user_aligned)
    ref_norm, ref_scale = normalize_body_scale(ref_aligned, reference_scale=user_scale)
    
    # Center both poses at hip
    user_centered = center_poses(user_norm)
    ref_centered = center_poses(ref_norm)
    
    # Step 3: Calculate scores
    print(f"\n[3/4] Calculating scores ({scoring_method} method)...")
    
    if scoring_method == 'noisy_samples' and noisy_samples is not None:
        # Use noisy samples method
        # Align noisy samples too
        from .utils import interpolate_sequence
        target_length = len(user_centered)
        noisy_aligned = np.array([
            interpolate_sequence(sample, target_length)
            for sample in noisy_samples
        ])
        noisy_norm = np.array([
            normalize_body_scale(sample, reference_scale=ref_scale)[0]
            for sample in noisy_aligned
        ])
        noisy_centered = np.array([center_poses(sample) for sample in noisy_norm])
        
        scores = score_with_noisy_reference(
            user_centered, 
            ref_centered, 
            noisy_samples=noisy_centered
        )
    else:
        # Use statistical bounds method (faster)
        scores = score_with_statistical_bounds(user_centered, ref_centered)
    
    # Step 4: Exercise-specific analysis
    print("\n[4/4] Generating exercise-specific feedback...")
    relevant_parts = get_joints_for_exercise(exercise_type)
    
    # Filter scores to relevant body parts
    relevant_scores = {
        part: scores['body_part_scores'][part]
        for part in relevant_parts
        if part in scores['body_part_scores']
    }
    
    # Calculate average for relevant parts
    relevant_avg = np.mean(list(relevant_scores.values())) if relevant_scores else scores['overall_score']
    
    # Generate feedback
    feedback = generate_feedback(scores, relevant_scores, exercise_type)
    
    # Compile results
    results = {
        'overall_score': float(scores['overall_score']),
        'relevant_score': float(relevant_avg),  # Score for exercise-specific body parts
        'body_part_scores': scores['body_part_scores'],
        'relevant_body_part_scores': relevant_scores,
        'frame_scores': scores.get('frame_scores', []),
        'per_joint_scores': scores.get('per_joint_scores', []),
        'feedback': feedback,
        'exercise_type': exercise_type,
        'num_frames_user': len(user_poses),
        'num_frames_ref': len(ref_poses),
        'num_frames_aligned': len(user_centered),
        'details': {
            'reference_poses': ref_centered,
            'user_poses': user_poses,
            'aligned_user_poses': user_centered,
            'body_part_details': scores.get('body_part_details', {}),
            'alignment_score': alignment_score if use_dtw else None,
        }
    }
    
    print(f"\n✓ Comparison complete!")
    print(f"  Overall score: {results['overall_score']:.2f}")
    print(f"  Relevant score: {results['relevant_score']:.2f}")
    
    return results


def generate_feedback(scores, relevant_scores, exercise_type):
    """
    Generate human-readable feedback based on scores
    
    Args:
        scores: Full scoring dictionary
        relevant_scores: Scores for exercise-specific body parts
        exercise_type: Type of exercise
    
    Returns:
        List of feedback strings
    """
    feedback = []
    
    # Overall feedback
    overall = scores['overall_score']
    if overall >= 90:
        feedback.append("Excellent form! Keep up the great work.")
    elif overall >= 75:
        feedback.append("Good form overall. Minor adjustments can improve your technique.")
    elif overall >= 60:
        feedback.append("Decent form, but there's room for improvement.")
    else:
        feedback.append("Focus on improving your form. Consider reviewing the reference video.")
    
    # Body part specific feedback
    if exercise_type.lower() == 'pushup':
        # Check core
        if 'core' in relevant_scores:
            core_score = relevant_scores['core']
            if core_score < 70:
                feedback.append("Keep your core engaged and back straight throughout the movement.")
        
        # Check arms
        arm_scores = [relevant_scores.get('right_arm', 0), relevant_scores.get('left_arm', 0)]
        avg_arm = np.mean(arm_scores)
        if avg_arm < 70:
            feedback.append("Focus on maintaining consistent arm positioning. Both arms should move symmetrically.")
        elif abs(arm_scores[0] - arm_scores[1]) > 15:
            feedback.append("Your arms are moving asymmetrically. Try to keep both sides balanced.")
    
    elif exercise_type.lower() == 'squat':
        # Check legs
        leg_scores = [relevant_scores.get('right_leg', 0), relevant_scores.get('left_leg', 0)]
        avg_leg = np.mean(leg_scores)
        if avg_leg < 70:
            feedback.append("Focus on proper leg positioning and depth in your squats.")
        elif abs(leg_scores[0] - leg_scores[1]) > 15:
            feedback.append("Your legs are moving asymmetrically. Focus on balanced movement.")
    
    # Find worst performing body part
    if relevant_scores:
        worst_part = min(relevant_scores.items(), key=lambda x: x[1])
        if worst_part[1] < 65:
            feedback.append(f"Pay special attention to your {worst_part[0].replace('_', ' ')} - it needs the most improvement.")
    
    return feedback


def score_exercise(user_video_path, reference_id='pushup', references_dir='references', 
                   use_dtw=True, scoring_method='statistical', force_reprocess=False):
    """
    Complete pipeline: process user video and score against reference
    
    Args:
        user_video_path: Path to user video
        reference_id: Exercise type / reference ID
        references_dir: Directory containing references
        use_dtw: Use DTW for alignment
        scoring_method: Scoring method to use
        force_reprocess: Force reprocessing even if cached data exists
    
    Returns:
        Scoring results dictionary
    """
    from .user_processor import process_user_video
    from .reference_processor import load_reference
    import shutil
    from pathlib import Path
    
    print("="*60)
    print("EXERCISE SCORING PIPELINE")
    print("="*60)
    
    # Load reference
    print(f"\nLoading reference: {reference_id}")
    ref_data = load_reference(reference_id, references_dir=references_dir)
    ref_poses = ref_data['poses_3d']
    noisy_samples = ref_data.get('noisy_samples')
    metadata = ref_data['metadata']
    
    print(f"  Reference frames: {len(ref_poses)}")
    print(f"  Exercise type: {metadata['exercise_type']}")
    
    # Clear cache if force reprocess
    if force_reprocess:
        cache_dir = Path('user_videos_cache') / Path(user_video_path).stem
        if cache_dir.exists():
            print(f"\n⚠ Clearing cache for {Path(user_video_path).name}")
            shutil.rmtree(cache_dir)
    
    # Process user video (uses cache if available)
    print(f"\nProcessing user video: {user_video_path}")
    user_data = process_user_video(user_video_path, cleanup=False)
    user_poses = user_data['poses_3d']
    
    print(f"  User frames: {len(user_poses)}")
    
    # Compare
    print(f"\nComparing motions...")
    results = compare_motions(
        user_poses,
        ref_poses,
        noisy_samples=noisy_samples,
        exercise_type=metadata['exercise_type'],
        use_dtw=use_dtw,
        scoring_method=scoring_method
    )
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare user video to reference')
    parser.add_argument('--user-video', type=str, required=True, help='Path to user video')
    parser.add_argument('--reference', type=str, default='pushup', help='Reference ID')
    parser.add_argument('--references-dir', type=str, default='references', help='References directory')
    parser.add_argument('--no-dtw', action='store_true', help='Disable DTW alignment')
    parser.add_argument('--method', type=str, default='statistical', choices=['statistical', 'noisy_samples'],
                       help='Scoring method')
    parser.add_argument('--force-reprocess', action='store_true', help='Force reprocessing (ignore cache)')
    parser.add_argument('--json', action='store_true', help='Output results as JSON for API consumption')
    parser.add_argument('--output', type=str, help='Save JSON output to file')
    parser.add_argument('--generate-video', action='store_true', help='Generate side-by-side comparison video')
    parser.add_argument('--video-output', type=str, help='Path for comparison video (default: comparison_<user_video>.mp4)')
    parser.add_argument('--video-fps', type=int, default=30, help='FPS for comparison video')
    
    args = parser.parse_args()
    
    try:
        results = score_exercise(
            args.user_video,
            reference_id=args.reference,
            references_dir=args.references_dir,
            use_dtw=not args.no_dtw,
            scoring_method=args.method,
            force_reprocess=args.force_reprocess
        )
        
        # Format output for API/LLM consumption
        if args.json:
            import json
            from pathlib import Path
            
            # Create clean API response
            api_response = {
                "status": "success",
                "exercise": {
                    "type": results['exercise_type'],
                    "reference": args.reference,
                    "user_video": str(Path(args.user_video).name)
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
                    "alignment_quality": float(round(results['details'].get('alignment_score', 0), 4)) if results['details'].get('alignment_score') else None,
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
                },
                "feedback": results['feedback'],
                "llm_context": {
                    "description": f"User performed {results['exercise_type']} exercise",
                    "scoring_method": args.method,
                    "interpretation": {
                        "score_range": "0-100, where 100 is perfect form matching the reference",
                        "position_error": "Lower is better. Measures average distance from reference pose in normalized units",
                        "in_tolerance": "Percentage of time user's form was within acceptable bounds"
                    }
                }
            }
            
            # Output to file or stdout
            json_output = json.dumps(api_response, indent=2)
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(json_output)
                print(f"✓ Results saved to {args.output}")
            else:
                print(json_output)
        else:
            # Human-readable output
            print("\n" + "="*60)
            print("SCORING RESULTS")
            print("="*60)
            print(f"\nOverall Score: {results['overall_score']:.2f}/100")
            print(f"Relevant Score: {results['relevant_score']:.2f}/100")
            print(f"\nBody Part Scores:")
            for part, score in results['relevant_body_part_scores'].items():
                print(f"  {part.replace('_', ' ').title()}: {score:.2f}/100")
            print(f"\nFeedback:")
            for i, fb in enumerate(results['feedback'], 1):
                print(f"  {i}. {fb}")
            
            # Debug information
            print("\n" + "="*60)
            print("DEBUG INFORMATION")
            print("="*60)
            details = results.get('details', {})
            print(f"\nFrame Counts:")
            print(f"  Reference frames: {len(details.get('reference_poses', []))}")
            print(f"  User frames (original): {len(details.get('user_poses', []))}")
            print(f"  User frames (aligned): {len(details.get('aligned_user_poses', []))}")
            
            if details.get('alignment_score') is not None:
                print(f"\nAlignment:")
                print(f"  DTW alignment score: {details['alignment_score']:.4f}")
            
            print(f"\nDetailed Body Part Metrics:")
            for part, metrics in details.get('body_part_details', {}).items():
                if part in results['relevant_body_part_scores']:
                    print(f"\n{part.replace('_', ' ').title()}:")
                    print(f"  Position Error (avg): {metrics.get('position_error', 0):.4f}")
                    print(f"  Position Error (max): {metrics.get('max_position_error', 0):.4f}")
                    print(f"  Tolerance Threshold: {metrics.get('tolerance_threshold', 0):.4f}")
                    print(f"  In-tolerance %: {metrics.get('in_tolerance_percentage', 0):.1f}%")
        
        # Generate comparison video if requested
        if args.generate_video:
            from pathlib import Path
            
            print("\n" + "="*60)
            print("GENERATING COMPARISON VIDEO")
            print("="*60)
            
            try:
                from .video_from_images import create_comparison_video_from_images
                from .user_processor import process_user_video
                from .reference_processor import load_reference
                
                # Determine output path
                if args.video_output:
                    video_output = args.video_output
                else:
                    user_video_stem = Path(args.user_video).stem
                    video_output = f"comparison_{user_video_stem}.mp4"
                
                # Find the pose3D image directories
                # User images: user_videos_cache/{video_name}/pose3D
                user_video_name = Path(args.user_video).stem
                user_image_dir = Path('user_videos_cache') / user_video_name / 'pose3D'
                
                # Reference images: references/{exercise}/temp_processing/pose3D
                ref_data = load_reference(args.reference, references_dir=args.references_dir)
                ref_dir = Path(ref_data['ref_dir'])
                reference_image_dir = ref_dir / 'temp_processing' / 'pose3D'
                
                # Check if directories exist
                if not user_image_dir.exists():
                    print(f"⚠ Warning: User pose3D images not found at {user_image_dir}")
                    print("  Attempting to process user video to generate images...")
                    process_user_video(args.user_video, cleanup=False)
                    user_image_dir = Path('user_videos_cache') / user_video_name / 'pose3D'
                
                if not reference_image_dir.exists():
                    # Try alternative location
                    reference_image_dir = ref_dir / 'pose3D'
                    if not reference_image_dir.exists():
                        raise FileNotFoundError(
                            f"Reference pose3D images not found. Tried:\n"
                            f"  {ref_dir / 'temp_processing' / 'pose3D'}\n"
                            f"  {ref_dir / 'pose3D'}"
                        )
                
                print(f"  User images: {user_image_dir}")
                print(f"  Reference images: {reference_image_dir}")
                
                # Create the video from existing images
                create_comparison_video_from_images(
                    user_image_dir=str(user_image_dir),
                    reference_image_dir=str(reference_image_dir),
                    output_path=video_output,
                    user_video_name="Your Form",
                    reference_name="Correct Form",
                    fps=args.video_fps
                )
                
            except ImportError as e:
                print(f"✗ Error: Missing dependency for video generation")
                print(f"  {e}")
                print("\nPlease ensure matplotlib and ffmpeg are installed:")
                print("  pip install matplotlib")
                print("  And install FFmpeg from: https://ffmpeg.org/download.html")
            except Exception as e:
                print(f"✗ Error generating comparison video: {e}")
                import traceback
                traceback.print_exc()
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

