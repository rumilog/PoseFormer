"""
Noise-Based Scoring System
Generates noisy reference samples and scores user poses against them
"""

import numpy as np
from .body_parts import (
    get_joint_noise_level,
    calculate_body_scale,
    JOINT_GROUPS
)


def create_noisy_samples(ref_poses, n_samples=100, noise_std=None, per_joint_noise=True):
    """
    Create noisy reference samples for scoring
    
    Args:
        ref_poses: Reference poses [frames, 17, 3]
        n_samples: Number of noisy samples to generate
        noise_std: Overall noise standard deviation (as fraction of body scale)
                  If None, uses per-joint noise levels
        per_joint_noise: If True, use different noise levels per joint
    
    Returns:
        noisy_samples: Array of shape [n_samples, frames, 17, 3]
    """
    ref_poses = np.array(ref_poses)
    body_scale = calculate_body_scale(ref_poses)
    
    noisy_samples = []
    
    for _ in range(n_samples):
        noisy_pose = ref_poses.copy()
        
        for frame_idx in range(len(ref_poses)):
            for joint_idx in range(17):
                if per_joint_noise:
                    # Use joint-specific noise level
                    joint_noise_std = get_joint_noise_level(joint_idx) * body_scale
                else:
                    # Use uniform noise
                    if noise_std is None:
                        noise_std = 0.05  # Default 5% of body scale
                    joint_noise_std = noise_std * body_scale
                
                # Add Gaussian noise to each coordinate
                noise = np.random.normal(
                    loc=0.0,
                    scale=joint_noise_std,
                    size=3
                )
                noisy_pose[frame_idx, joint_idx, :] += noise
        
        noisy_samples.append(noisy_pose)
    
    return np.array(noisy_samples)


def calculate_statistical_bounds(ref_poses, noise_std=0.015, confidence=0.95):
    """
    Calculate statistical bounds (mean ± std) for reference poses
    
    Args:
        ref_poses: Reference poses [frames, 17, 3]
        noise_std: Noise standard deviation (as fraction of body scale, default 1.5%)
        confidence: Confidence level (0.95 = 95%)
    
    Returns:
        mean_poses: Mean poses [frames, 17, 3]
        lower_bound: Lower bound [frames, 17, 3]
        upper_bound: Upper bound [frames, 17, 3]
        tolerance: Tolerance per joint [frames, 17]
    """
    ref_poses = np.array(ref_poses)
    body_scale = calculate_body_scale(ref_poses)
    
    # Generate many samples and calculate statistics
    n_samples = 1000
    noisy_samples = create_noisy_samples(
        ref_poses, 
        n_samples=n_samples, 
        noise_std=noise_std,
        per_joint_noise=False
    )
    
    # Calculate mean and std
    mean_poses = np.mean(noisy_samples, axis=0)
    std_poses = np.std(noisy_samples, axis=0)
    
    # Calculate bounds based on confidence level
    # For 95% confidence, use ~2 standard deviations
    z_score = 1.96 if confidence == 0.95 else 2.576  # 99% confidence
    
    lower_bound = mean_poses - z_score * std_poses
    upper_bound = mean_poses + z_score * std_poses
    
    # Tolerance is the distance from mean to bound
    tolerance = z_score * std_poses
    
    return mean_poses, lower_bound, upper_bound, tolerance


def score_with_noisy_reference(user_poses, ref_poses, noisy_samples=None, n_samples=100):
    """
    Score user poses against noisy reference samples
    
    Args:
        user_poses: User pose sequence [frames, 17, 3]
        ref_poses: Reference pose sequence [frames, 17, 3]
        noisy_samples: Pre-generated noisy samples [n_samples, frames, 17, 3]
                      If None, generates them
        n_samples: Number of samples to generate if noisy_samples is None
    
    Returns:
        scores: Dictionary with overall and per-body-part scores
    """
    user_poses = np.array(user_poses)
    ref_poses = np.array(ref_poses)
    
    # Generate noisy samples if not provided
    if noisy_samples is None:
        noisy_samples = create_noisy_samples(ref_poses, n_samples=n_samples)
    
    # Align temporally (simple resampling for now, DTW in comparison.py)
    from .utils import interpolate_sequence
    target_length = max(len(user_poses), len(ref_poses))
    user_aligned = interpolate_sequence(user_poses, target_length)
    ref_aligned = interpolate_sequence(ref_poses, target_length)
    noisy_aligned = np.array([
        interpolate_sequence(sample, target_length) 
        for sample in noisy_samples
    ])
    
    # Normalize by body scale
    from .utils import normalize_body_scale
    user_norm, _ = normalize_body_scale(user_aligned)
    ref_norm, ref_scale = normalize_body_scale(ref_aligned)
    noisy_norm = np.array([
        normalize_body_scale(sample, reference_scale=ref_scale)[0]
        for sample in noisy_aligned
    ])
    
    # Calculate scores per body part
    body_part_scores = {}
    frame_scores = []
    
    for frame_idx in range(len(user_norm)):
        user_frame = user_norm[frame_idx]  # [17, 3]
        ref_frame = ref_norm[frame_idx]
        noisy_frames = noisy_norm[:, frame_idx, :, :]  # [n_samples, 17, 3]
        
        # Calculate distance from user to reference
        user_to_ref_dist = np.linalg.norm(user_frame - ref_frame, axis=1)  # [17]
        
        # Calculate distances from each noisy sample to reference
        noisy_to_ref_dists = np.array([
            np.linalg.norm(noisy_frame - ref_frame, axis=1)
            for noisy_frame in noisy_frames
        ])  # [n_samples, 17]
        
        # Score: percentage of noisy samples that are "worse" than user
        # (i.e., user is within acceptable range)
        frame_scores_per_joint = []
        for joint_idx in range(17):
            user_dist = user_to_ref_dist[joint_idx]
            noisy_dists = noisy_to_ref_dists[:, joint_idx]
            
            # How many noisy samples are further from reference than user?
            better_than = np.sum(noisy_dists > user_dist)
            score = (better_than / len(noisy_dists)) * 100
            frame_scores_per_joint.append(score)
        
        frame_scores.append(frame_scores_per_joint)
    
    frame_scores = np.array(frame_scores)  # [frames, 17]
    
    # Aggregate by body part
    for part_name, joint_indices in JOINT_GROUPS.items():
        part_scores = frame_scores[:, joint_indices]
        body_part_scores[part_name] = float(np.mean(part_scores))
    
    # Overall score
    overall_score = float(np.mean(frame_scores))
    
    return {
        'overall_score': overall_score,
        'body_part_scores': body_part_scores,
        'frame_scores': frame_scores.tolist(),
        'per_joint_scores': np.mean(frame_scores, axis=0).tolist()
    }


def score_with_statistical_bounds(user_poses, ref_poses, noise_std=0.015):
    """
    Score using statistical bounds (faster than noisy samples)
    
    Args:
        user_poses: User pose sequence [frames, 17, 3]
        ref_poses: Reference pose sequence [frames, 17, 3]
        noise_std: Noise standard deviation (as fraction of body scale)
                   Default 0.015 = 1.5% tolerance, good for form checking
    
    Returns:
        scores: Dictionary with overall and per-body-part scores
    """
    user_poses = np.array(user_poses)
    ref_poses = np.array(ref_poses)
    
    # Calculate bounds (already accounts for body scale)
    mean_poses, lower_bound, upper_bound, tolerance = calculate_statistical_bounds(
        ref_poses, noise_std=noise_std
    )
    
    # Align temporally
    from .utils import interpolate_sequence
    target_length = max(len(user_poses), len(ref_poses))
    user_aligned = interpolate_sequence(user_poses, target_length)
    mean_aligned = interpolate_sequence(mean_poses, target_length)
    
    # Normalize poses (but not tolerance - it's already in the right scale)
    from .utils import normalize_body_scale
    user_norm, user_scale = normalize_body_scale(user_aligned)
    mean_norm, _ = normalize_body_scale(mean_aligned)
    
    # Scale the tolerance by the same factor used for normalization
    # This keeps it proportional to the noise_std parameter
    body_scale = calculate_body_scale(user_aligned)
    tolerance_scaled = tolerance * (1.0 / body_scale)
    tolerance_aligned = interpolate_sequence(tolerance_scaled, target_length)
    
    # Check if user poses are within tolerance
    distances = np.linalg.norm(user_norm - mean_norm, axis=2)  # [frames, 17]
    tolerance_per_joint = np.linalg.norm(tolerance_aligned, axis=2)  # [frames, 17]
    
    # Score: percentage of time within tolerance
    within_tolerance = distances < tolerance_per_joint
    joint_scores = np.mean(within_tolerance, axis=0) * 100  # [17]
    frame_scores = np.mean(within_tolerance, axis=1) * 100  # [frames]
    
    # Aggregate by body part with detailed metrics
    body_part_scores = {}
    body_part_details = {}
    
    for part_name, joint_indices in JOINT_GROUPS.items():
        # Score
        body_part_scores[part_name] = float(np.mean(joint_scores[joint_indices]))
        
        # Detailed metrics for this body part
        part_distances = distances[:, joint_indices]  # [frames, num_joints_in_part]
        part_tolerance = tolerance_per_joint[:, joint_indices]
        part_within = within_tolerance[:, joint_indices]
        
        body_part_details[part_name] = {
            'position_error': float(np.mean(part_distances)),
            'max_position_error': float(np.max(part_distances)),
            'in_tolerance_percentage': float(np.mean(part_within) * 100),
            'tolerance_threshold': float(np.mean(part_tolerance)),
        }
    
    overall_score = float(np.mean(frame_scores))
    
    return {
        'overall_score': overall_score,
        'body_part_scores': body_part_scores,
        'body_part_details': body_part_details,
        'frame_scores': frame_scores.tolist(),
        'per_joint_scores': joint_scores.tolist()
    }


if __name__ == "__main__":
    # Test noise scoring
    print("Testing noise-based scoring...")
    
    # Create test data
    ref_poses = np.random.randn(50, 17, 3)
    user_poses = ref_poses + np.random.normal(0, 0.1, ref_poses.shape)  # Slightly different
    
    # Test noisy sample generation
    noisy_samples = create_noisy_samples(ref_poses, n_samples=50)
    print(f"Generated {len(noisy_samples)} noisy samples")
    print(f"Noisy samples shape: {noisy_samples.shape}")
    
    # Test statistical bounds
    mean, lower, upper, tolerance = calculate_statistical_bounds(ref_poses)
    print(f"Statistical bounds calculated: mean shape {mean.shape}")
    
    # Test scoring
    scores = score_with_statistical_bounds(user_poses, ref_poses)
    print(f"\nScoring results:")
    print(f"  Overall score: {scores['overall_score']:.2f}")
    print(f"  Body part scores: {scores['body_part_scores']}")
    
    print("\nNoise scoring tests passed!")

