"""
Temporal Alignment using Dynamic Time Warping (DTW)
Aligns sequences of different lengths for comparison
"""

import numpy as np
try:
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
    HAS_FASTDTW = True
except ImportError:
    HAS_FASTDTW = False
    print("Warning: fastdtw not installed. Using simple interpolation instead.")
    print("Install with: pip install fastdtw")


def align_sequences_dtw(seq1, seq2, distance_func=None):
    """
    Align two sequences using Dynamic Time Warping
    
    Args:
        seq1: First sequence [frames, ...]
        seq2: Second sequence [frames, ...]
        distance_func: Distance function (default: euclidean)
    
    Returns:
        aligned_seq1, aligned_seq2: Aligned sequences of same length
        path: DTW alignment path
    """
    if not HAS_FASTDTW:
        # Fallback: simple interpolation to same length
        target_length = max(len(seq1), len(seq2))
        from .utils import interpolate_sequence
        if len(seq1.shape) == 3:  # [frames, joints, coords]
            aligned_seq1 = interpolate_sequence(seq1, target_length)
            aligned_seq2 = interpolate_sequence(seq2, target_length)
        else:
            # Flatten for interpolation
            original_shape1 = seq1.shape
            original_shape2 = seq2.shape
            seq1_flat = seq1.reshape(len(seq1), -1)
            seq2_flat = seq2.reshape(len(seq2), -1)
            aligned_seq1_flat = interpolate_sequence(seq1_flat, target_length)
            aligned_seq2_flat = interpolate_sequence(seq2_flat, target_length)
            aligned_seq1 = aligned_seq1_flat.reshape((target_length,) + original_shape1[1:])
            aligned_seq2 = aligned_seq2_flat.reshape((target_length,) + original_shape2[1:])
        return aligned_seq1, aligned_seq2, None
    
    # Flatten sequences for DTW
    seq1_flat = seq1.reshape(len(seq1), -1)
    seq2_flat = seq2.reshape(len(seq2), -1)
    
    # Use provided distance function or default
    if distance_func is None:
        distance_func = euclidean
    
    # Compute DTW
    distance, path = fastdtw(seq1_flat, seq2_flat, dist=distance_func)
    
    # Create aligned sequences using the path
    aligned_seq1_indices = [p[0] for p in path]
    aligned_seq2_indices = [p[1] for p in path]
    
    aligned_seq1 = seq1[aligned_seq1_indices]
    aligned_seq2 = seq2[aligned_seq2_indices]
    
    return aligned_seq1, aligned_seq2, path


def align_poses_sequences(poses1, poses2):
    """
    Align two pose sequences temporally
    
    Args:
        poses1: First pose sequence [frames, 17, 3]
        poses2: Second pose sequence [frames, 17, 3]
    
    Returns:
        aligned_poses1, aligned_poses2: Aligned pose sequences
    """
    poses1 = np.array(poses1)
    poses2 = np.array(poses2)
    
    # Use DTW to align
    aligned_poses1, aligned_poses2, _ = align_sequences_dtw(poses1, poses2)
    
    return aligned_poses1, aligned_poses2


def find_phase_alignment(user_poses, ref_poses):
    """
    Find optimal phase alignment between user and reference sequences
    Uses DTW to handle different speeds and timing
    
    Args:
        user_poses: User pose sequence [frames, 17, 3]
        ref_poses: Reference pose sequence [frames, 17, 3]
    
    Returns:
        aligned_user, aligned_ref: Phase-aligned sequences
        alignment_score: Quality of alignment (lower is better)
    """
    user_poses = np.array(user_poses)
    ref_poses = np.array(ref_poses)
    
    # Align sequences
    aligned_user, aligned_ref, path = align_sequences_dtw(user_poses, ref_poses)
    
    # Calculate alignment quality (mean distance after alignment)
    if path is not None and HAS_FASTDTW:
        # Calculate average distance along path
        distances = []
        for i, j in path:
            dist = np.linalg.norm(user_poses[i] - ref_poses[j])
            distances.append(dist)
        alignment_score = np.mean(distances)
    else:
        # Fallback: mean distance between aligned sequences
        alignment_score = np.mean(np.linalg.norm(aligned_user - aligned_ref, axis=2))
    
    return aligned_user, aligned_ref, alignment_score


def resample_to_common_length(poses1, poses2, target_length=None):
    """
    Resample both sequences to common length
    
    Args:
        poses1: First pose sequence [frames, 17, 3]
        poses2: Second pose sequence [frames, 17, 3]
        target_length: Target length (default: average of both)
    
    Returns:
        resampled_poses1, resampled_poses2: Resampled sequences
    """
    from fitness_coach.utils import interpolate_sequence
    
    poses1 = np.array(poses1)
    poses2 = np.array(poses2)
    
    if target_length is None:
        target_length = (len(poses1) + len(poses2)) // 2
    
    resampled_poses1 = interpolate_sequence(poses1, target_length)
    resampled_poses2 = interpolate_sequence(poses2, target_length)
    
    return resampled_poses1, resampled_poses2


if __name__ == "__main__":
    # Test temporal alignment
    print("Testing temporal alignment...")
    
    # Create test sequences of different lengths
    seq1 = np.random.randn(50, 17, 3)
    seq2 = np.random.randn(75, 17, 3)
    
    print(f"Original lengths: {len(seq1)} vs {len(seq2)}")
    
    # Test alignment
    aligned_seq1, aligned_seq2, path = align_sequences_dtw(seq1, seq2)
    print(f"Aligned lengths: {len(aligned_seq1)} vs {len(aligned_seq2)}")
    
    if path is not None:
        print(f"DTW path length: {len(path)}")
    else:
        print("Using interpolation fallback")
    
    # Test phase alignment
    aligned_user, aligned_ref, score = find_phase_alignment(seq1, seq2)
    print(f"Alignment score: {score:.4f}")
    
    print("Temporal alignment tests passed!")

