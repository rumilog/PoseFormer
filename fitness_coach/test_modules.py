"""
Test script for fitness_coach modules
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fitness_coach.body_parts import (
    get_body_part_joints,
    get_joint_noise_level,
    get_joints_for_exercise,
    calculate_body_scale
)
from fitness_coach.utils import (
    normalize_body_scale,
    center_poses,
    calculate_joint_distances,
    interpolate_sequence
)
from fitness_coach.temporal_align import (
    align_sequences_dtw,
    align_poses_sequences
)
from fitness_coach.noise_scoring import (
    create_noisy_samples,
    score_with_statistical_bounds
)


def test_body_parts():
    """Test body parts module"""
    print("=" * 50)
    print("Testing body_parts module...")
    print("=" * 50)
    
    # Test getting body part joints
    right_arm = get_body_part_joints('right_arm')
    print(f"Right arm joints: {right_arm}")
    assert right_arm == [14, 15, 16], "Right arm joints incorrect"
    
    # Test noise levels
    hip_noise = get_joint_noise_level(0)
    print(f"Hip noise level: {hip_noise}")
    assert hip_noise == 0.02, "Hip noise level incorrect"
    
    # Test exercise focus
    pushup_parts = get_joints_for_exercise('pushup')
    print(f"Push-up body parts: {pushup_parts}")
    assert 'core' in pushup_parts, "Push-up should include core"
    
    # Test body scale calculation
    test_poses = np.random.randn(10, 17, 3)
    scale = calculate_body_scale(test_poses)
    print(f"Body scale: {scale:.4f}")
    assert scale > 0, "Body scale should be positive"
    
    print("[OK] body_parts module tests passed!\n")


def test_utils():
    """Test utils module"""
    print("=" * 50)
    print("Testing utils module...")
    print("=" * 50)
    
    # Test normalization
    test_poses = np.random.randn(10, 17, 3) * 10
    normalized, scale = normalize_body_scale(test_poses)
    print(f"Normalization: scale = {scale:.4f}")
    assert normalized.shape == test_poses.shape, "Normalized shape should match"
    
    # Test centering
    centered = center_poses(test_poses)
    hip_pos = centered[0, 0]
    print(f"Centering: hip position = {hip_pos}")
    assert np.allclose(hip_pos, [0, 0, 0]), "Hip should be at origin"
    
    # Test distances
    pose1 = test_poses[0]
    pose2 = test_poses[1]
    dists = calculate_joint_distances(pose1, pose2)
    print(f"Joint distances: mean = {np.mean(dists):.4f}")
    assert len(dists) == 17, "Should have 17 joint distances"
    
    # Test interpolation
    short_seq = np.random.randn(5, 17, 3)
    long_seq = interpolate_sequence(short_seq, 10)
    print(f"Interpolation: {len(short_seq)} -> {len(long_seq)} frames")
    assert len(long_seq) == 10, "Interpolated length should be 10"
    
    print("[OK] utils module tests passed!\n")


def test_temporal_align():
    """Test temporal alignment module"""
    print("=" * 50)
    print("Testing temporal_align module...")
    print("=" * 50)
    
    # Create sequences of different lengths
    seq1 = np.random.randn(30, 17, 3)
    seq2 = np.random.randn(50, 17, 3)
    
    print(f"Original lengths: {len(seq1)} vs {len(seq2)}")
    
    # Test alignment
    aligned_seq1, aligned_seq2, path = align_sequences_dtw(seq1, seq2)
    print(f"Aligned lengths: {len(aligned_seq1)} vs {len(aligned_seq2)}")
    assert len(aligned_seq1) == len(aligned_seq2), "Aligned sequences should have same length"
    
    # Test pose sequence alignment
    aligned_poses1, aligned_poses2 = align_poses_sequences(seq1, seq2)
    print(f"Pose alignment: {len(aligned_poses1)} vs {len(aligned_poses2)}")
    assert len(aligned_poses1) == len(aligned_poses2), "Aligned poses should have same length"
    
    print("[OK] temporal_align module tests passed!\n")


def test_noise_scoring():
    """Test noise scoring module"""
    print("=" * 50)
    print("Testing noise_scoring module...")
    print("=" * 50)
    
    # Create test data
    ref_poses = np.random.randn(20, 17, 3)
    user_poses = ref_poses + np.random.normal(0, 0.05, ref_poses.shape)
    
    # Test noisy sample generation
    noisy_samples = create_noisy_samples(ref_poses, n_samples=20)
    print(f"Generated {len(noisy_samples)} noisy samples")
    assert noisy_samples.shape == (20, 20, 17, 3), "Noisy samples shape incorrect"
    
    # Test scoring
    scores = score_with_statistical_bounds(user_poses, ref_poses)
    print(f"Overall score: {scores['overall_score']:.2f}")
    print(f"Body part scores: {list(scores['body_part_scores'].keys())}")
    
    assert 'overall_score' in scores, "Should have overall_score"
    assert 'body_part_scores' in scores, "Should have body_part_scores"
    assert 0 <= scores['overall_score'] <= 100, "Score should be 0-100"
    
    print("[OK] noise_scoring module tests passed!\n")


def main():
    """Run all tests"""
    print("\n" + "=" * 50)
    print("FITNESS COACH MODULE TESTS")
    print("=" * 50 + "\n")
    
    try:
        test_body_parts()
        test_utils()
        test_temporal_align()
        test_noise_scoring()
        
        print("=" * 50)
        print("ALL TESTS PASSED! [OK]")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n[FAILED] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

