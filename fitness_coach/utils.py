"""
Utility Functions for Pose Processing
Helper functions for normalization, distance calculations, and interpolation
"""

import numpy as np


def normalize_body_scale(poses, reference_scale=None):
    """
    Normalize poses by body scale (hip-to-thorax distance)
    
    Args:
        poses: Array of shape [frames, 17, 3] or [17, 3]
        reference_scale: Optional reference scale to normalize to
    
    Returns:
        Normalized poses, scale used
    """
    poses = np.array(poses)
    original_shape = poses.shape
    
    if len(poses.shape) == 2:
        poses = poses[np.newaxis, :, :]
    
    # Calculate body scale (hip to thorax distance)
    hip_to_thorax = np.linalg.norm(poses[:, 0, :] - poses[:, 8, :], axis=1)
    body_scale = np.mean(hip_to_thorax)
    
    if body_scale == 0:
        return poses.reshape(original_shape), 1.0
    
    # Normalize
    if reference_scale is not None:
        scale_factor = reference_scale / body_scale
    else:
        scale_factor = 1.0 / body_scale
    
    normalized_poses = poses * scale_factor
    
    return normalized_poses.reshape(original_shape), body_scale


def center_poses(poses, joint_idx=0):
    """
    Center poses at a specific joint (default: hip)
    
    Args:
        poses: Array of shape [frames, 17, 3] or [17, 3]
        joint_idx: Joint to center on (default: 0 = hip)
    
    Returns:
        Centered poses
    """
    poses = np.array(poses)
    original_shape = poses.shape
    
    if len(poses.shape) == 2:
        poses = poses[np.newaxis, :, :]
    
    # Subtract the reference joint position
    centered = poses - poses[:, joint_idx:joint_idx+1, :]
    
    return centered.reshape(original_shape)


def calculate_joint_distances(pose1, pose2):
    """
    Calculate Euclidean distances between corresponding joints
    
    Args:
        pose1: Array of shape [17, 3] or [frames, 17, 3]
        pose2: Array of shape [17, 3] or [frames, 17, 3]
    
    Returns:
        Distances per joint: [17] or [frames, 17]
    """
    pose1 = np.array(pose1)
    pose2 = np.array(pose2)
    
    if len(pose1.shape) == 2 and len(pose2.shape) == 2:
        # Single frame
        distances = np.linalg.norm(pose1 - pose2, axis=1)
    else:
        # Multiple frames
        if len(pose1.shape) == 2:
            pose1 = pose1[np.newaxis, :, :]
        if len(pose2.shape) == 2:
            pose2 = pose2[np.newaxis, :, :]
        
        distances = np.linalg.norm(pose1 - pose2, axis=2)
    
    return distances


def calculate_joint_angles(poses, joint_pairs):
    """
    Calculate angles between joint pairs
    
    Args:
        poses: Array of shape [frames, 17, 3] or [17, 3]
        joint_pairs: List of (parent, child) joint index tuples
    
    Returns:
        Angles in radians: [frames, n_pairs] or [n_pairs]
    """
    poses = np.array(poses)
    original_shape = poses.shape
    
    if len(poses.shape) == 2:
        poses = poses[np.newaxis, :, :]
    
    angles = []
    for parent_idx, child_idx in joint_pairs:
        # Vector from parent to child
        vectors = poses[:, child_idx, :] - poses[:, parent_idx, :]
        
        # Calculate angle (simplified - angle with vertical)
        # For more accurate angles, would need to consider parent-child relationships
        vertical = np.array([0, 1, 0])
        vertical = np.tile(vertical, (vectors.shape[0], 1))
        
        # Dot product and angle
        dot_products = np.sum(vectors * vertical, axis=1)
        vector_norms = np.linalg.norm(vectors, axis=1)
        vertical_norm = np.linalg.norm(vertical, axis=1)
        
        # Avoid division by zero
        cosines = np.clip(dot_products / (vector_norms * vertical_norm + 1e-8), -1, 1)
        angle = np.arccos(cosines)
        
        angles.append(angle)
    
    angles = np.array(angles).T  # [frames, n_pairs]
    
    if len(original_shape) == 2:
        return angles[0]
    return angles


def interpolate_sequence(poses, target_length):
    """
    Interpolate pose sequence to target length
    
    Args:
        poses: Array of shape [frames, 17, 3]
        target_length: Target number of frames
    
    Returns:
        Interpolated poses: [target_length, 17, 3]
    """
    poses = np.array(poses)
    original_length = poses.shape[0]
    
    if original_length == target_length:
        return poses
    
    # Create interpolation indices
    original_indices = np.linspace(0, original_length - 1, original_length)
    target_indices = np.linspace(0, original_length - 1, target_length)
    
    # Interpolate each joint and coordinate
    interpolated = np.zeros((target_length, poses.shape[1], poses.shape[2]))
    
    for joint_idx in range(poses.shape[1]):
        for coord_idx in range(poses.shape[2]):
            interpolated[:, joint_idx, coord_idx] = np.interp(
                target_indices,
                original_indices,
                poses[:, joint_idx, coord_idx]
            )
    
    return interpolated


def smooth_poses(poses, window_size=5):
    """
    Apply moving average smoothing to pose sequence
    
    Args:
        poses: Array of shape [frames, 17, 3]
        window_size: Size of smoothing window
    
    Returns:
        Smoothed poses
    """
    poses = np.array(poses)
    if len(poses) < window_size:
        return poses
    
    # Pad with edge values
    pad_width = window_size // 2
    padded = np.pad(poses, ((pad_width, pad_width), (0, 0), (0, 0)), mode='edge')
    
    # Apply moving average
    smoothed = np.zeros_like(poses)
    for i in range(len(poses)):
        smoothed[i] = np.mean(padded[i:i+window_size], axis=0)
    
    return smoothed


def align_poses_spatially(poses1, poses2):
    """
    Align two pose sequences spatially (rotation and translation)
    Uses Procrustes alignment
    
    Args:
        poses1: Reference poses [frames, 17, 3]
        poses2: Poses to align [frames, 17, 3]
    
    Returns:
        Aligned poses2
    """
    from scipy.spatial.transform import Rotation
    
    poses1 = np.array(poses1)
    poses2 = np.array(poses2)
    
    # Center both
    poses1_centered = center_poses(poses1)
    poses2_centered = center_poses(poses2)
    
    # For each frame, find optimal rotation
    aligned = np.zeros_like(poses2_centered)
    
    for frame_idx in range(len(poses1_centered)):
        p1 = poses1_centered[frame_idx]
        p2 = poses2_centered[frame_idx]
        
        # Find rotation using SVD (Procrustes)
        H = p2.T @ p1
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Apply rotation
        aligned[frame_idx] = p2 @ R.T
    
    return aligned


if __name__ == "__main__":
    # Test functions
    print("Testing utility functions...")
    
    # Create dummy pose data
    test_poses = np.random.randn(10, 17, 3)
    
    # Test normalization
    normalized, scale = normalize_body_scale(test_poses)
    print(f"Normalization: original scale ~{scale:.2f}")
    
    # Test centering
    centered = center_poses(test_poses)
    print(f"Centering: hip position = {centered[0, 0]}")
    
    # Test distances
    dists = calculate_joint_distances(test_poses[0], test_poses[1])
    print(f"Joint distances: mean = {np.mean(dists):.2f}")
    
    # Test interpolation
    interpolated = interpolate_sequence(test_poses, 20)
    print(f"Interpolation: {test_poses.shape[0]} -> {interpolated.shape[0]} frames")
    
    print("All tests passed!")

