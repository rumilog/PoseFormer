"""
Pose-invariant comparison focusing on joint angles and relative positions
Rather than absolute 3D coordinates in space
"""

import numpy as np
from scipy.spatial.transform import Rotation


def align_to_canonical_pose(poses):
    """
    Align poses to a canonical orientation using the torso as reference
    This removes rotation/translation variations
    
    Args:
        poses: Pose sequence [frames, 17, 3]
    
    Returns:
        aligned_poses: Poses in canonical orientation [frames, 17, 3]
    """
    poses = np.array(poses)
    aligned_poses = np.zeros_like(poses)
    
    # Joint indices
    # 0: pelvis, 1: R_hip, 2: R_knee, 3: R_ankle
    # 4: L_hip, 5: L_knee, 6: L_ankle
    # 7: spine, 8: thorax, 9: neck, 10: head
    # 11: L_shoulder, 12: L_elbow, 13: L_wrist
    # 14: R_shoulder, 15: R_elbow, 16: R_wrist
    
    for frame_idx in range(len(poses)):
        pose = poses[frame_idx].copy()
        
        # Center at pelvis (joint 0)
        pelvis = pose[0]
        pose_centered = pose - pelvis
        
        # Define torso orientation using pelvis (0) and thorax (8)
        pelvis_pt = pose_centered[0]  # Should be [0,0,0] now
        thorax_pt = pose_centered[8]
        
        # Create coordinate system from torso
        # Y-axis: pelvis to thorax (up direction)
        y_axis = thorax_pt - pelvis_pt
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)
        
        # X-axis: perpendicular to torso in frontal plane
        # Use shoulders to determine front direction
        left_shoulder = pose_centered[11]
        right_shoulder = pose_centered[14]
        shoulder_vec = left_shoulder - right_shoulder
        
        # Z-axis: perpendicular to both (forward direction)
        z_axis = np.cross(shoulder_vec, y_axis)
        z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)
        
        # Recompute X-axis to ensure orthogonality
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
        
        # Rotation matrix from pose frame to canonical frame
        rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
        
        # Apply rotation to all joints
        pose_aligned = pose_centered @ rotation_matrix
        
        aligned_poses[frame_idx] = pose_aligned
    
    return aligned_poses


def compute_joint_angles(poses):
    """
    Compute joint angles from 3D poses
    
    Args:
        poses: Pose sequence [frames, 17, 3]
    
    Returns:
        angles: Joint angles [frames, n_angles]
    """
    poses = np.array(poses)
    n_frames = len(poses)
    
    # Define bone connections for angle calculation
    # Format: (parent_joint, child_joint_1, child_joint_2)
    angle_triplets = [
        # Right leg angles
        (0, 1, 2),   # Hip-knee angle (right)
        (1, 2, 3),   # Knee-ankle angle (right)
        
        # Left leg angles
        (0, 4, 5),   # Hip-knee angle (left)
        (4, 5, 6),   # Knee-ankle angle (left)
        
        # Spine angles
        (0, 7, 8),   # Pelvis-spine-thorax
        (7, 8, 9),   # Spine-thorax-neck
        (8, 9, 10),  # Thorax-neck-head
        
        # Right arm angles
        (8, 14, 15), # Thorax-shoulder-elbow (right)
        (14, 15, 16),# Shoulder-elbow-wrist (right)
        
        # Left arm angles
        (8, 11, 12), # Thorax-shoulder-elbow (left)
        (11, 12, 13),# Shoulder-elbow-wrist (left)
    ]
    
    angles = np.zeros((n_frames, len(angle_triplets)))
    
    for frame_idx in range(n_frames):
        pose = poses[frame_idx]
        
        for angle_idx, (j0, j1, j2) in enumerate(angle_triplets):
            # Vectors
            v1 = pose[j1] - pose[j0]
            v2 = pose[j2] - pose[j1]
            
            # Normalize
            v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
            v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
            
            # Angle between vectors
            cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            angles[frame_idx, angle_idx] = angle
    
    return angles


def compute_relative_distances(poses):
    """
    Compute relative distances between key joint pairs
    (bone lengths, normalized by body scale)
    
    Args:
        poses: Pose sequence [frames, 17, 3]
    
    Returns:
        distances: Relative distances [frames, n_distances]
    """
    poses = np.array(poses)
    n_frames = len(poses)
    
    # Define important bone connections
    bone_pairs = [
        # Limbs
        (1, 2),   # Right thigh
        (2, 3),   # Right shin
        (4, 5),   # Left thigh
        (5, 6),   # Left shin
        (14, 15), # Right upper arm
        (15, 16), # Right forearm
        (11, 12), # Left upper arm
        (12, 13), # Left forearm
        
        # Torso
        (0, 8),   # Pelvis to thorax
        (8, 9),   # Thorax to neck
        (9, 10),  # Neck to head
        
        # Shoulder width
        (11, 14), # Left to right shoulder
        
        # Hip width
        (1, 4),   # Left to right hip
    ]
    
    distances = np.zeros((n_frames, len(bone_pairs)))
    
    for frame_idx in range(n_frames):
        pose = poses[frame_idx]
        
        for dist_idx, (j1, j2) in enumerate(bone_pairs):
            dist = np.linalg.norm(pose[j2] - pose[j1])
            distances[frame_idx, dist_idx] = dist
    
    return distances


def pose_invariant_comparison(user_poses, ref_poses):
    """
    Compare poses in an orientation-invariant way
    Focuses on joint angles and relative bone lengths
    
    Args:
        user_poses: User pose sequence [frames, 17, 3]
        ref_poses: Reference pose sequence [frames, 17, 3]
    
    Returns:
        angle_similarity: Similarity score based on joint angles (0-100)
        distance_similarity: Similarity score based on bone lengths (0-100)
        combined_score: Combined similarity score (0-100)
    """
    # Align both sequences to canonical orientation
    user_aligned = align_to_canonical_pose(user_poses)
    ref_aligned = align_to_canonical_pose(ref_poses)
    
    # Compute joint angles
    user_angles = compute_joint_angles(user_aligned)
    ref_angles = compute_joint_angles(ref_aligned)
    
    # Compute relative distances
    user_distances = compute_relative_distances(user_aligned)
    ref_distances = compute_relative_distances(ref_aligned)
    
    # Normalize distances by body scale
    user_scale = np.mean(user_distances)
    ref_scale = np.mean(ref_distances)
    user_distances_norm = user_distances / (user_scale + 1e-8)
    ref_distances_norm = ref_distances / (ref_scale + 1e-8)
    
    # Compare angles (in radians)
    angle_diff = np.abs(user_angles - ref_angles)
    # Convert to degrees for interpretability
    angle_diff_deg = np.rad2deg(angle_diff)
    
    # Score: percentage of angles within tolerance (15 degrees)
    angle_tolerance_deg = 15.0
    within_angle_tolerance = angle_diff_deg < angle_tolerance_deg
    angle_similarity = float(np.mean(within_angle_tolerance) * 100)
    
    # Compare relative distances
    distance_diff = np.abs(user_distances_norm - ref_distances_norm)
    
    # Score: percentage of distances within tolerance (10% of normalized value)
    distance_tolerance = 0.10
    within_distance_tolerance = distance_diff < distance_tolerance
    distance_similarity = float(np.mean(within_distance_tolerance) * 100)
    
    # Combined score (weighted average)
    # Angles are more important for form
    combined_score = 0.7 * angle_similarity + 0.3 * distance_similarity
    
    return angle_similarity, distance_similarity, combined_score


def pose_invariant_score_by_body_part(user_poses, ref_poses, body_part_groups):
    """
    Compute pose-invariant scores for each body part
    
    Args:
        user_poses: User pose sequence [frames, 17, 3]
        ref_poses: Reference pose sequence [frames, 17, 3]
        body_part_groups: Dict mapping body part names to joint indices
    
    Returns:
        body_part_scores: Dict of scores per body part
    """
    # Align poses
    user_aligned = align_to_canonical_pose(user_poses)
    ref_aligned = align_to_canonical_pose(ref_poses)
    
    # Compute angles
    user_angles = compute_joint_angles(user_aligned)
    ref_angles = compute_joint_angles(ref_aligned)
    
    # Angle triplet to joint mapping
    # (which joints are involved in each angle)
    angle_to_joints = [
        [0, 1, 2],   # Right hip-knee
        [1, 2, 3],   # Right knee-ankle
        [0, 4, 5],   # Left hip-knee
        [4, 5, 6],   # Left knee-ankle
        [0, 7, 8],   # Pelvis-spine-thorax
        [7, 8, 9],   # Spine-thorax-neck
        [8, 9, 10],  # Thorax-neck-head
        [8, 14, 15], # Right thorax-shoulder-elbow
        [14, 15, 16],# Right shoulder-elbow-wrist
        [8, 11, 12], # Left thorax-shoulder-elbow
        [11, 12, 13],# Left shoulder-elbow-wrist
    ]
    
    body_part_scores = {}
    
    for part_name, joint_indices in body_part_groups.items():
        # Find angles that involve joints from this body part
        relevant_angle_indices = []
        for angle_idx, angle_joints in enumerate(angle_to_joints):
            # If any joint in the angle belongs to this body part
            if any(j in joint_indices for j in angle_joints):
                relevant_angle_indices.append(angle_idx)
        
        if relevant_angle_indices:
            # Score based on relevant angles
            angle_diff = np.abs(user_angles[:, relevant_angle_indices] - ref_angles[:, relevant_angle_indices])
            angle_diff_deg = np.rad2deg(angle_diff)
            
            within_tolerance = angle_diff_deg < 15.0  # 15 degree tolerance
            score = float(np.mean(within_tolerance) * 100)
            body_part_scores[part_name] = score
        else:
            body_part_scores[part_name] = 100.0  # No relevant angles
    
    return body_part_scores


if __name__ == "__main__":
    # Test pose-invariant comparison
    print("Testing pose-invariant comparison...")
    
    # Create test poses
    test_poses1 = np.random.randn(100, 17, 3)
    test_poses2 = test_poses1 + np.random.randn(100, 17, 3) * 0.1
    
    # Test alignment
    aligned = align_to_canonical_pose(test_poses1)
    print(f"Aligned poses shape: {aligned.shape}")
    
    # Test angles
    angles = compute_joint_angles(test_poses1)
    print(f"Joint angles shape: {angles.shape}")
    
    # Test comparison
    angle_sim, dist_sim, combined = pose_invariant_comparison(test_poses1, test_poses2)
    print(f"Angle similarity: {angle_sim:.2f}")
    print(f"Distance similarity: {dist_sim:.2f}")
    print(f"Combined score: {combined:.2f}")
    
    print("Tests passed!")










