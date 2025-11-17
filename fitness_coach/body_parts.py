"""
Body Part Groupings and Joint Metadata
Defines how 17-joint skeleton maps to body part groups for scoring
"""

import numpy as np

# Joint indices for 17-joint Human3.6M format
# 0: Hip, 1-3: Right leg, 4-6: Left leg, 7-10: Spine/Head, 11-13: Left arm, 14-16: Right arm
JOINT_NAMES = [
    'Hip',              # 0
    'RightHip',         # 1
    'RightKnee',        # 2
    'RightAnkle',       # 3
    'LeftHip',          # 4
    'LeftKnee',         # 5
    'LeftAnkle',        # 6
    'Spine',            # 7
    'Thorax',           # 8
    'Neck',             # 9
    'Head',             # 10
    'LeftShoulder',     # 11
    'LeftElbow',        # 12
    'LeftWrist',        # 13
    'RightShoulder',    # 14
    'RightElbow',       # 15
    'RightWrist',       # 16
]

# Body part groupings for scoring
JOINT_GROUPS = {
    'right_arm': [14, 15, 16],      # Right shoulder, elbow, wrist
    'left_arm': [11, 12, 13],       # Left shoulder, elbow, wrist
    'right_leg': [1, 2, 3],          # Right hip, knee, ankle
    'left_leg': [4, 5, 6],           # Left hip, knee, ankle
    'torso': [0, 7, 8, 9, 10],      # Hip, spine, thorax, neck, head
    'core': [0, 7, 8],              # Hip, spine, thorax (for core exercises like push-ups)
    'upper_body': [7, 8, 9, 10, 11, 12, 13, 14, 15, 16],  # Everything above hip
    'lower_body': [0, 1, 2, 3, 4, 5, 6],  # Everything below and including hip
}

# Noise levels per joint type (as fraction of body scale)
# Different joints have different acceptable variation
JOINT_NOISE_LEVELS = {
    'core': 0.02,           # Hip, spine - very tight tolerance
    'shoulders': 0.04,      # Shoulder joints
    'elbows': 0.06,         # Elbow, knee
    'wrists': 0.08,         # Wrist, ankle
    'hands': 0.10,          # Hands, feet - most variation
}

# Map each joint to its noise level category
JOINT_TO_NOISE_CATEGORY = {
    0: 'core',      # Hip
    1: 'shoulders', # Right hip (treated as shoulder-like for movement)
    2: 'elbows',    # Right knee
    3: 'wrists',    # Right ankle
    4: 'shoulders', # Left hip
    5: 'elbows',    # Left knee
    6: 'wrists',    # Left ankle
    7: 'core',      # Spine
    8: 'core',      # Thorax
    9: 'shoulders', # Neck
    10: 'shoulders', # Head
    11: 'shoulders', # Left shoulder
    12: 'elbows',    # Left elbow
    13: 'wrists',    # Left wrist
    14: 'shoulders', # Right shoulder
    15: 'elbows',    # Right elbow
    16: 'wrists',    # Right wrist
}

# Joint pairs for calculating angles (parent-child relationships)
JOINT_PAIRS = [
    (0, 1),   # Hip -> Right Hip
    (1, 2),   # Right Hip -> Right Knee
    (2, 3),   # Right Knee -> Right Ankle
    (0, 4),   # Hip -> Left Hip
    (4, 5),   # Left Hip -> Left Knee
    (5, 6),   # Left Knee -> Left Ankle
    (0, 7),   # Hip -> Spine
    (7, 8),   # Spine -> Thorax
    (8, 9),   # Thorax -> Neck
    (9, 10),  # Neck -> Head
    (8, 11),  # Thorax -> Left Shoulder
    (11, 12), # Left Shoulder -> Left Elbow
    (12, 13), # Left Elbow -> Left Wrist
    (8, 14),  # Thorax -> Right Shoulder
    (14, 15), # Right Shoulder -> Right Elbow
    (15, 16), # Right Elbow -> Right Wrist
]


def get_body_part_joints(part_name):
    """
    Get joint indices for a body part group
    
    Args:
        part_name: Name of body part (e.g., 'right_arm', 'core')
    
    Returns:
        List of joint indices
    """
    if part_name not in JOINT_GROUPS:
        raise ValueError(f"Unknown body part: {part_name}. Available: {list(JOINT_GROUPS.keys())}")
    return JOINT_GROUPS[part_name]


def get_joint_noise_level(joint_idx):
    """
    Get noise level for a specific joint
    
    Args:
        joint_idx: Joint index (0-16)
    
    Returns:
        Noise level (float) as fraction of body scale
    """
    if joint_idx not in JOINT_TO_NOISE_CATEGORY:
        return 0.05  # Default
    category = JOINT_TO_NOISE_CATEGORY[joint_idx]
    return JOINT_NOISE_LEVELS[category]


def get_all_body_parts():
    """
    Get all available body part names
    
    Returns:
        List of body part names
    """
    return list(JOINT_GROUPS.keys())


def get_joint_name(joint_idx):
    """
    Get human-readable name for a joint
    
    Args:
        joint_idx: Joint index (0-16)
    
    Returns:
        Joint name string
    """
    if 0 <= joint_idx < len(JOINT_NAMES):
        return JOINT_NAMES[joint_idx]
    return f"Joint_{joint_idx}"


def get_joints_for_exercise(exercise_type):
    """
    Get relevant body parts for a specific exercise type
    
    Args:
        exercise_type: Type of exercise (e.g., 'pushup', 'squat', 'plank')
    
    Returns:
        List of body part names relevant to the exercise
    """
    exercise_focus = {
        'pushup': ['core', 'right_arm', 'left_arm', 'torso'],
        'squat': ['core', 'right_leg', 'left_leg', 'torso'],
        'plank': ['core', 'torso', 'right_arm', 'left_arm'],
        'lunge': ['core', 'right_leg', 'left_leg', 'torso'],
        'all': list(JOINT_GROUPS.keys()),
    }
    
    return exercise_focus.get(exercise_type.lower(), exercise_focus['all'])


def calculate_body_scale(poses):
    """
    Calculate body scale (hip-to-shoulder distance) for normalization
    
    Args:
        poses: Array of shape [frames, 17, 3] or [17, 3]
    
    Returns:
        Average body scale (float)
    """
    poses = np.array(poses)
    if len(poses.shape) == 2:
        poses = poses[np.newaxis, :, :]
    
    # Hip (0) to Thorax (8) distance
    hip_to_thorax = np.linalg.norm(poses[:, 0, :] - poses[:, 8, :], axis=1)
    return np.mean(hip_to_thorax)


if __name__ == "__main__":
    # Test the module
    print("Body Part Groups:")
    for part, joints in JOINT_GROUPS.items():
        joint_names = [JOINT_NAMES[j] for j in joints]
        print(f"  {part}: {joints} - {joint_names}")
    
    print("\nJoint Noise Levels:")
    for i in range(17):
        print(f"  {JOINT_NAMES[i]}: {get_joint_noise_level(i)}")
    
    print("\nExercise Focus (Push-up):")
    print(f"  {get_joints_for_exercise('pushup')}")

