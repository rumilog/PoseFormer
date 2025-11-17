# Fitness Coach Scoring System

A comprehensive scoring system for comparing user exercise performance to reference videos using 3D pose estimation.

## Overview

This module provides tools to:
- Process reference videos and generate noisy samples for scoring
- Process user videos and extract 3D poses
- Compare user performance to reference using noise-based scoring
- Generate per-body-part scores and overall performance metrics

## Module Structure

### Core Modules (Phase 1 - COMPLETE ✓)

1. **`body_parts.py`** - Body part groupings and joint metadata
   - Defines 17-joint skeleton structure
   - Maps joints to body part groups (arms, legs, core, etc.)
   - Provides noise levels per joint type
   - Exercise-specific body part focus

2. **`utils.py`** - Utility functions
   - Pose normalization and centering
   - Joint distance calculations
   - Sequence interpolation
   - Spatial alignment

3. **`temporal_align.py`** - Temporal alignment
   - Dynamic Time Warping (DTW) for sequence alignment
   - Handles sequences of different lengths
   - Phase alignment for motion comparison

4. **`noise_scoring.py`** - Noise-based scoring
   - Generates noisy reference samples
   - Statistical bounds calculation
   - Per-body-part scoring
   - Overall performance scoring

## Usage Example

```python
from fitness_coach.noise_scoring import score_with_statistical_bounds
import numpy as np

# Load poses (from processed videos)
user_poses = np.load('user_keypoints_3D.npz')['reconstruction']
ref_poses = np.load('reference_keypoints_3D.npz')['reconstruction']

# Score the user's performance
scores = score_with_statistical_bounds(user_poses, ref_poses)

print(f"Overall Score: {scores['overall_score']:.2f}")
print(f"Body Part Scores: {scores['body_part_scores']}")
```

## Testing

Run the test suite:
```bash
python fitness_coach/test_modules.py
```

All core modules have been tested and verified working.

## Next Steps (Phase 2)

1. **Reference Processor** - Process reference videos once
2. **User Processor** - Process user videos
3. **Comparison Module** - Full comparison pipeline
4. **API Integration** - REST API endpoints

## Dependencies

- numpy
- scipy
- fastdtw (optional, for better temporal alignment)

Install with:
```bash
pip install fastdtw
```

## Status

✅ Phase 1: Core Infrastructure - **COMPLETE**
- [x] Body parts module
- [x] Utils module  
- [x] Temporal alignment module
- [x] Noise scoring module
- [x] All tests passing

🔄 Phase 2: Processing Pipelines - **IN PROGRESS**
- [ ] Reference processor
- [ ] User processor
- [ ] Integration with existing vis.py

⏳ Phase 3: Comparison & API - **PENDING**
- [ ] Full comparison module
- [ ] API endpoints
- [ ] Documentation

