# Submission Guide - PoseFormer Fitness Coaching System

This guide helps you identify which files to include in your project submission.

## ✅ Files to INCLUDE

### 1. Core Source Code
- **`fitness_coach/`** (entire directory)
  - All Python files: `__init__.py`, `body_parts.py`, `comparison.py`, `noise_scoring.py`, `reference_processor.py`, `temporal_align.py`, `user_processor.py`, `utils.py`, `video_comparison.py`, `video_from_images.py`
  - `README.md` (if exists)

- **`demo/`** (source code only)
  - `vis.py`
  - `process_reference.py`
  - `lib/` (entire directory - contains HRNet, YOLOv3 code)
  - **EXCLUDE**: `demo/output/`, `demo/video/` (large video files)

- **`common/`** (entire directory)
  - All Python files and subdirectories

- **`mpi_inf_3dhp/`** (entire directory)
  - All source code files

- **Root level Python files**
  - `run_poseformer.py`

### 2. Configuration & Dependencies
- **`requirements.txt`** - Python dependencies
- **`.gitignore`** - Git ignore rules

### 3. Documentation
- **`README.md`** - Original PoseFormerV2 README
- **`README_FITNESS.md`** - Fitness coaching system documentation
- **`GITHUB_SETUP.md`** - GitHub setup instructions
- **`VIDEO_COMPARISON_GUIDE.md`** - Video comparison guide
- **`LICENSE`** - License file

### 4. Example/Demo Files (Optional - if small)
- **`images/`** - Demo GIFs and images (if not too large)
- **`demo/video/sample_video.mp4`** - Only if it's a small example file

## ❌ Files to EXCLUDE

### 1. Large Model Files (too big for submission)
- `demo/lib/checkpoint/*.bin` (model weights)
- `demo/lib/checkpoint/*.pth` (HRNet weights)
- `demo/lib/checkpoint/*.weights` (YOLOv3 weights)
- `checkpoint/` directory (if exists)

### 2. Cache & Generated Outputs
- `user_videos_cache/` - User video processing cache
- `references/` - Processed reference data
- `demo/output/` - Generated pose visualizations
- `temp_user_processing/` - Temporary processing files
- `temp_processing/` - Any temporary directories

### 3. Large Video Files
- `demo/video/*.mp4` - Original video files (unless small examples)
- `comparison_*.mp4` - Generated comparison videos
- `comparison_*.gif` - Generated comparison GIFs

### 4. Python Cache
- `__pycache__/` directories (all of them)
- `*.pyc` files

### 5. Test Scripts (Optional)
- `test_perfect_score.cmd`
- `generate_comparison_video.cmd`

## 📦 Recommended Submission Structure

```
PoseFormer/
├── fitness_coach/          # Your fitness coaching system
├── demo/                    # Pose estimation code (no output/, no video/)
├── common/                  # Shared utilities
├── mpi_inf_3dhp/           # 3DHP dataset code
├── images/                  # Demo images (if small)
├── requirements.txt
├── .gitignore
├── README.md
├── README_FITNESS.md        # Main documentation for your work
├── LICENSE
└── run_poseformer.py
```

## 🚀 Quick Checklist

Before submitting, verify:

- [ ] All Python source code is included
- [ ] `requirements.txt` is included
- [ ] Documentation files (README_FITNESS.md) are included
- [ ] No model weight files (*.bin, *.pth, *.weights)
- [ ] No cache directories (user_videos_cache/, references/)
- [ ] No large video files
- [ ] No __pycache__ directories
- [ ] No generated output files (comparison videos, gifs)
- [ ] Total submission size is reasonable (< 100MB recommended)

## 📝 Notes for Submission

1. **Model Weights**: Mention in your README that users need to download model weights separately (provide links to original PoseFormerV2 repository)

2. **Example Videos**: If you want to include example videos, use small, compressed samples (< 10MB each)

3. **Dependencies**: Make sure `requirements.txt` is complete and accurate

4. **Documentation**: Your `README_FITNESS.md` is the main documentation - ensure it's clear and complete

