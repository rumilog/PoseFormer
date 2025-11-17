# PoseFormer - Fitness Coaching API

A fitness coaching system that uses 3D human pose estimation to analyze exercise form and provide real-time feedback. Built on top of [PoseFormerV2](https://github.com/QitaoZhao/PoseFormerV2).

## Features

- 🎯 **Exercise Form Analysis**: Compare user exercise videos against reference videos
- 📊 **Detailed Scoring**: Per-body-part scoring with statistical analysis
- 🎬 **Visual Comparison**: Side-by-side 3D pose comparison videos
- 🤖 **API-Ready**: JSON output format for LLM integration
- ⚡ **GPU-Accelerated**: Full CUDA support for fast processing
- 💾 **Smart Caching**: Automatic caching to avoid reprocessing

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n poseformerv2 python=3.9
conda activate poseformerv2

# Install PyTorch with CUDA (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Pre-trained Models

Download the PoseFormerV2 model weights and place them in `./checkpoint/`:
- Model: `27_243_45.2.bin` (or other variants from [PoseFormerV2](https://github.com/QitaoZhao/PoseFormerV2))

Download HRNet and YOLOv3 weights for 2D pose detection:
- HRNet: Place in `./demo/lib/checkpoint/`
- YOLOv3: Place in `./demo/lib/checkpoint/`

### 3. Process Reference Video

```bash
python demo/process_reference.py --video "path/to/reference_video.mp4" --exercise pushup --samples 50
```

This creates a reference dataset with noisy samples for robust scoring.

### 4. Score User Exercise

```bash
python -m fitness_coach.comparison --user-video "path/to/user_video.mp4" --reference pushup --json
```

### 5. Generate Comparison Video

```bash
python -m fitness_coach.comparison --user-video "path/to/user_video.mp4" --reference pushup --generate-video
```

## Usage Examples

### Basic Scoring

```bash
python -m fitness_coach.comparison \
    --user-video "demo/video/user.mp4" \
    --reference pushup \
    --json \
    --output scores.json
```

### With Comparison Video

```bash
python -m fitness_coach.comparison \
    --user-video "demo/video/user.mp4" \
    --reference pushup \
    --json \
    --generate-video \
    --video-output comparison.mp4
```

### Perfect Match Test

Compare reference against itself (should score 100%):

```bash
python -m fitness_coach.comparison \
    --user-video "demo/video/reference_video.mp4" \
    --reference pushup \
    --json
```

## API Output Format

The JSON output is designed for easy integration with LLMs and other APIs:

```json
{
  "status": "success",
  "exercise": {
    "type": "pushup",
    "reference": "pushup",
    "user_video": "user.mp4"
  },
  "scores": {
    "overall": 85.5,
    "relevant": 90.2,
    "body_parts": {
      "Core": 95.0,
      "Right_Arm": 88.5,
      "Left_Arm": 90.0,
      "Torso": 87.2
    }
  },
  "metrics": {
    "frames": {
      "user": 300,
      "reference": 325,
      "aligned": 325
    },
    "alignment_quality": 0.1234,
    "body_part_details": {
      "Right_Arm": {
        "position_error_avg": 0.0234,
        "position_error_max": 0.0456,
        "tolerance_threshold": 0.0200,
        "in_tolerance_percentage": 85.5
      }
    }
  },
  "feedback": [
    "Good form overall!",
    "Keep your core engaged throughout the movement.",
    "Focus on maintaining consistent arm positioning."
  ]
}
```

## Project Structure

```
PoseFormerV2/
├── fitness_coach/          # Fitness coaching modules
│   ├── body_parts.py      # Body part groupings and joint definitions
│   ├── comparison.py      # Main scoring and comparison logic
│   ├── noise_scoring.py   # Statistical scoring methods
│   ├── reference_processor.py  # Reference video processing
│   ├── temporal_align.py  # DTW and temporal alignment
│   ├── user_processor.py  # User video processing with caching
│   ├── utils.py           # Pose normalization utilities
│   └── video_from_images.py  # Video generation from pose3D images
├── demo/                  # Original PoseFormerV2 demo code
├── references/            # Processed reference videos (generated)
├── user_videos_cache/     # Cached user video processing (generated)
└── requirements.txt       # Python dependencies
```

## Scoring Methodology

The system uses a multi-stage approach:

1. **Temporal Alignment**: Dynamic Time Warping (DTW) or interpolation to synchronize sequences
2. **Spatial Normalization**: Body scale normalization and centering
3. **Statistical Scoring**: Noise-based tolerance zones for each body part
4. **Exercise-Specific Analysis**: Focus on relevant body parts per exercise type

### Body Part Groupings

- **Core**: Hips, spine, core joints
- **Arms**: Shoulders, elbows, wrists (left/right)
- **Torso**: Upper body and spine
- **Legs**: Hips, knees, ankles (left/right)

## Dependencies

Key dependencies:
- PyTorch (with CUDA support)
- NumPy, SciPy
- Matplotlib (for visualization)
- OpenCV (for video processing)
- fastdtw (optional, for better temporal alignment)
- Pillow (for image processing)

See `requirements.txt` for complete list.

## Troubleshooting

### FFmpeg Not Found

For MP4 video generation, install FFmpeg:

```bash
# Conda (recommended)
conda install -c conda-forge ffmpeg

# Or winget (Windows)
winget install ffmpeg
```

The system will automatically fall back to GIF format if FFmpeg is not available.

### CUDA Issues

Ensure PyTorch is installed with CUDA support:

```python
import torch
print(torch.cuda.is_available())  # Should be True
```

### Model Loading Errors

If you see `DataParallel` key mismatches, the code automatically handles this. If issues persist, ensure you're using the correct model checkpoint.

## Credits

This project extends [PoseFormerV2](https://github.com/QitaoZhao/PoseFormerV2) by Qitao Zhao et al., which was accepted to CVPR 2023.

Original PoseFormerV2 citation:
```bibtex
@InProceedings{Zhao_2023_CVPR,
    author    = {Zhao, Qitao and Zheng, Ce and Liu, Mengyuan and Wang, Pichao and Chen, Chen},
    title     = {PoseFormerV2: Exploring Frequency Domain for Efficient and Robust 3D Human Pose Estimation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {8877-8886}
}
```

## License

Please refer to the original PoseFormerV2 repository for licensing information.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

