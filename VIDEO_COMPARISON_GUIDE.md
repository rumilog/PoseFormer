# 3D Pose Comparison Video Guide

This guide explains how to generate side-by-side comparison videos showing your exercise form next to the reference (correct) form.

## Features

- **Side-by-side view**: Your 3D pose and the reference pose displayed simultaneously
- **Frame synchronized**: After temporal alignment using DTW
- **Customizable**: Adjust FPS, camera angles, and output location
- **Color-coded**: Reference in green, user in red for easy comparison

## Requirements

- matplotlib (already in requirements.txt)
- FFmpeg installed and in your system PATH
  - Download from: https://ffmpeg.org/download.html
  - Windows users: Add FFmpeg `bin` folder to PATH environment variable

## Basic Usage

### Option 1: Using the comparison script directly

```powershell
conda activate poseformerv2
python -m fitness_coach.comparison --user-video "demo/video/user.mp4" --reference pushup --generate-video
```

This will:
1. Score your exercise
2. Generate a comparison video named `comparison_user.mp4`
3. Show scoring results in the terminal

### Option 2: Using the batch script

Simply run:
```powershell
generate_comparison_video.cmd
```

### Option 3: Standalone video generation

If you already have processed poses, you can generate just the video:

```powershell
python -m fitness_coach.video_comparison --user-poses "user_videos_cache/user/keypoints_3D.npz" --reference-poses "references/pushup/keypoints_3D.npz" --output "my_comparison.mp4"
```

## Advanced Options

### Custom Output Path
```powershell
python -m fitness_coach.comparison --user-video "demo/video/user.mp4" --reference pushup --generate-video --video-output "my_custom_name.mp4"
```

### Adjust Frame Rate
```powershell
python -m fitness_coach.comparison --user-video "demo/video/user.mp4" --reference pushup --generate-video --video-fps 60
```

### Custom Camera Angles
Using the standalone video generator:
```powershell
python -m fitness_coach.video_comparison --user-poses "user.npz" --reference-poses "ref.npz" --elev 20 --azim 45
```

- `--elev`: Elevation angle (degrees) - vertical viewing angle
- `--azim`: Azimuth angle (degrees) - horizontal rotation

### Combined with JSON Output
Get both scoring data and comparison video:
```powershell
python -m fitness_coach.comparison --user-video "demo/video/user.mp4" --reference pushup --json --output scores.json --generate-video
```

## Video Output

The generated video will show:
- **Left side**: Reference (correct) form in GREEN
- **Right side**: Your form in RED
- **Labels**: Frame counter and video names
- **3D skeleton**: Joints connected showing body pose from all angles

## Troubleshooting

### "FFmpeg not found" error
1. Download FFmpeg from https://ffmpeg.org/download.html
2. Extract to a folder (e.g., `C:\ffmpeg`)
3. Add `C:\ffmpeg\bin` to your system PATH
4. Restart your terminal/PowerShell

### Video generation is slow
- The video generation process can take 2-5 minutes depending on video length
- Progress updates are shown every 30 frames
- Reduce FPS with `--video-fps 15` for faster generation

### "Object of type float32 is not JSON serializable"
This has been fixed. Make sure you're using the latest version of `fitness_coach/comparison.py`.

## Examples

### Compare your pushup form
```powershell
python -m fitness_coach.comparison --user-video "my_pushup.mp4" --reference pushup --generate-video
```

### Perfect match test (should score 100%)
```powershell
python -m fitness_coach.comparison --user-video "demo/video/tester.mp4" --reference pushup --generate-video --video-output "perfect_match.mp4"
```

### API-ready output with video
```powershell
python -m fitness_coach.comparison --user-video "user.mp4" --reference pushup --json --generate-video
```

