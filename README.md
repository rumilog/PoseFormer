# AI Fitness Coach - PoseFormerV2

An AI-powered fitness coaching application that analyzes exercise form using 3D pose estimation. Upload a video of yourself performing an exercise, and get detailed technical feedback plus personalized coaching from AI personas.

## Features

- **3D Pose Estimation**: Uses PoseFormerV2 to extract accurate 3D human poses from video
- **Exercise Comparison**: Compares your form against reference "gold standard" poses
- **AI Coach Personas**: Get feedback in different styles:
  - 🔥 **Hype Beast** - Energetic, motivational feedback
  - 📊 **Data Scientist** - Technical, metrics-focused analysis
  - 💪 **No-Nonsense Pro** - Direct, efficient coaching
  - 🧘 **Mindful Aligner** - Form-focused, mindful approach

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/rumilog/PoseFormer.git
cd PoseFormer
```

### 2. Set Up Environment

```bash
conda create -n poseformerv2 python=3.9
conda activate poseformerv2
pip install -r requirements.txt
```

### 3. Download Model Weights (Required!)

You must download these model weights and place them in `demo/lib/checkpoint/`:

| Model | File | Download Link |
|-------|------|---------------|
| YOLOv3 | `yolov3.weights` | [Download](https://pjreddie.com/media/files/yolov3.weights) |
| HRNet | `pose_hrnet_w48_384x288.pth` | [Download](https://drive.google.com/file/d/1UoJhTtjHNByZSm96W3yFTfU5upJnsKiS/view) |
| PoseFormerV2 | `27_243_45.2.bin` | [Download](https://drive.google.com/file/d/14SpqPyq9yiblCzTH5CorymKCUsXapmkg/view) |

After downloading, your folder structure should look like:
```
demo/lib/checkpoint/
├── yolov3.weights
├── pose_hrnet_w48_384x288.pth
└── 27_243_45.2.bin
```

### 4. Run the Application

```bash
python fitness_backend_server.py
```

Or use the batch file (Windows):
```bash
run_full_app.bat
```

The server will start at `http://localhost:8000`. Open the frontend HTML file in your browser to use the application.

## How It Works

1. **Video Upload**: User uploads a video of themselves doing an exercise
2. **2D Pose Detection**: YOLOv3 detects the person, HRNet extracts 2D keypoints
3. **3D Pose Lifting**: PoseFormerV2 converts 2D poses to 3D
4. **Comparison**: User's poses are aligned and compared to reference poses
5. **Scoring**: Multiple metrics are computed (joint angles, timing, overall form)
6. **AI Feedback**: Fine-tuned persona models generate personalized coaching

## Project Structure

```
PoseFormer/
├── fitness_backend_server.py   # Main backend server
├── run_full_app.bat            # Windows launcher
├── requirements.txt            # Python dependencies
│
├── fitness_coach/              # Core fitness analysis modules
│   ├── comparison.py           # Main scoring logic
│   ├── reference_processor.py  # Process reference videos
│   ├── user_processor.py       # Process user videos
│   ├── temporal_align.py       # DTW alignment
│   └── noise_scoring.py        # Noise-robust scoring
│
├── demo/                       # Pose estimation pipeline
│   ├── vis.py                  # Main pose extraction
│   ├── video/                  # Reference videos
│   └── lib/
│       ├── checkpoint/         # Model weights (download required)
│       ├── hrnet/              # HRNet 2D pose detector
│       ├── yolov3/             # Person detector
│       └── sort/               # Object tracking
│
├── common/                     # PoseFormerV2 model code
│   └── model_poseformer.py     # 3D pose lifting model
│
└── references/                 # Reference exercise data
    └── pushup/
        └── metadata.json       # Exercise metadata
```

## First Run - Reference Processing

On first run, the application will automatically process the reference video in `demo/video/` to generate the reference pose data. This may take a few minutes. The processed data will be saved to `references/pushup/` for future use.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/analyze` | POST | Analyze uploaded exercise video |
| `/api/health` | GET | Health check |

## For Developers

### Adding New Exercises

1. Add a reference video to `demo/video/`
2. Create a metadata file in `references/<exercise_name>/metadata.json`
3. The reference will be auto-processed on first use

### Fine-tuning Persona Models

The persona models are hosted on HuggingFace and loaded automatically:
- `rlogh/fitness-coach-persona-hype-beast`
- `rlogh/fitness-coach-persona-data-scientist`
- `rlogh/fitness-coach-persona-no-nonsense-pro`
- `rlogh/fitness-coach-persona-mindful-aligner`

## Requirements

- Python 3.9+
- CUDA-compatible GPU (recommended for real-time processing)
- ~4GB disk space for model weights

## Citation

This project builds on PoseFormerV2:

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

MIT License
