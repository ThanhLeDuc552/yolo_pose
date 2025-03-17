# YOLO Pose Estimation

This repository provides real-time pose estimation using YOLOv8 Pose models. The project includes scripts for both **real-time inference via webcam** and **video-based inference**.

## Features
- **Real-time Pose Estimation** with `inference.py`
- **Video File Pose Estimation** with `inference_vid.py`
- **Skeleton Overlay Visualization**

## Requirements
Ensure you have the following dependencies installed:

```bash
pip install ultralytics opencv-python numpy torch
```

## Usage

### 1. **Real-Time Inference (Webcam)**
Run the following command to start real-time pose estimation using your webcam:

```bash
python inference.py --model path/to/model --cam_idx 0 (1, 2, 3,...)
```

### 2. **Video File Inference**
To perform pose estimation on a video file:

```bash
python inference_vid.py --video path/to/video.mp4 --model path/to/model
```

## Model
This project uses YOLO Pose models from the Ultralytics library. By default, `yolo11n-pose.pt` is used, but you can replace it with another model:

```python
model = YOLO("yolo11n-pose.pt") 
```

## Keypoint Mapping
The keypoint indices correspond to:
- **0**: Nose
- **1, 2**: Eyes (Left, Right)
- **3, 4**: Ears (Left, Right)
- **5, 6**: Shoulders (Left, Right)
- **7, 8**: Elbows (Left, Right)
- **9, 10**: Wrists (Left, Right)
- **11, 12**: Hips (Left, Right)
- **13, 14**: Knees (Left, Right)
- **15, 16**: Ankles (Left, Right)
