# AI-Powered Behaviour Detection System

A complete, production-ready Python project for real-time object detection and behavior analysis using YOLOv8 and custom tracking/rules engines.

## Overview

This project provides three integrated systems:

1. **Phase 1: YOLOv8 Object Detection** - Real-time person detection on images, videos, and webcam streams
2. **Phase 2: Behaviour Detection** - AI-powered behavior analysis detecting running, loitering, and falls using tracking and rule-based heuristics
3. **Phase 3: Violence Classification** - Deep learning-based violence detection using YOLOv8 classification model (97.7% accuracy)

## Features

### Phase 1 - Object Detection
- Real-time detection using YOLOv8 (nano model)
- Support for multiple input sources:
  - Images (JPG, PNG, BMP, etc.)
  - Videos (MP4, AVI, MOV, MKV, etc.)
  - Webcam streams
  - Folders of media
- Configurable confidence thresholds
- FPS counter with smoothed calculation
- Optional frame display and batch output saving

### Phase 2 - Behavior Detection
- **Multi-object Tracking**: IoU-based lightweight tracker maintains object identities
- **Running Detection**: Identifies when people exceed a speed threshold (150 pixels/sec by default)
- **Loitering Detection**: Tracks dwell time in defined zones with low-speed filter
- **Fall Detection**: Detects falling based on:
  - Bounding box aspect ratio change (vertical → horizontal)
  - Centroid downward motion
- **Dangerous Object Detection**: Detects knives, scissors with red warning boxes
- **Armed Person Detection**: Associates weapons with nearby persons
- **Event Logging**: Records all detected behaviors to CSV with timestamps
- **Real-time Annotation**: Visual feedback with:
  - Bounding boxes and track IDs
  - Behavior labels color-coded by type
  - Zone indicators
  - FPS counter

### Phase 3 - Violence Classification
- **Deep Learning Classification**: YOLOv8n-cls model trained on "Real Life Violence Situations" dataset
- **High Accuracy**: 97.7% top-1 accuracy on validation set
- **Real-time Inference**: ~2.7ms per frame on CPU
- **Visual Feedback**: Violence probability bar and warning banner
- **Event Logging**: Violence events logged with timestamps

## Installation

### Prerequisites
- Python 3.10 or higher
- Windows, macOS, or Linux
- Webcam (optional, for webcam mode)

### Setup (Windows PowerShell)

```powershell
# Clone or extract the repository
cd .\behavior_detection

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Setup (macOS/Linux)

```bash
cd behavior_detection

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Phase 1: Object Detection

#### Detect objects in a single image
```powershell
python yolo_object_detection/main.py --source path/to/image.jpg --conf 0.5 --show --save-dir runs/image
```

#### Detect objects in a video file
```powershell
python yolo_object_detection/main.py --source path/to/video.mp4 --conf 0.5 --show --save-dir runs/video
```

#### Real-time webcam detection
```powershell
python yolo_object_detection/main.py --source 0 --conf 0.5 --show
```

#### Batch process a folder
```powershell
python yolo_object_detection/main.py --source path/to/folder --conf 0.5 --save-dir runs/folder
```

#### Command-line Arguments for Phase 1

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--source` | str | Required | Path to image/video/folder or `0` for webcam |
| `--conf` | float | 0.5 | Confidence threshold for detections |
| `--show` | flag | False | Display annotated frames |
| `--save-dir` | str | None | Directory to save annotated outputs |

### Phase 2: Behavior Detection

#### Detect behaviors in a video
```powershell
python run_behaviour.py --source path/to/video.mp4 --show --events-csv runs/events.csv --save-dir runs/behaviour
```

#### Real-time webcam behavior analysis
```powershell
python run_behaviour.py --source 0 --show --events-csv runs/events.csv
```

#### Command-line Arguments for Phase 2

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--source` | str | Required | `0` for webcam or path to video/image |
| `--model` | str | yolov8n.pt | YOLO model (or path to custom model) |
| `--conf` | float | 0.25 | Detection confidence threshold |
| `--show` | flag | False | Display annotated frames |
| `--save-dir` | str | None | Save annotated frames to directory |
| `--events-csv` | str | None | Save event log to CSV file |
| `--debug` | flag | False | Print all detected objects |
| `--violence-model` | str | None | Path to violence classification model |
| `--violence-threshold` | float | 0.5 | Violence detection threshold |

### Phase 3: Violence Classification

Phase 3 adds deep learning-based violence detection using a YOLOv8 classification model.

#### Step 1: Prepare the Dataset

Download the "Real Life Violence Situations" dataset and place it in:
```
datasets/real_life_violence/
├── Violence/
│   ├── video1.mp4
│   └── ...
└── NonViolence/
    ├── video1.mp4
    └── ...
```

Then extract frames from videos:
```powershell
python prepare_violence_data.py
```

This creates the YOLO classification dataset structure:
```
datasets/violence_classification/
├── train/
│   ├── violence/      # ~19,000 frames
│   └── nonviolence/   # ~19,000 frames
└── val/
    ├── violence/      # ~3,000 frames
    └── nonviolence/   # ~3,000 frames
```

#### Step 2: Train the Violence Classifier

```powershell
python train_violence_cls.py --epochs 30
```

**Training Output:**
- Time: ~4 hours on CPU, ~30 minutes on GPU
- Model saved to: `runs/violence_cls/train/weights/best.pt`
- Accuracy: 97.7% top-1 accuracy

#### Step 3: Test the Violence Classifier

```powershell
python train_violence_cls.py --test runs/violence_cls/train/weights/best.pt
```

#### Step 4: Run Full System with Violence Detection

```powershell
python run_behaviour.py --source 0 --show --violence-model runs/violence_cls/train/weights/best.pt
```

This runs all three phases together:
- Object detection (YOLO)
- Behavior detection (running, loitering, falls)
- Violence classification (real-time probability)

**Note:** The system only displays threats - dangerous objects (knives, scissors) and armed persons. Normal people and other objects are hidden for cleaner visualization.

#### Violence Detection Visual Output
- **Violence probability bar** at bottom of screen
- **Red "VIOLENCE DETECTED" banner** when probability > 50%
- **Dangerous objects** shown with red boxes labeled "DANGER: KNIFE"
- **Armed persons** shown with red boxes labeled "ARMED: KNIFE"
- **Normal detections** (people, objects) are hidden
- **Events logged** to CSV with timestamps

### Example Workflows

#### Create a demo with saved video
```powershell
# Run detection on webcam for 10 seconds and save results
python run_behaviour.py --source 0 --show --save-dir runs/webcam_demo --events-csv runs/webcam_events.csv
```

#### Analyze existing video with full output
```powershell
python run_behaviour.py --source demo_video.mp4 --show --save-dir runs/analysis --events-csv runs/analysis_events.csv
```

#### Full system with violence detection
```powershell
python run_behaviour.py --source 0 --show --violence-model runs/violence_cls/train/weights/best.pt --events-csv runs/events.csv
```

#### Just get statistics (no display)
```powershell
python run_behaviour.py --source demo_video.mp4 --events-csv runs/stats.csv
```

## Project Structure

```
behavior_detection/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── run_behaviour.py                   # CLI for behavior detection (Phase 2 & 3)
│
├── yolo_object_detection/             # Phase 1: Object Detection
│   ├── __init__.py
│   ├── main.py                        # CLI entry point for detection
│   ├── detectors.py                   # YOLOv8 wrapper
│   └── utils.py                       # Drawing, FPS, argument parsing
│
├── behaviour_detection/               # Phase 2: Behavior Detection
│   ├── __init__.py
│   ├── tracker.py                     # Multi-object IoU-based tracker
│   ├── features.py                    # Motion and shape feature extraction
│   ├── rules.py                       # Behavior rules engine
│   ├── pipeline.py                    # End-to-end processing pipeline
│   └── violence_classifier.py         # Phase 3: Violence classification wrapper
│
├── prepare_violence_data.py           # Phase 3: Extract frames from violence videos
├── train_violence_cls.py              # Phase 3: Train violence classifier
├── train_custom.py                    # Train custom YOLO detection models
├── evaluate.py                        # Evaluation and testing tools
│
├── datasets/                          # Dataset directory
│   ├── real_life_violence/            # Raw violence videos (input)
│   └── violence_classification/       # Extracted frames (generated)
│
└── runs/                              # Output directory (created on first run)
    ├── image/                         # Detection outputs for images
    ├── video/                         # Detection outputs for videos
    ├── webcam/                        # Webcam detection outputs
    ├── behaviour/                     # Behavior detection outputs
    ├── violence_cls/                  # Violence classifier training outputs
    │   └── train/weights/best.pt      # Trained violence model (97.7% accuracy)
    └── events.csv                     # Event log
```

## Behavior Detection Details

### Running Detection
- **Trigger**: Average speed over 5 frames exceeds 150 pixels/second
- **Use Case**: Identifying fast-moving people
- **Output**: RED bounding box with "RUNNING" label

### Loitering Detection
- **Trigger**: Person stays in defined zone for more than 10 seconds with speed below 50 pixels/second
- **Zones**: Customizable rectangular regions of interest (default: center zone)
- **Output**: BLUE bounding box with "LOITER (zone_name)" label
- **Note**: Events logged once per loitering period

### Fall Detection
- **Trigger**: Combined heuristics:
  1. Bounding box aspect ratio drops below 40% of previous ratio (tall → short)
  2. Centroid moves downward by at least 20 pixels
- **Limitations**: Heuristic-based, works best with clear floor visibility
- **Output**: ORANGE bounding box with "FALL" label

### Configuration Thresholds

Edit `run_behaviour.py` to customize thresholds:

```python
cfg = {
    "RUN_SPEED_THRESHOLD": 150.0,          # pixels/sec
    "LOITER_TIME_THRESHOLD": 10.0,         # seconds
    "LOITER_SPEED_THRESHOLD": 50.0,        # pixels/sec
    "FALL_VERTICAL_RATIO_DROP": 0.4,       # aspect ratio threshold (0-1)
    "FALL_DOWNWARD_DISTANCE": 20.0,        # pixels
}
```

### Violence Classification

The violence classifier uses a YOLOv8n-cls model trained on 29,569 frames extracted from the "Real Life Violence Situations" dataset.

| Metric | Value |
|--------|-------|
| Model | YOLOv8n-cls (1.4M parameters) |
| Training Images | 23,683 |
| Validation Images | 5,886 |
| Top-1 Accuracy | **97.7%** |
| Inference Speed | 2.7ms per frame (CPU) |
| Training Time | ~4 hours (CPU) |

### Dangerous Object Detection

The system detects dangerous objects using COCO-trained YOLO:
- **Knife** (COCO class 43)
- **Scissors** (COCO class 76)

When a person holds a dangerous object (bounding boxes overlap), they are marked as "ARMED".

## Output Formats

### Annotated Frames
- Format: JPEG images
- Naming: `frame_XXXXXX.jpg` (sequential)
- Contents: Bounding boxes, track IDs, behavior labels, violence bar, FPS counter

### Event Log (CSV)
```csv
timestamp,type,track_id,zone_name,centroid_x,centroid_y
1702000000.123,RUN,1,,320.5,240.3
1702000001.456,LOITER,1,center,310.2,235.8
1702000002.789,FALL,2,,350.1,280.5
1702000003.000,DANGER,-1,KNIFE,400.0,300.0
1702000004.000,ARMED_PERSON,1,KNIFE,320.5,240.3
1702000005.000,VIOLENCE,-1,85.2%,320.0,240.0
```

| Event Type | Description |
|------------|-------------|
| RUN | Person running (speed > threshold) |
| FALL | Person fell (aspect ratio change) |
| LOITER | Person loitering in zone |
| DANGER | Dangerous object detected |
| ARMED_PERSON | Person holding dangerous object |
| VIOLENCE | Violence detected by classifier |

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| ultralytics | ≥8.0.0 | YOLOv8 model and inference |
| opencv-python | ≥4.8.0 | Image/video processing |
| numpy | ≥1.24.0 | Numerical computations |
| scipy | ≥1.10.0 | Linear assignment for tracking |
| tqdm | ≥4.65.0 | Progress bars |

## Limitations and Assumptions

1. **2D Vision**: Analysis is limited to 2D camera views; depth information not available
2. **Heuristic-Based**: Behavior detection (run/fall/loiter) uses hand-crafted rules
3. **Violence Classification**: Uses deep learning but trained on specific dataset; may not generalize to all scenarios
3. **Single Camera**: System designed for single fixed camera; multi-camera setup not supported
4. **Lighting Conditions**: Performance degrades in low light or high glare
5. **Occlusion**: Overlapping people or severe occlusion can confuse tracking
6. **Real-Time Latency**: Runs at ~30 FPS on modern hardware; older hardware may be slower

### Performance Notes
- Model: YOLOv8n (nano) - smallest, fastest model
- GPU recommended for consistent real-time performance
- CPU-only: ~20-30 FPS on modern CPUs
- GPU: ~50-100+ FPS depending on GPU

## Troubleshooting

### "CUDA out of memory" error
- Use CPU mode (model will auto-select) or reduce frame resolution

### Webcam not detected
- Check device permissions (Windows may require administrator)
- Try specifying camera index: modify `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`, etc.

### Very slow FPS
- Reduce frame resolution before processing
- Lower confidence threshold (faster but less accurate)
- Use smaller YOLO model (already using nano model)

### Tracking IDs flickering
- Increase `iou_threshold` in Tracker (default 0.3)
- Increase `max_age` for longer track persistence

### Falls/runs detected incorrectly
- Adjust thresholds in `cfg` dictionary in `run_behaviour.py`
- Increase `FALL_VERTICAL_RATIO_DROP` for stricter fall detection
- Decrease `RUN_SPEED_THRESHOLD` for more sensitive running detection

## Advanced Usage

### Programmatic API

```python
from yolo_object_detection.detectors import YoloDetector
from behaviour_detection.tracker import Tracker
from behaviour_detection.rules import RulesEngine
from behaviour_detection.pipeline import BehaviourPipeline

# Initialize components
detector = YoloDetector(confidence_threshold=0.5)
tracker = Tracker()
rules_engine = RulesEngine(zones={"zone1": (100, 100, 300, 300)})

# Create pipeline
pipeline = BehaviourPipeline(
    detector=detector,
    tracker=tracker,
    rules_engine=rules_engine,
    save_events_to="events.csv"
)

# Process video
pipeline.process_stream("video.mp4", show=True, save_dir="output")

# Access events programmatically
events = rules_engine.get_all_events()
for event in events:
    print(f"Type: {event['type']}, Track ID: {event['track_id']}, Time: {event['timestamp']}")
```

### Custom Zones for Loitering

Edit `run_behaviour.py` to define custom loitering zones:

```python
zones = {
    "entrance": (0, 0, 200, 480),      # Left side
    "checkout": (400, 200, 640, 480),  # Right bottom
    "staff_area": (300, 0, 640, 100),  # Top right
}
```

## Model Information

This project uses **YOLOv8n** (nano model):
- Smallest in the YOLO v8 family
- ~3M parameters
- Trained on COCO dataset (80 classes)
- Automatically downloads on first run (~7 MB)

### Changing Model Size

In `detectors.py`, modify the model name:
- `yolov8n.pt` - nano (fastest, least accurate)
- `yolov8s.pt` - small
- `yolov8m.pt` - medium
- `yolov8l.pt` - large
- `yolov8x.pt` - xlarge (most accurate, slowest)

## Performance Benchmarks

Testing on Intel i7-8700K with RTX 2070:

| Model | Detection FPS | Tracking FPS | Total FPS | Memory |
|-------|---------------|--------------|-----------|--------|
| YOLOv8n | 80-100 | N/A | 50-70 | 500 MB |
| YOLOv8s | 50-70 | N/A | 35-50 | 800 MB |

## Contributing

This is a self-contained project. To extend:

1. **Add new behaviors**: Implement in `behaviour_detection/rules.py`
2. **Improve tracking**: Modify `behaviour_detection/tracker.py`
3. **Enhance features**: Add to `behaviour_detection/features.py`

## License

MIT License - Free for commercial and personal use.

## References

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV Documentation](https://docs.opencv.org/)
- [SORT Tracking Paper](https://arxiv.org/abs/1602.00763)

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Verify all dependencies are installed: `pip list`
3. Try with a simple webcam test first: `python run_behaviour.py --source 0 --show`

---

**Version**: 1.0.0  
**Last Updated**: December 2024  
**Tested On**: Windows 10/11, Python 3.10+
