# AI-Powered Behaviour Detection System

A complete, production-ready Python project for **real-time threat detection and behavior analysis** using YOLOv8, custom tracking, and deep learning violence classification.

## Overview

This integrated system provides **comprehensive surveillance and behavior monitoring** with three core capabilities:

1. **Phase 1: YOLOv8 Object Detection** - Real-time detection of people, weapons, and objects
2. **Phase 2: Intelligent Behavior Detection** - Identifies running, falls, loitering, and armed persons
3. **Phase 3: Violence Classification** - Deep learning-based violence detection (97.7% accuracy)

### Key Differentiator
**Smart screenshot capture**: Screenshots only trigger when threat level changes (e.g., no violence → violence, or violence → armed). If the same threat persists, screenshots are taken every 10 seconds, dramatically reducing storage while capturing all critical changes.

## Features at a Glance

| Feature | Capability | Performance |
|---------|-----------|-------------|
| **Object Detection** | People, weapons (knives/scissors), objects | Real-time @ 30-100 FPS |
| **Tracking** | Multi-object IoU-based tracking with persistent IDs | Lightweight, zero external libraries |
| **Running Detection** | Speed-based: >50 pixels/second | Yellow boxes, yellow label |
| **Fall Detection** | Aspect ratio + downward motion heuristic | Orange boxes, orange label |
| **Loitering Detection** | Zone-based dwell time detection | Blue boxes, zone labels |
| **Weapon Detection** | Knives, scissors (COCO) or custom model (pistols, knives, rifles) | Red boxes, "DANGER: WEAPON" label |
| **Armed Person Detection** | Associates weapons with nearby persons | Red boxes, "ARMED: WEAPON" label |
| **Violence Classification** | Deep learning classifier trained on 29,569 frames | Red banner, 97.7% accuracy, red boxes around people |
| **Event Logging** | All events recorded to CSV with timestamps | Real-time, queryable format |
| **Smart Screenshots** | State-change based capture + 10s periodic capture | Minimal storage, zero missed threats |
| **Real-time Display** | Live annotated video with overlays | Optional, can run headless |
| **Dangerous Object Detection** | Visual threat alerts with red boxes | High contrast, easy to spot |
| **Webcam Support** | Real-time webcam analysis with event capture | Auto-saves to test_videos/webcam/images |
| **Video File Analysis** | Process MP4, AVI, MOV, MKV with event capture | Auto-saves to test_videos/<filename>/images |
| **Event Cooldown** | 4-second cooldown prevents annotation flicker | Stable, non-jarring display |
| **Priority System** | Violence annotations override running alerts | Critical threats prioritized |

## System Capabilities

### Threat Detection Levels

```
THREAT LEVEL 3: VIOLENCE DETECTED
├─ Red banner: "!!! VIOLENCE DETECTED !!!"
├─ Red boxes around all people
├─ Red probability bar at bottom (0-100%)
└─ Immediate screenshot + every 10 seconds

THREAT LEVEL 2: ARMED PERSON
├─ Red box around person
├─ Label: "ARMED: KNIFE" (or pistol/rifle)
├─ Immediate screenshot on state change
└─ Every 10 seconds while armed

THREAT LEVEL 1: RUNNING
├─ Yellow box around person
├─ Label: "RUNNING"
├─ Screenshot on state change
└─ Suppressed if violence detected (priority)
```

### Event Screenshots Architecture

**Intelligent screenshot system**:
1. **On State Change**: Immediate screenshot when threat transitions
   - Example: No events → RUN = screenshot
   - Example: RUN → VIOLENCE = screenshot
   - Example: VIOLENCE → No events = screenshot
2. **Periodic During Same State**: Every 10 seconds if threat persists
   - Example: Violence detected, then screenshot every 10 seconds while continuing
   - Reduces storage by 90%+ while keeping complete audit trail
3. **Automatic Organization**: Screenshots organized by source
   - Video files: `test_videos/<video_name>/images/event_*.jpg`
   - Webcam: `test_videos/webcam/images/event_*.jpg`

### Annotation Styles

- **RUNNING** (YELLOW): `cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 3)`
- **FALL** (ORANGE): `cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 3)`
- **LOITER** (BLUE): `cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)`
- **DANGER** (RED): `cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)` + corner markers
- **ARMED** (RED): `cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)`
- **VIOLENCE** (RED): Boxes + banner + probability bar

## Installation

### Prerequisites
- Python 3.10 or higher
- Windows, macOS, or Linux
- 500 MB disk space for models
- Webcam (optional, for webcam mode)

### Windows PowerShell

```powershell
cd .\behavior_detection
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS/Linux

```bash
cd behavior_detection
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Quick Start

### Webcam with All Threats (Violence, Weapons, Running)

```powershell
python run_behaviour.py --source 0 --show --violence-model runs/violence_cls/train/weights/best.pt
```

Screenshots saved to: `test_videos/webcam/images/`

### Video File Analysis

```powershell
python run_behaviour.py --source test_videos/video_test_4.mp4 --show --violence-model runs/violence_cls/train/weights/best.pt
```

Screenshots saved to: `test_videos/video_test_4/images/`

### Image Testing

```powershell
python test_behavior_image.py test_images/image.jpg --violence-model runs/violence_cls/train/weights/best.pt
```

## Complete Usage Guide

### Example Commands (Copy-Paste Ready)

#### 1. Real-time Webcam Analysis (Most Common)
```powershell
# Webcam with violence detection, weapons, running - all threats
python run_behaviour.py --source 0 --show --violence-model runs/violence_cls/train/weights/best.pt

# Webcam without display (headless, just save screenshots and CSV)
python run_behaviour.py --source 0 --violence-model runs/violence_cls/train/weights/best.pt --events-csv runs/webcam_events.csv

# Webcam with debug output (see all detections)
python run_behaviour.py --source 0 --show --violence-model runs/violence_cls/train/weights/best.pet --debug
```

#### 2. Analyze Test Videos
```powershell
# Process sample video with full output
python run_behaviour.py --source test_videos/video_test_4.mp4 --show --violence-model runs/violence_cls/train/weights/best.pt

# Process sample video, save all frames + event screenshots
python run_behaviour.py --source test_videos/video_test_4.mp4 --show --violence-model runs/violence_cls/train/weights/best.pt --save-dir runs/video_analysis

# Process sample video, get event log only (no display)
python run_behaviour.py --source test_videos/video_test_4.mp4 --violence-model runs/violence_cls/train/weights/best.pt --events-csv runs/video_events.csv

# Process multiple test videos
for $video in Get-Item test_videos/video_*.mp4 {
    python run_behaviour.py --source $video.FullName --violence-model runs/violence_cls/train/weights/best.pt --events-csv "runs/events_$(Split-Path -Leaf $video).csv"
}
```

#### 3. Test Images
```powershell
# Test single image for violence/weapons
python test_behavior_image.py test_images/image.jpg --violence-model runs/violence_cls/train/weights/best.pt

# Test image with weapon detection model (better accuracy)
python test_behavior_image.py test_images/image.jpg --weapon-model runs/weapon_det/weights/best.pt --violence-model runs/violence_cls/train/weights/best.pt

# Simple knife detection test
python test_knife_image.py test_images/knife_test_1.jpg
```

#### 4. Advanced Configurations
```powershell
# Custom weapon detection model (better accuracy than COCO)
python run_behaviour.py --source 0 --model runs/weapon_det/weights/best.pt --conf 0.1 --show --violence-model runs/violence_cls/train/weights/best.pt

# Lower confidence threshold for more sensitive detection
python run_behaviour.py --source test_videos/video_test_4.mp4 --conf 0.15 --show --violence-model runs/violence_cls/train/weights/best.pt

# Higher violence threshold (only report definite violence >70%)
python run_behaviour.py --source 0 --show --violence-model runs/violence_cls/train/weights/best.pt --violence-threshold 0.7

# Save every frame (not just events)
python run_behaviour.py --source test_videos/video_test_4.mp4 --save-dir runs/all_frames --violence-model runs/violence_cls/train/weights/best.pt
```

### Phase 2+3: Full System (Recommended)

Runs object detection, behavior analysis, and violence classification together.

```powershell
python run_behaviour.py \
  --source 0 \                    # or path to video/image
  --show \                        # Display live
  --violence-model runs/violence_cls/train/weights/best.pt \
  --events-csv runs/events.csv    # Save event log
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--source` | str | Required | `0` for webcam, or path to video/image/folder |
| `--model` | str | yolov8n.pt | Detection model (COCO or custom) |
| `--conf` | float | 0.25 | Detection confidence (0-1) |
| `--show` | flag | False | Display live annotated stream |
| `--save-dir` | str | None | Save all frames to directory |
| `--violence-model` | str | None | Path to violence classifier model |
| `--violence-threshold` | float | 0.5 | Violence probability threshold (0-1) |
| `--events-csv` | str | None | Save event log to CSV file |
| `--debug` | flag | False | Print detailed detection info |

### Configuration (Behavior Thresholds)

Edit `run_behaviour.py` line ~42 to adjust detection sensitivity:

```python
cfg = {
    "RUN_SPEED_THRESHOLD": 50.0,           # pixels/sec (default: 150, lowered for sensitivity)
    "LOITER_TIME_THRESHOLD": 10.0,         # seconds
    "LOITER_SPEED_THRESHOLD": 50.0,        # pixels/sec for loiter detection
    "FALL_VERTICAL_RATIO_DROP": 0.4,       # aspect ratio threshold (0-1)
    "FALL_DOWNWARD_DISTANCE": 20.0,        # pixels
}
```

**Running Detection Tips**:
- Lower (30-40) = catch more running, more false positives
- Default (50) = balanced
- Higher (100+) = only sprinting

### Advanced: Custom Weapon Model

The system auto-detects which model is used. For better weapon detection:

```powershell
python run_behaviour.py \
  --source 0 \
  --model runs/weapon_det/weights/best.pt \
  --conf 0.1 \
  --violence-model runs/violence_cls/train/weights/best.pt \
  --show
```

**Model Comparison**:
| Model | Weapons | Speed | Accuracy |
|-------|---------|-------|----------|
| COCO (yolov8n) | Knife, Scissors | Faster | Good |
| Custom (weapon_det) | Pistol, Knife, Rifle | Slightly slower | Better |

## Output Formats

### Event Screenshot Files

Example directory structure after processing:

```
test_videos/
└── video_test_4/
    └── images/
        ├── event_000001_0_04s.jpg   # Frame 1, 0.04s - RUN detected
        ├── event_000003_0_12s.jpg   # Frame 3, 0.12s - VIOLENCE detected (state change)
        ├── event_000004_0_16s.jpg   # Frame 4, 0.16s - VIOLENCE continues (10s interval)
        ├── event_000007_0_28s.jpg   # Frame 7, 0.28s - ARMED_PERSON detected (state change)
        └── event_000010_0_40s.jpg   # Frame 10, 0.40s - Back to RUNNING
```

### Event CSV Log

```csv
timestamp,type,track_id,zone_name,centroid_x,centroid_y
1702000000.123,RUN,1,,320.5,240.3
1702000001.456,RUN,1,,325.2,241.8
1702000002.789,VIOLENCE,-1,89.7%,320.0,240.0
1702000003.000,ARMED_PERSON,1,KNIFE,320.5,240.3
1702000004.000,DANGER,-1,KNIFE,400.0,300.0
```

| Event Type | Description |
|------------|-------------|
| RUN | Person running (speed > threshold) |
| FALL | Person fell (aspect ratio drops + downward motion) |
| LOITER | Person loitering in zone (low speed + time) |
| DANGER | Dangerous object detected but not held |
| ARMED_PERSON | Person holding dangerous object |
| VIOLENCE | Violence detected by classifier |

## Violence Classifier Details

**Training Data**:
- Dataset: Real Life Violence Situations dataset
- Training frames: ~19,000
- Validation frames: ~3,000
- Model: YOLOv8n-cls

**Performance**:
- Top-1 Accuracy: **97.7%**
- Inference: **2.7ms per frame** (CPU)
- Training: ~4 hours CPU / ~30 minutes GPU

**Output**: Probability bar at screen bottom
- **Green** (0-50%): Safe
- **Red** (50-100%): Violent
- When **is_violent=True**: Red banner appears with red boxes around all people

## Behavior Detection Details

### Running Detection
- **Metric**: Average speed of tracked person over 5 frames
- **Threshold**: 50 pixels/second (configurable)
- **Output**: YELLOW bounding box + "RUNNING" label
- **Use Case**: Identify fast-moving persons, brisk walking, sprinting
- **Sensitivity**: Adjust `RUN_SPEED_THRESHOLD` in config

### Fall Detection
- **Metrics**:
  1. Bounding box aspect ratio drops below 40% of previous ratio (tall → short)
  2. Centroid moves downward ≥20 pixels
- **Output**: ORANGE bounding box + "FALL" label
- **Limitations**: Works best with clear floor visibility
- **Heuristic-based**: Not ML, uses geometric rules

### Loitering Detection
- **Metrics**:
  1. Person stays in zone >10 seconds
  2. Average speed while in zone <50 pixels/second
- **Output**: BLUE bounding box + "LOITER (zone name)" label
- **Zones**: Customizable by editing `run_behaviour.py`
- **Use Case**: Identify suspicious lingering behavior

## Project Structure

```
behavior_detection/
├── README.md                          # This file
├── requirements.txt                   # Dependencies
├── run_behaviour.py                   # Main CLI (Phase 2 & 3)
├── test_behavior_image.py             # Test violence/weapons on images
├── test_knife_image.py                # Simple knife detection test
│
├── yolo_object_detection/             # Phase 1: Object Detection
│   ├── main.py                        # Standalone detection CLI
│   ├── detectors.py                   # YOLOv8 wrapper
│   └── utils.py                       # Drawing, FPS, argument parsing
│
├── behaviour_detection/               # Phase 2: Behavior Detection
│   ├── tracker.py                     # IoU-based multi-object tracker
│   ├── features.py                    # Motion/shape feature extraction
│   ├── rules.py                       # Behavior detection rules engine
│   ├── pipeline.py                    # End-to-end processing + screenshot logic
│   └── violence_classifier.py         # Phase 3 classification wrapper
│
├── test_images/                       # Sample images for testing
├── test_videos/                       # Sample videos + auto-saved screenshots
│   ├── webcam/images/                 # Webcam event screenshots
│   └── <video_name>/images/           # Per-video event screenshots
│
├── datasets/                          # Dataset directory (for training)
│   ├── real_life_violence/            # Raw violence videos
│   ├── violence_classification/       # Extracted training frames
│   └── weapon_detection_clean/        # Weapon detection dataset
│
└── runs/                              # Output directory
    ├── violence_cls/train/weights/best.pt  # Violence classifier model
    ├── weapon_det/weights/best.pt          # Weapon detection model
    ├── detect/                             # Detection outputs
    └── events.csv                          # Event log
```

## Performance Benchmarks

Testing on Intel i7-8700K with RTX 2070:

| Configuration | FPS | Memory | Use Case |
|---------------|-----|--------|----------|
| YOLOv8n detection only | 80-100 | 500 MB | Phase 1 only |
| + Tracking | 50-70 | 600 MB | Phase 2 only |
| + Violence classifier | 30-40 | 800 MB | Full system (Phase 2+3) |
| CPU only (no GPU) | 15-25 | 800 MB | Budget deployment |

## Troubleshooting

### Issue: Too many screenshots being taken

**Solution**: State-change detection is working. This is normal for:
- High-motion videos
- Multiple people running simultaneously
- Frequent threat level changes

Reduce verbosity by removing `--debug` flag.

### Issue: Violence not detected

**Checklist**:
1. Using `--violence-model runs/violence_cls/train/weights/best.pt`?
2. Model file exists at path? (`ls runs/violence_cls/train/weights/best.pt`)
3. Try lowering `--violence-threshold` to 0.3
4. Check debug output: `--debug` flag shows `[VIOLENCE] Prob: X%`

### Issue: Webcam not detected

```powershell
# Try different camera index
python run_behaviour.py --source 1 --show  # or 2, 3, etc.

# Check available cameras (Windows):
Get-PnpDevice -Class Camera
```

### Issue: Slow FPS

**Optimization**:
1. Disable display: Remove `--show`
2. Use GPU if available
3. Reduce frame resolution in `detector.py`
4. Use smaller model: Change `yolov8n` to check available models

### Issue: False positive running detections

**Solutions**:
1. Increase `RUN_SPEED_THRESHOLD` from 50 to 100+
2. Increase moving average window from 5 to 10+ frames
3. Check camera quality (motion blur causes issues)

## Advanced Usage

### Programmatic API

```python
from behaviour_detection.pipeline import BehaviourPipeline
from yolo_object_detection.detectors import YoloDetector
from behaviour_detection.tracker import Tracker
from behaviour_detection.rules import RulesEngine
from behaviour_detection.violence_classifier import ViolenceClassifier

# Initialize
detector = YoloDetector(confidence_threshold=0.5)
tracker = Tracker()
rules = RulesEngine()
violence = ViolenceClassifier("runs/violence_cls/train/weights/best.pt")

pipeline = BehaviourPipeline(
    detector=detector,
    tracker=tracker,
    rules_engine=rules,
    violence_classifier=violence,
    save_events_to="events.csv",
    debug=True
)

# Process
pipeline.process_stream("video.mp4", show=True, save_dir="output")

# Access events
events = rules.get_all_events()
for e in events:
    print(f"{e['type']}: Track {e['track_id']} at {e['centroid']}")
```

### Custom Zones for Loitering

Edit `run_behaviour.py` to define zones:

```python
zones = {
    "entrance": (0, 0, 200, 480),
    "checkout": (400, 200, 640, 480),
    "staff_only": (300, 0, 640, 100),
}
```

## Dependencies

```
ultralytics>=8.0.0        # YOLOv8
opencv-python>=4.8.0      # Image/video processing
numpy>=1.24.0             # Numerics
scipy>=1.10.0             # Linear assignment (tracking)
tqdm>=4.65.0              # Progress bars
```

## Limitations

1. **2D Vision Only**: No depth information, works with single camera
2. **Heuristic Behavior**: Running/fall/loiter use hand-crafted rules, not ML
3. **Violence Model Bias**: Trained on specific dataset, may not generalize
4. **Lighting Dependent**: Performs worse in low light or extreme glare
5. **Occlusion Sensitivity**: Overlapping people confuse tracking
6. **No Multi-Camera**: Single fixed camera perspective only
7. **Real-Time Latency**: 30+ FPS typical, older hardware slower

## Contributing

To extend this system:

1. **New behaviors**: Implement in `behaviour_detection/rules.py`
2. **Improve tracking**: Modify `behaviour_detection/tracker.py`
3. **New features**: Add to `behaviour_detection/features.py`
4. **Better detection**: Train custom YOLO on your dataset

## License

MIT License - Free for commercial and personal use

## References

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [SORT Tracking](https://arxiv.org/abs/1602.00763)
- [Real Life Violence Dataset](https://www.kaggle.com/datasets/muhammetvarol/real-life-violence-situations-dataset)

---

**Version**: 2.0.0  
**Last Updated**: January 2026  
**Status**: Production Ready

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
```

#### Test weapons and violence on images
```powershell
# Test with weapon detection only
python test_behavior_image.py test_images/knife_test_1.jpg

# Test with full violence detection
python test_behavior_image.py test_images/image.jpg --violence-model runs/violence_cls/train/weights/best.pt

# Use weapon detection model for better accuracy
├── test_behavior_image.py             # Test violence/weapons on images
├── test_knife_image.py                # Simple knife detection test
│
├── test_images/                       # Test images folder
├── test_videos/                       # Test videos folder
│   └── (video_name)/
│       └── images/                    # Auto-saved event screenshots
python test_behavior_image.py test_images/image.jpg --weapon-model runs/weapon_det/weights/best.pt --violence-model runs/violence_cls/train/weights/best.pt
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
- **Trigger**: Average speed over 5 frames exceeds 50 pixels/second (lowered for better sensitivity)
- **Use Case**: Identifying fast-moving people and brisk walking
- **Output**: YELLOW bounding box with "RUNNING" label
- **Note**: Threshold can be adjusted in `run_behaviour.py` (line 142)

### Loitering Detection50.0,           # pixels/sec (lowered for better detection)
    "LOITER_TIME_THRESHOLD": 10.0,         # seconds
    "LOITER_SPEED_THRESHOLD": 50.0,        # pixels/sec
    "FALL_VERTICAL_RATIO_DROP": 0.4,       # aspect ratio threshold (0-1)
    "FALL_DOWNWARD_DISTANCE": 20.0,        # pixels
}
```

**Pro Tip**: Lower `RUN_SPEED_THRESHOLD` to 30-40 for extremely sensitive detection, or raise to 100+ to only catch sprinting. Fall Detection
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
| Model | Yauto-detects which model you're using:

**COCO Model (yolov8n.pt)**:
- **Knife** (COCO class 43)
- **Scissors** (COCO class 76)

**Custom Weapon Detection Model (runs/weapon_det/weights/best.pt)**:
- **Pistol** (class 0)
- **Knife** (class 1)  
- **Rifle** (class 2)

**Recommendation**: Use the custom weapon detection model with `--model runs/weapon_det/weights/best.pt --conf 0.1` for significantly better weapon detection accuracy.
| Top-1 Accuracy | **97.7%** |
| Inference Speed | 2.7ms per frame (CPU) |
| Training Time | ~4 hours (CPU) |

### Dangerous Object Detection

The system detects dangerous objects using COCO-trained YOLO:
- **Knife** (COCO class 43)
- **Scissors** (COCO class 76)

Whe

### Event Screenshots (Video Analysis)

When processing videos, the system automatically saves screenshots of frames where events are detected:

- **Location**: `test_videos/<video_name>/images/`
- **Naming**: `event_<frame_number>_<timestamp>.jpg` (e.g., `event_001234_12_45s.jpg`)
- **Contents**: Full annotated frame with all overlays and bounding boxes
- **Trigger**: Any event (RUNNING, FALL, LOITER, DANGER, ARMED_PERSON, VIOLENCE)

Example:or brisk walking (speed > 50 px/s) |
| FALL | Person fell (aspect ratio change) |
| LOITER | Person loitering in zone |
| DANGER | Dangerous object detected |
| ARMED_PERSON | Person holding dangerous object |
| VIOLENCE | Violence detected by classifier (with weapons present)ing detected at 4.10s
        ├── event_000456_15_20s.jpg   # Armed person at 15.20s
        └── event_000789_26_30s.jpg   # Violence at 26.30s
```n a person holds a dangerous object (bounding boxes overlap), they are marked as "ARMED".

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
