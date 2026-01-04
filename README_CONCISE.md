# AI-Powered Behaviour Detection System

**Real-time threat detection for confined closed spaces** using YOLOv8 + custom tracking + deep learning violence classification.

---

## 🚀 Getting Started

### Install & Run
```powershell
pip install -r requirements.txt
python run_behaviour.py --source 0 --show --violence-model runs/violence_cls/train/weights/best.pt
```

---

## 🔍 What Gets Detected

| Behavior | Detection Method | Works on Images? |
|----------|------------------|-----------------|
| **VIOLENCE** | Deep learning classifier (YOLOv8-cls) | ✅ Yes |
| **DANGER** | Weapon proximity check (±50px margin) | ✅ Yes |
| **RUN** | Speed > 50 px/sec | ✅ Yes |
| **FALL** | Aspect ratio drops >40% | ✅ Yes |
| **LOITER** | Stationary >10 sec in zone | ❌ No (needs video) |

**Accuracy**: 97.7% on validation set (NOT measured on real-world data)

---

## 📂 Project Structure

```
behaviour_detection/          # Core detection modules
├── pipeline.py              # Main orchestrator (detect→track→analyze→output)
├── tracker.py               # Multi-object tracking (IoU-based)
├── rules.py                 # Behavior detection (RUN/FALL/LOITER)
├── features.py              # Feature extraction (speed, motion history)
├── violence_classifier.py   # Deep learning violence detector
└── __init__.py

yolo_object_detection/       # YOLO detection wrapper
├── detectors.py             # YOLOv8 inference
├── utils.py                 # FPS meter, drawing functions
└── main.py                  # Phase 1 only (detection without tracking)

run_behaviour.py             # MAIN ENTRY POINT - Start here
CONFIG.py                    # Tunable parameters (thresholds, zones)
requirements.txt             # Dependencies (ultralytics, opencv, scipy)

datasets/
├── violence_classification/ # 29,569 training frames
└── weapon_detection_clean/  # 7,368 weapon images

runs/
├── violence_cls/train/weights/best.pt  # Violence classifier model
└── weapon_det/weights/best.pt          # Weapon detector (optional)
```

---

## 📝 Quick Commands

```powershell
# Webcam with violence detection (most common)
python run_behaviour.py --source 0 --show --violence-model runs/violence_cls/train/weights/best.pt

# Video analysis with event logging
python run_behaviour.py --source test_videos/video_test_4.mp4 --violence-model runs/violence_cls/train/weights/best.pt --events-csv events.csv

# Headless mode (no display)
python run_behaviour.py --source 0 --violence-model runs/violence_cls/train/weights/best.pt --events-csv events.csv

# Test image
python test_behavior_image.py test_images/test.jpg --violence-model runs/violence_cls/train/weights/best.pt

# Validate installation
python validate_project.py

# Run tests
python -m unittest test_behavior_detection -v
```

**Arguments**:
- `--source`: 0=webcam or path to video/image (required)
- `--violence-model`: Path to violence classifier
- `--violence-threshold`: Sensitivity (0-1, default 0.5)
- `--events-csv`: Save event log to CSV
- `--show`: Display live video
- `--conf`: Detection confidence threshold

---

## 📋 File Documentation (Brief)

| File | Purpose |
|------|---------|
| **run_behaviour.py** | Main entry point - initializes detector, tracker, rules engine, and runs pipeline |
| **pipeline.py** | Orchestrates all 4 phases: detect → track → analyze → output (~700 lines) |
| **tracker.py** | IoU-based multi-object tracker with persistent IDs (~218 lines) |
| **rules.py** | Behavior detection engine for RUN/FALL/LOITER with event logging (~257 lines) |
| **features.py** | Feature extraction: speed, aspect ratio, motion history (~250 lines) |
| **violence_classifier.py** | YOLOv8-cls wrapper for violence detection (~200 lines) |
| **detectors.py** | YOLOv8 inference wrapper with lower confidence for weapons |
| **utils.py** | FPS meter, drawing functions, argument parsing |
| **CONFIG.py** | Central configuration - all thresholds, zones, parameters |
| **validate_project.py** | Validates installation and imports |

---

## ⚠️ Important Limitations

| Feature | Status | Notes |
|---------|--------|-------|
| **Real-world accuracy** | ❌ Unknown | 97.7% verified on clean dataset only |
| **People limit** | <15 optimal | Degrades with more people |
| **Loitering** | Video only | Doesn't work on single images |
| **Occlusions** | Lost after 30 frames | No re-identification |
| **FPS** | Not benchmarked | 50-100+ claimed, not verified |
| **RTSP streams** | Not validated | Syntax supported only |

---

## 📊 CSV Event Output

```csv
timestamp,type,track_id,zone_name,centroid_x,centroid_y
1699564200.123,VIOLENCE,42,87.5%,640.2,480.1
1699564201.456,DANGER,42,main_floor,640.2,480.1
1699564202.789,RUN,43,office_a,512.5,360.0
```

**Columns**: timestamp (float), type (str), track_id (int), zone_name/violence% (str), centroid_x (float), centroid_y (float)

**Event Cooldown**: 4 seconds per behavior per person

---

## ⚙️ Configuration (CONFIG.py)

```python
DETECTION_CONFIDENCE = 0.5              # YOLOv8 confidence
RUN_SPEED_THRESHOLD = 150.0             # px/sec for running
LOITER_TIME_THRESHOLD = 10.0            # seconds of stillness
LOITER_SPEED_THRESHOLD = 50.0           # max px/sec to be considered loitering
FALL_VERTICAL_RATIO_DROP = 0.4          # aspect ratio threshold (40%)
TRACKER_MAX_AGE = 30                    # frames before track dies
TRACKER_IOU_THRESHOLD = 0.3             # IoU for association
ZONES = {
    "center": (120, 40, 520, 440),      # (x1, y1, x2, y2) in pixels
}
```

---

## 🎯 Quick Answers

**Q: How are behaviors detected?**
- **VIOLENCE**: Deep learning (YOLOv8-cls), probability > 0.5
- **DANGER**: Weapon within ±50px of person
- **RUN**: Speed > 50 px/sec
- **FALL**: Aspect ratio drops >40%
- **LOITER**: Speed < 50 px/sec for >10 sec in zone

**Q: What's the accuracy?**
- Validation: 97.7% (5,886 cleaned frames)
- Real-world: NOT measured

**Q: What are limitations?**
- Confined spaces only (<15 people optimal)
- LOITER doesn't work on images
- No re-identification after occlusions
- Per-frame violence (no temporal smoothing)

**Q: How does the pipeline work?**
Detection → Tracking → Behavior Analysis → Output (CSV/display/screenshots)

---

## 📚 Full Documentation

- **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** - Detailed technical architecture
- **[CODE_AUDIT_VERIFICATION_REPORT.md](CODE_AUDIT_VERIFICATION_REPORT.md)** - Audit findings
- **[README.md](README.md)** - Complete detailed guide (original long version)
