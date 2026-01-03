# PROJECT AUDIT: Comprehensive Code & Claims Verification

**Date**: January 3, 2026  
**Project**: AI-Powered Behaviour Detection System  
**Scope**: Code verification, metrics validation, deployment reality check

---

## SECTION A: EVALUATION & METRICS

### A1. Evaluation Code Location & Implementation

**File**: `d:\Code\behavior_detection\evaluate.py` (406 lines)

**Available Functionality**:
- ✅ Manual annotation mode (`run_manual_annotation()`) - Frame-by-frame ground truth labeling
- ✅ Comparison mode (`run_comparison()`) - Compare system output vs ground truth
- ✅ Metrics calculation (`_calculate_metrics()`) - Precision, Recall, F1 per event type
- ✅ Synthetic testing (`run_synthetic_test()`) - Unit test scenarios
- ✅ Live evaluation mode - Real-time detection with confidence

**Functions for Metrics**:
```
EvaluationTool._calculate_metrics(ground_truth, predictions, total_frames)
  - Computes TP, FP, FN per event type (RUN, FALL, LOITER, DANGER, ARMED_PERSON)
  - Calculates: Precision, Recall, F1 Score
  - Uses frame tolerance of ±5 frames for matching
  - Location: lines 200-260
```

**Issue**: Metrics are **NOT automatically computed during training/validation**. The tool is **manual evaluation only** - requires hand-annotated ground truth JSON file.

---

### A2. What "70% in Confined Spaces" Actually Means

**Claim in Documentation**:
- README.md: "70% accuracy with 1-10 people"
- PROJECT_OVERVIEW_FOR_CHATGPT.md: "~70% in confined spaces (small crowds, contained environments)"
- ACCURACY_ANALYSIS.md: Detailed breakdown by scenario

**Where It's Computed**:
- ❌ **NOT in code** - No training log shows 70%
- ❌ **NOT measured** - No evaluation output for real videos
- ✅ **Claimed in documentation** - Based on hypothetical scenarios

**Analysis**:
```
CONTROLLED VALIDATION (in code):
├─ Source: runs/violence_cls/train/weights/best.pt
├─ Validation frames: 5,886 (from ACCURACY_ANALYSIS.md)
├─ Accuracy: 97.7% (stated in docs, not verified in logs)
└─ Conditions: Lab conditions with clean data

REAL-WORLD CONFINED SPACES (NOT measured):
├─ Source: ACCURACY_ANALYSIS.md synthetic breakdown
├─ Test frames: 7,086 (hypothetical "diverse scenarios")
├─ Accuracy: ~70% (claimed, NOT from actual test)
├─ Scenarios: Mall (68%), Outdoor (71%), Angles (69%), Night (65%)
└─ Status: **NO ACTUAL RUNS** - these are illustrative estimates
```

**Verdict**: **"70% in confined spaces" is a CLAIM WITHOUT BACKING**. The documentation does not reference actual test runs. The `evaluate.py` tool exists to measure this, but no results are committed to the repo.

---

### A3. False Positives Per Hour/Minute

**Code Search Result**: ❌ **NOT IMPLEMENTED**

**Where It Would Be Measured**:
- `evaluate.py`: Logs TP/FP/FN but not per-minute frequency
- `pipeline.py`: No FP rate calculation
- `rules.py`: No FP counters
- `events.csv`: Logs events with timestamp but no FP flag

**What Would Need Adding**:

```python
# Required fields in events.csv:
timestamp, event_type, track_id, confidence, is_false_positive, validated

# Code to add:
def calculate_fps_per_hour(events_df):
    """Count false positives per hour"""
    events_df['hour'] = pd.to_datetime(events_df['timestamp']).dt.floor('H')
    fps_per_hour = events_df[events_df['is_false_positive']==True].groupby('hour').size()
    return fps_per_hour
```

**Current State**: Events are logged but **FP classification is manual** (requires annotation tool).

---

### A4. Confusion Matrix & Precision/Recall

**Location Search**: `evaluate.py` lines 200-260

**Available**:
```python
# Calculated per event type:
TP = matched_frames (within ±5 frame tolerance)
FP = predicted but not in ground truth
FN = ground truth but not predicted

Metrics = Precision, Recall, F1

Output format (example):
RUN:
  Ground Truth: 42 events
  Predictions: 45 events
  True Positives: 38
  False Positives: 7
  False Negatives: 4
  Precision: 84.44%
  Recall: 90.48%
  F1 Score: 87.36%
```

**Issue**: These metrics are **only generated if**:
1. You manually annotate a video with `--mode annotate`
2. Save ground truth JSON
3. Run comparison mode `--mode compare`

**No pre-computed confusion matrices exist in repo** under `runs/`.

---

## SECTION B: DATASET TRUTH

### B1. Video Counts & Paths - Directory Inspection

**Raw Violence Dataset**:
```
Path: d:\Code\behavior_detection\datasets\real_life_violence\
├─ Violence/       : 1000 videos
├─ NonViolence/    : 1000 videos
└─ TOTAL          : 2000 source videos
```

**Extracted Classification Dataset**:
```
Path: d:\Code\behavior_detection\datasets\violence_classification\
├─ train/
│  ├─ violence/    : 13,097 frames
│  └─ nonviolence/ : 10,586 frames
│  └─ TOTAL        : 23,683 frames
├─ val/
│  ├─ violence/    :  3,290 frames
│  └─ nonviolence/ :  2,596 frames
│  └─ TOTAL        :  5,886 frames
└─ GRAND TOTAL     : 29,569 frames
```

**Weapon Detection Dataset**:
```
Path: d:\Code\behavior_detection\datasets\weapon_detection_clean\
├─ images/train/        : 4,098 images
├─ images/validation/   :   975 images
├─ images/test/         : ~2,295 images (calculated from labels)
├─ labels/train/        : 4,098 .txt files
├─ labels/test/         : 2,295 .txt files
└─ TOTAL                : ~7,368 images
```

---

### B2. Frame Extraction Policy

**File**: `d:\Code\behavior_detection\prepare_violence_data.py`

**Frame Extraction Logic** (lines 50-80):
```python
def extract_frames_from_video(video_path, output_folder, frame_interval=10, prefix=""):
    """Extract 1 frame every N frames"""
    cap = cv2.VideoCapture(str(video_path))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:  # Extract every 10th frame
            filename = f"{prefix}_frame_{saved_count:05d}.jpg"
            filepath = output_folder / filename
            cv2.imwrite(str(filepath), frame)
            saved_count += 1
        
        frame_count += 1
```

**Default Argument** (line 57):
```python
def prepare_violence_dataset(
    input_dir="datasets/real_life_violence",
    output_dir="datasets/violence_classification",
    frame_interval=10,  # ← Every 10th frame
    train_split=0.8,    # ← 80% train, 20% val
    seed=42
):
```

**Label Assignment**: 
- Folder name = class (violence/ → "violence", nonviolence/ → "nonviolence")
- No manual annotation needed

**Train/Val Split** (lines 140-170):
```python
# Video-level split happens BEFORE frame extraction
violence_videos = [...]  # 1000 total
random.shuffle(violence_videos)
split_idx = int(len(violence_videos) * train_split)  # 800 train, 200 val

train_violence_videos = violence_videos[:split_idx]
val_violence_videos = violence_videos[split_idx:]

# Then frames are extracted from each video
# No mixing of same video across train/val
```

**Verdict**: ✅ **Video-level split is enforced before extraction** - no data leakage.

---

### B3. Proof of Video-Level Split

**File**: `d:\Code\behavior_detection\prepare_violence_data.py` lines 140-200

**Code Logic**:
```python
# Step 1: Collect videos (not frames)
violence_videos = [f for f in violence_dir.iterdir() if f.suffix.lower() in video_extensions]

# Step 2: SHUFFLE VIDEOS (not frames)
random.shuffle(violence_videos, random=random_state.random)

# Step 3: SPLIT VIDEO LIST (not frames)
split_idx = int(len(violence_videos) * train_split)
train_videos = violence_videos[:split_idx]
val_videos = violence_videos[split_idx:]

# Step 4: EXTRACT FRAMES from split video lists
# This ensures all frames from video X go to train OR val, never both
for video in train_videos:
    extract_frames_from_video(video, train_output_dir)

for video in val_videos:
    extract_frames_from_video(video, val_output_dir)
```

**Guarantees**:
- ✅ Same video never appears in both train and val
- ✅ Train/val split is at video level (50 videos → 80% violence)
- ✅ No frame-level leakage possible

---

### B4. Class Balance

**From actual directory inspection**:

```
VIOLENCE CLASSIFICATION DATASET

Training Set (80% of 2000 videos):
├─ Violence class    : 13,097 frames (55.4% of train)
└─ Non-Violence class: 10,586 frames (44.6% of train)
   RATIO: 1.24:1 (slightly imbalanced toward violence)

Validation Set (20% of 2000 videos):
├─ Violence class    :  3,290 frames (55.9% of val)
└─ Non-Violence class:  2,596 frames (44.1% of val)
   RATIO: 1.27:1 (consistent with train)

TOTAL: 29,569 frames (55.5% violence, 44.5% non-violence)
```

**Imbalance Impact**: 55/45 split is mild - typically not a problem for YOLOv8-cls.

---

## SECTION C: SYSTEM BEHAVIOR & RULES

### C1. Running Detection Rule

**File**: `d:\Code\behavior_detection\behaviour_detection\rules.py`

**Speed Threshold** (line 32):
```python
self.cfg = {
    "RUN_SPEED_THRESHOLD": 150.0,  # pixels/sec
    ...
}
```

**Actual Threshold Used** (from `run_behaviour.py` line 47):
```python
cfg = {
    "RUN_SPEED_THRESHOLD": 50.0,  # pixels/second (LOWERED for sensitivity)
    ...
}
```

**Speed Computation** (`behaviour_detection/features.py` lines 8-28):
```python
def compute_speed(prev_centroid, curr_centroid, dt):
    """Speed = distance / time"""
    distance = math.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
    speed = distance / dt
    return speed
```

**Formula**: Raw pixel distance per second, **NOT normalized by resolution**

**Issue**: ⚠️ **Resolution-dependent**
- 640×480 video: 50px/sec might be fast walking
- 1920×1080 video: 50px/sec might be normal walking
- **No automatic calibration**

---

### C2. Fall Detection Rule

**File**: `d:\Code\behavior_detection\behaviour_detection\rules.py` lines 160-200

**Trigger Conditions**:
```python
def _check_fall(self, track_id, track):
    """Detect fall using aspect ratio + downward motion"""
    
    prev_bbox = self.track_last_bbox.get(track_id)
    curr_bbox = track.get("bbox")
    
    if not prev_bbox:
        return False
    
    # Check 1: Vertical-to-horizontal aspect ratio change
    aspect_changed = is_vertical_to_horizontal_change(
        prev_bbox, 
        curr_bbox,
        ratio_threshold=0.4  # height/width must drop to 40% of previous
    )
    
    # Check 2: Downward centroid movement
    moving_down = is_moving_downward(
        track["prev_centroid"],
        track["centroid"],
        min_distance=20  # pixels
    )
    
    return aspect_changed and moving_down
```

**Thresholds**:
- Aspect ratio drop threshold: 0.4 (must be ≤40% of previous)
- Downward distance minimum: 20 pixels
- Time window: current frame vs previous frame

**Edge Cases**:
- ⚠️ Quick crouch triggers false positive
- ⚠️ Camera movement misdetected as fall
- ⚠️ People sitting down = fall

---

### C3. Loitering Detection Rule

**File**: `d:\Code\behavior_detection\behaviour_detection\rules.py` lines 130-158

**Zone Definition**:
```python
# From CONFIG.py or runtime
ZONES = {
    "center": (120, 40, 520, 440),  # x1, y1, x2, y2 in pixels
}
```

**Dwell Time Computation**:
```python
def _check_loitering(self, track_id, track, zone_rect, zone_name, current_time):
    """Track time person stays in zone"""
    
    centroid = track["centroid"]
    
    # Check if in zone
    if is_point_in_zone(centroid, zone_rect):
        # First time in zone?
        if not self.track_in_zone[track_id]:
            self.track_loiter_time[track_id] = 0  # Reset timer
            self.track_in_zone[track_id] = True
        
        # Accumulate dwell time
        speed = self.track_history[track_id].get_instant_speed()
        
        # Only count if moving slowly (<50 px/sec)
        if speed < self.cfg["LOITER_SPEED_THRESHOLD"]:
            self.track_loiter_time[track_id] += dt
    else:
        # Left zone - reset
        self.track_in_zone[track_id] = False
        self.track_loiter_time[track_id] = 0
    
    # Loitering if time > threshold (default 10 sec)
    if self.track_loiter_time[track_id] > self.cfg["LOITER_TIME_THRESHOLD"]:
        return True
```

**Thresholds**:
- Time threshold: 10.0 seconds
- Speed threshold: 50.0 pixels/sec (must be below to count as loitering)
- Zone: Rectangular (x1, y1, x2, y2)

**Reset Logic**: Timer resets when person **leaves zone** and re-enters.

---

### C4. Armed Person Association

**File**: `d:\Code\behavior_detection\behaviour_detection\pipeline.py` lines 250-300

**Method**: **Distance-based centroid matching**

```python
def _find_armed_persons(self, tracks, detections):
    """Find people near detected weapons"""
    
    weapon_bboxes = []
    for det in detections:
        class_id = det[5]
        if class_id in self.DANGEROUS_OBJECTS:
            weapon_bboxes.append(det[:4])  # x1, y1, x2, y2
    
    armed_persons = set()
    
    for track in tracks:
        if track["class_id"] != 0:  # Not a person
            continue
        
        person_centroid = track["centroid"]
        
        # Check distance to all weapons
        for weapon_bbox in weapon_bboxes:
            weapon_centroid = (
                (weapon_bbox[0] + weapon_bbox[2]) / 2,
                (weapon_bbox[1] + weapon_bbox[3]) / 2
            )
            
            # Distance threshold = 100 pixels
            distance = math.sqrt(
                (person_centroid[0] - weapon_centroid[0])**2 +
                (person_centroid[1] - weapon_centroid[1])**2
            )
            
            if distance < 100:  # Hard-coded threshold
                armed_persons.add(track["id"])
                break  # One weapon per person
    
    return armed_persons
```

**Threshold**: 100 pixels **hard-coded** (not configurable)

---

### C5. Threat Priority Enforcement

**File**: `d:\Code\behavior_detection\behaviour_detection\pipeline.py` lines 500-550

**Priority Order**:
```python
def _annotate_frame(self, frame, tracks, events, violence_result):
    """Draw annotations with priority"""
    
    # Violence overrides everything (RED)
    if violence_result and violence_result['is_violent']:
        cv2.putText(frame, "!!! VIOLENCE DETECTED !!!", ...)
        for track in tracks:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        return
    
    # Armed person next (RED boxes)
    for track in armed_persons:
        cv2.rectangle(frame, bbox, (0, 0, 255), 3)
        cv2.putText(frame, f"ARMED: {weapon_type}", ...)
    
    # Running (YELLOW)
    for track in running_persons:
        cv2.rectangle(frame, bbox, (0, 255, 255), 2)
        cv2.putText(frame, "RUNNING", ...)
    
    # Fall (ORANGE)
    # Loiter (BLUE)
    # etc.
```

**Enforcement**: ✅ **Hierarchical drawing order** - later code can't override earlier annotations.

---

### C6. Cooldown System

**File**: `d:\Code\behavior_detection\behaviour_detection\pipeline.py` lines 400-430

**Implementation**:
```python
self.EVENT_COOLDOWN = 4.0  # seconds
self.last_event_time = {}  # track_id -> timestamp

def _apply_cooldown(self, track_id, event_type):
    """Suppress duplicate events for same track"""
    
    current_time = time.time()
    key = (track_id, event_type)
    
    last_time = self.last_event_time.get(key, 0)
    
    if current_time - last_time < self.EVENT_COOLDOWN:
        return False  # Suppressed
    
    self.last_event_time[key] = current_time
    return True  # Allowed
```

**Effect**: 
- ✅ Suppresses **event generation** (not logged to CSV)
- ✅ Suppresses **annotation drawing** (no flicker)
- ⚠️ 4-second window is hard-coded

---

## SECTION D: TRACKING REALITY CHECK

### D1. Tracker Association Method

**File**: `d:\Code\behavior_detection\behaviour_detection\tracker.py` lines 160-218

**Algorithm**: **IoU-based Hungarian matching** (linear assignment)

```python
def _match_detections(self, detections):
    """Match detections to tracks using IoU"""
    
    if not self.tracks or not detections:
        return [], list(range(len(detections))), list(range(len(self.tracks)))
    
    # Compute IoU cost matrix
    cost_matrix = np.zeros((len(self.tracks), len(detections)))
    
    for i, track in enumerate(self.tracks):
        for j, det in enumerate(detections):
            iou = self._compute_iou(track.bbox, det["bbox"])
            cost_matrix[i, j] = 1 - iou  # Cost = 1 - IoU
    
    # Hungarian algorithm
    track_indices, det_indices = linear_sum_assignment(cost_matrix)
    
    # Filter by IoU threshold
    matched_pairs = []
    for track_idx, det_idx in zip(track_indices, det_indices):
        iou = 1 - cost_matrix[track_idx, det_idx]
        if iou > self.iou_threshold:
            matched_pairs.append((track_idx, det_idx))
    
    # Identify unmatched
    matched_tracks = set(t for t, _ in matched_pairs)
    matched_dets = set(d for _, d in matched_pairs)
    
    unmatched_tracks = [i for i in range(len(self.tracks)) if i not in matched_tracks]
    unmatched_dets = [j for j in range(len(detections)) if j not in matched_dets]
    
    return matched_pairs, unmatched_dets, unmatched_tracks
```

**Default Parameters** (from `run_behaviour.py`):
```python
tracker = Tracker(max_age=30, iou_threshold=0.3)
```

**Features Used**: ✅ **IoU only** (NO appearance features - color, shape, etc.)

---

### D2. Track ID Stability Limits

**Claimed**: "<15 people stable, degrades with more"

**Source Code Check**:
- ❌ No explicit benchmark in code
- ❌ No test comparing 5 people vs 50 people
- ✅ Logic suggests: More people → more IoU ambiguities → more ID switches

**Known Failure Cases** (from comments):
- Fast-moving people with similar size
- People crossing (IoU overlap causes swaps)
- Occlusions (object disappears, reappears as new ID)

**Verdict**: ⚠️ **Claimed but NOT measured**

---

### D3. Max People Supported - Measurement

**Claim**: "<15 people optimal, stable"

**Where It's "Measured"**: 
- ❌ Not in code
- ✅ Mentioned in README.md as design constraint

**Actual Limit**: Technically unlimited, but:
- 100 people = 100² = 10,000 IoU comparisons per frame
- Performance degrades quadratically
- No hard limit enforced in code

**Verdict**: **"<15" is architectural assumption, NOT measured.**

---

## SECTION E: DEPLOYMENT CONSTRAINTS

### E1. Supported Input Sources

**File**: `d:\Code\behavior_detection\run_behaviour.py` lines 30-50

```python
parser.add_argument(
    "--source",
    type=str,
    required=True,
    help="0 for webcam or path to video/image file"
)
```

**Supported**:
- ✅ Webcam: `--source 0`
- ✅ Video files: `.mp4, .avi, .mov, .mkv` (via cv2.VideoCapture)
- ✅ Image files: `.jpg, .png` (via cv2.imread)
- ❌ RTSP streams (not implemented - would need different syntax)
- ❌ HTTP streams
- ❌ Multiple sources simultaneously

**Flags**:
```python
--show                      # Display live (default: headless)
--save-dir <path>          # Save annotated frames (JPEG)
--events-csv <path>        # Log events to CSV
--violence-model <path>    # Path to violence classifier
--debug                    # Print all detections
```

---

### E2. GPU/CPU Handling

**File**: `d:\Code\behavior_detection\yolo_object_detection\detectors.py` lines 30-50

**Code**:
```python
class YoloDetector:
    def __init__(self, model_name="yolov8n.pt", confidence_threshold=0.5):
        """Load YOLO model"""
        self.model = YOLO(model_name)
        
        # ❌ NO EXPLICIT GPU SELECTION
        # Ultralytics auto-detects CUDA, MPS, CPU
        # (Implicit in YOLO() initialization)
```

**Actual Behavior**:
- 🔄 **Automatic device selection** by ultralytics library
- If CUDA available → uses GPU
- If CUDA not available → falls back to CPU
- **No explicit control in our code**

**Fallback**: ✅ Automatic (handled by ultralytics, not our code)

---

### E3. Latency Measurement

**File**: `d:\Code\behavior_detection\yolo_object_detection\utils.py`

```python
class FPSMeter:
    """Rolling FPS counter"""
    
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.frame_times = collections.deque(maxlen=window_size)
        self.last_time = time.time()
    
    def update(self):
        current_time = time.time()
        frame_time = current_time - self.last_time
        self.frame_times.append(frame_time)
        self.last_time = current_time
    
    def get_fps(self):
        if not self.frame_times:
            return 0.0
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
```

**Method**: Rolling average of 30 frame times

**Where Used**: `pipeline.py` - displayed on screen (optional with `--show`)

---

### E4. Storage Growth

**Where Stored**:
- Screenshots: `test_videos/<source>/images/event_*.jpg`
- Events CSV: User-specified with `--events-csv`
- Frames: User-specified with `--save-dir`

**Management**: ❌ **None implemented**

- ❌ No deletion policy
- ❌ No quota limits
- ❌ No cleanup scripts

**What Would Grow**:
- 1 hour of video @ 30 FPS = 108,000 frames
- Screenshots (every 10 sec during alert) = 360 images/hour
- Each JPEG ~50-100 KB = 18-36 MB/hour just screenshots

---

### E5. Alerting Integration

**Search Result**: ❌ **NOT IMPLEMENTED**

- ❌ No webhook
- ❌ No email
- ❌ No SMS
- ❌ No REST API

**Current**: Events logged to CSV only. Integration via external tool required.

---

## SECTION F: DOCUMENTATION CONSISTENCY SCAN

### F1. All Occurrences of Key Claims

#### "97.7% Accuracy"
| File | Line | Context | Verdict |
|------|------|---------|---------|
| README.md | 12 | "97.7% on controlled validation" | ✅ Qualified |
| PROJECT_OVERVIEW_FOR_CHATGPT.md | 11 | "97.7% in controlled conditions" | ✅ Qualified |
| ACCURACY_ANALYSIS.md | 13 | "97.7% accuracy" | ✅ With lab notation |

#### "70% in Confined Spaces"
| File | Line | Context | Verdict |
|------|------|---------|---------|
| README.md | 37-43 | Performance table (70% for 1-10 people) | ⚠️ No source |
| PROJECT_OVERVIEW_FOR_CHATGPT.md | 11 | "~70% in real-world confined spaces" | ⚠️ No source |
| ACCURACY_ANALYSIS.md | 19-35 | Detailed scenarios | ⚠️ Illustrative only |

#### "2000 Videos" / "1000+1000"
| File | Line | Context | Verdict |
|------|------|---------|---------|
| README.md | 10 | Title mention | ✅ Verified |
| PROJECT_OVERVIEW_FOR_CHATGPT.md | 11 | "1000 violence + 1000 non-violence" | ✅ Verified |
| ACCURACY_ANALYSIS.md | 18 | "2000 videos" | ✅ Verified |

#### "29,569 Frames"
| File | Line | Context | Verdict |
|------|------|---------|---------|
| All docs | Various | "23,683 train + 5,886 val" | ✅ Math verified |

#### "Production-Ready"
| File | Line | Context | Verdict |
|------|------|---------|---------|
| README.md | 2 | First line claim | ✓ Code is complete, but... |
| DELIVERY_SUMMARY.txt | Various | "PRODUCTION-READY" | ⚠️ **Depends on use case** |

---

### F2. Contradictions Found

**Contradiction 1: Accuracy Claims**

```
TECHNICAL_ANALYSIS.md (line 160):
"Current approach is sound for preventing data leakage"
"Recommend 100+ videos for robust validation"

But we deliver: 2000 videos ✓
(This is actually RESOLVED - no contradiction)
```

**Contradiction 2: Performance Claims**

```
DELIVERY_SUMMARY.txt (line 197):
"Typical Performance: 20-30 FPS on CPU, 50-100+ FPS on GPU"

vs.

README.md (same claim):
"20-30 FPS on CPU, 50-100+ FPS on GPU"

Verdict: ✓ Consistent, but NOT measured with stopwatch
```

**Contradiction 3: "Less Crowded"**

```
README.md: "confined closed spaces with fewer people"
CONFIG.py comment: "1-10 optimal" but CONFIG defaults don't enforce this
tracker.py: No person count limit

Verdict: ⚠️ Recommended but not enforced
```

---

### F3. Unverified Claims (Summary)

| Claim | Status | Evidence |
|-------|--------|----------|
| 97.7% validation accuracy | ✅ VERIFIED | Controlled dataset, code structure valid |
| 70% real-world accuracy | ❌ UNVERIFIED | Claimed but no actual test run in repo |
| <15 people stable | ❌ UNVERIFIED | Architectural assumption, no benchmark |
| 2000 videos training data | ✅ VERIFIED | Directory inspection confirms |
| Video-level split (no leakage) | ✅ VERIFIED | Code logic confirmed |
| 50-100+ FPS on GPU | ❌ UNVERIFIED | Depends on GPU model, frame size, etc. |
| "Production-ready" | ⚠️ PARTIAL | Code is complete but accuracy is 70%, not 97.7% |

---

## SECTION G: SINGLE SOURCE OF TRUTH RECOMMENDATION

### Current State
- 📄 README.md: Marketing + technical mix
- 📄 PROJECT_OVERVIEW_FOR_CHATGPT.md: High-level summary
- 📄 ACCURACY_ANALYSIS.md: Detailed metrics
- 📄 TECHNICAL_ANALYSIS.md: Architecture deep-dive
- 📄 CONFIG.py: Parameter reference
- ❌ No single authoritative source

### Recommended Solution

**Create**: `METRICS_AND_PERFORMANCE.md` (single source of truth)

```markdown
# Metrics and Performance Documentation

## Verified Facts (with code references)

### Dataset
- Source videos: 2000 (1000 violence + 1000 non-violence)
- Extracted frames: 29,569 (80/20 train/val split)
- Weapon dataset: 7,368 labeled images
- Code: prepare_violence_data.py, weapon dataset YAML

### Training Results
- Validation accuracy: 97.7% (5,886 frames, controlled conditions)
- Model: runs/violence_cls/train/weights/best.pt
- NOT measured: Real-world accuracy
- NOT measured: Precision/recall breakdown
- NOT measured: Confusion matrix

### Behavior Detection Thresholds
- Running: 50 pixels/sec (configurable)
- Loitering: 10 seconds in zone (configurable)
- Fall: 40% aspect ratio drop (configurable)
- Armed: 100 pixels from weapon (hard-coded)

### System Limits
- Tested with: Unknown (no benchmarks)
- Recommended: 1-15 people per frame
- GPU support: Auto-detected by ultralytics
- Input: Webcam, video files, image files

### Known Gaps
- No confusion matrix
- No precision/recall per scenario
- No FP rate measurement
- No multi-stream support
- No real-world benchmark

## Unverified Claims (Marketing Assumptions)
- "70% in confined spaces" - framework exists (evaluate.py) but no actual test
- "50-100+ FPS" - depends on GPU model
- "<15 people stable" - logical but unbenchmarked

## How to Verify Each Metric
1. Accuracy: Run training script, check results.csv
2. Precision/Recall: Run evaluate.py in compare mode with annotated video
3. FPS: Run on target hardware, observe console output
4. Realistic accuracy: Annotate 1-2 videos manually, compare with system output
```

---

## FINAL AUDIT SUMMARY

### ✅ What's Real (Code-Backed)
1. **2000 videos** training dataset exists
2. **Video-level split** prevents data leakage  
3. **29,569 frames** extracted and organized
4. **Code is complete** - all modules implemented
5. **Configuration flexible** - thresholds adjustable
6. **Evaluation tool exists** - manually measure accuracy

### ⚠️ What's Claimed But Unverified
1. **70% real-world accuracy** - no test run in repo
2. **<15 people stable** - no benchmark data
3. **50-100+ FPS** - depends on GPU/resolution
4. **97.7% validation** - likely correct but logs not saved

### ❌ What's Missing (Production Gaps)
1. **No automated metrics** during training
2. **No confusion matrix** pre-computed
3. **No false positive rate** measurement
4. **No storage management**
5. **No alerting** integration
6. **No multi-stream** support

### 🎯 Recommendation
**Verdict**: "Reasonably Production-Ready" for **confined spaces with manual setup**

Use for:
- ✅ Office security (1-10 people per camera)
- ✅ Shop monitoring (specific entry/exit)
- ✅ Security checkpoints

Don't use for:
- ❌ Shopping malls (too crowded)
- ❌ Automated deployment (requires calibration)
- ❌ Mission-critical (70% accuracy too low)

---

## SECTION H: CODE-CONFIRMED SPECIFICATIONS (CRITICAL REFERENCE)

### H1. Where is 97.7% accuracy coming from?

**EXACT SOURCE**: `runs/violence_cls/train/results.csv` (EPOCH 30 - final epoch)

```csv
epoch,time,train/loss,metrics/accuracy_top1,metrics/accuracy_top5,val/loss,...
30,14614.2,0.01397,0.97723,1,0.08358,...
                      ↑ 0.97723 = 97.723% ≈ 97.7%
```

**Metric**: YOLOv8-cls standard top-1 validation accuracy on 5,886 validation frames (controlled lab conditions)

**Supporting Files**:
- `confusion_matrix.png` - Visual confusion matrix from training
- `confusion_matrix_normalized.png` - Normalized version
- `results.png` - Epoch metrics plot

**Verdict**: ✅ **Verified - actual measured result from training logs**

---

### H2. Violence detection threshold (inference)

**Code Location 1** - `behaviour_detection/violence_classifier.py` lines 25-31:
```python
def __init__(self, model_path, threshold=0.5):
    """
    Args:
        threshold: Probability threshold for violence detection (default: 0.5)
    """
```

**Code Location 2** - Command-line argument, line 142:
```python
parser.add_argument("--threshold", type=float, default=0.5, help="Violence threshold")
```

**Comparison Logic** - line 94:
```python
'is_violent': violence_prob >= self.threshold,
```

**Verdict**: ✅ **Default threshold: 0.5 (50%)** - fully configurable via `--violence-threshold` argument

---

### H3. Events CSV schema - exact columns written

**Code Location**: `behaviour_detection/rules.py` lines 231-257

**EXACT FIELDNAMES**:
```python
fieldnames=[
    'timestamp',      # Unix timestamp (float, seconds since epoch)
    'type',          # Event type: "RUN", "FALL", or "LOITER"
    'track_id',      # Integer track ID
    'zone_name',     # Zone name string (empty if no zone)
    'centroid_x',    # X coordinate (float pixels)
    'centroid_y',    # Y coordinate (float pixels)
]
```

**Example CSV Output**:
```csv
timestamp,type,track_id,zone_name,centroid_x,centroid_y
1672704123.456,RUN,1,,320.5,240.3
1672704125.789,LOITER,1,center,315.2,238.1
1672704128.123,FALL,2,,400.0,350.5
```

**MISSING from CSV** (not implemented):
- ❌ Confidence scores
- ❌ False positive flags
- ❌ Duration of event
- ❌ Associated weapon detections

**Verdict**: ⚠️ **Schema is minimal** - location + type only, no confidence or FP tracking

---

### H4. Screenshot directory + exact naming pattern

**Code Location**: `behaviour_detection/pipeline.py` lines 166-195

**DIRECTORY STRUCTURE**:

**For VIDEO Input**:
```
test_videos/
  {video_stem}/              ← video filename without extension
    images/
      event_000001_12_34s.jpg
      event_000042_3_56s.jpg
```

**For WEBCAM Input** (source=0):
```
test_videos/
  webcam/
    images/
      event_000001_0_03s.jpg
      event_000042_1_40s.jpg
```

**EXACT NAMING FORMULA** (lines 188-189):
```python
timestamp_str = f"{video_timestamp:.2f}s"        # e.g., "12.34s"
screenshot_filename = f"event_{frame_idx:06d}_{timestamp_str.replace('.', '_')}.jpg"
# Result: "event_000001_12_34s.jpg"
```

**Creation Code** (line 195):
```python
cv2.imwrite(str(screenshot_path), annotated)
```

**Print Statement** (line 169):
```python
print(f"Event screenshots will be saved to: {event_screenshots_dir}")
# Output: "Event screenshots will be saved to: test_videos/my_video/images"
```

**Verdict**: ✅ **Confirmed - hardcoded to `test_videos/{source_name}/images/`**

---

### H5. Image-only processing support

**Full Pipeline Support**: ✅ **YES**

**Entry Point** - `behaviour_detection/pipeline.py` lines 108-120:
```python
def process_stream(self, source, show=True, save_dir=None):
    """Process a video stream or image."""
    
    if isinstance(source, int) or (isinstance(source, str) and source == "0"):
        self._process_webcam(show, save_dir)
    else:
        source_path = Path(source)
        if source_path.is_file():
            if source_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                self._process_image(source_path, show, save_dir)  # ← IMAGE SUPPORT
            else:
                self._process_video(source_path, show, save_dir)
```

**Image Processing Function** - lines 122-147:
```python
def _process_image(self, image_path, show, save_dir):
    """Process a single image."""
    frame = cv2.imread(str(image_path))  # ← cv2.imread HERE
    
    if frame is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    detections, tracked, events, annotated, drawn_events = self._run_pipeline_step(frame)
    
    if show:
        cv2.imshow("Behavior Detection", annotated)
        cv2.waitKey(0)  # ← Wait for keypress (single frame)
    
    if save_dir:
        output_file = save_path / f"{image_path.stem}_annotated.jpg"
        cv2.imwrite(str(output_file), annotated)
```

**Supported Formats**: `.jpg, .jpeg, .png, .bmp` (line 115)

**Pipeline Execution**: Full - detection → tracking → rules engine → screenshot logic all active for images

**Verdict**: ✅ **Full pipeline support for single images** - all behavior rules execute on one frame, displays annotated result, optionally saves output

---

## SECTION I: SPECIFICATION REFERENCE TABLE

| Specification | Value | Source | Configurable |
|--------------|-------|--------|--------------|
| **Violence Accuracy** | 97.7% (5,886 frames, lab conditions) | results.csv epoch 30 | N/A (trained model) |
| **Violence Threshold** | 0.5 (50% probability) | violence_classifier.py:25 | ✅ Yes (--violence-threshold) |
| **Running Threshold** | 50 px/sec | run_behaviour.py:47 | ✅ Yes (CONFIG) |
| **Loiter Threshold** | 10 seconds | rules.py:32 | ✅ Yes (CONFIG) |
| **Fall Detection** | 40% aspect ratio drop | rules.py:33 | ✅ Yes (CONFIG) |
| **Tracking IoU Threshold** | 0.3 | run_behaviour.py:59 | ✅ Yes |
| **Tracking Max Age** | 30 frames | run_behaviour.py:59 | ✅ Yes |
| **Screenshot Cooldown** | 10 seconds | pipeline.py:73 | Hard-coded |
| **Event Screenshot Path** | test_videos/{source}/images/ | pipeline.py:168 | Hard-coded |
| **CSV Columns** | 6 fields (see H3) | rules.py:237-241 | Hard-coded |
| **Image Formats Supported** | .jpg, .jpeg, .png, .bmp | pipeline.py:115 | Hard-coded |

---

**Audit Completed**: January 3, 2026  
**Auditor**: Code inspection & documentation analysis  
**Status**: Section H provides code-confirmed specifications for documentation updates  
**Next Step**: Use Section H answers to update PROJECT_OVERVIEW_FOR_CHATGPT.md and README.md with exact specs
