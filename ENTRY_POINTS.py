"""
ENTRY POINTS AND HOW TO RUN THE PROJECT

This file describes all the ways to run and interact with this project.
"""

# =============================================================================
# STARTUP SCRIPTS (Recommended for first-time users)
# =============================================================================

"""
Windows (Batch File):
    startup.bat
    - Interactive menu
    - Automatic setup and installation
    - Option to test both phases
    - Just double-click to run!

Windows (PowerShell):
    powershell -ExecutionPolicy Bypass -File startup.ps1
    - Same as .bat but in PowerShell
    - More flexible scripting
    - Colored output

Linux/macOS:
    bash startup.sh
    - Automatic setup on Unix-like systems
    - Create this file if needed
"""

# =============================================================================
# COMMAND-LINE ENTRY POINTS
# =============================================================================

# Phase 1: Object Detection
"""
PHASE 1 - YOLO Object Detection

Single image detection:
    python yolo_object_detection/main.py --source image.jpg --show

Video file detection:
    python yolo_object_detection/main.py --source video.mp4 --show --save-dir runs/video

Webcam detection:
    python yolo_object_detection/main.py --source 0 --show

Batch folder processing:
    python yolo_object_detection/main.py --source images/ --save-dir runs/batch

With custom confidence:
    python yolo_object_detection/main.py --source 0 --conf 0.3 --show

Arguments:
    --source: 0 for webcam, path to image/video/folder
    --conf: Confidence threshold (default 0.5)
    --show: Display frames in window
    --save-dir: Directory to save annotated outputs
"""

# Phase 2: Behavior Detection
"""
PHASE 2 - Behavior Detection

Webcam with behavior detection:
    python run_behaviour.py --source 0 --show

Video file analysis:
    python run_behaviour.py --source video.mp4 --show --events-csv results.csv

Webcam with event logging:
    python run_behaviour.py --source 0 --show --events-csv events.csv --save-dir output

Video analysis with saved frames:
    python run_behaviour.py --source video.mp4 --show --save-dir annotated_frames --events-csv events.csv

Arguments:
    --source: 0 for webcam, path to video/image (required)
    --conf: Detection confidence (default 0.5)
    --show: Display frames
    --save-dir: Save annotated frames
    --events-csv: Save detected events to CSV
"""

# =============================================================================
# VALIDATION AND TESTING
# =============================================================================

"""
Validation:
    python validate_project.py
    - Validates all imports
    - Tests basic functionality
    - Verifies project setup

Unit Tests:
    python -m unittest test_behavior_detection -v
    - Runs all unit tests
    - Tests tracker, features, rules
    - Tests integration

Specific test class:
    python -m unittest test_behavior_detection.TestTracker -v
    python -m unittest test_behavior_detection.TestFeatures -v
    python -m unittest test_behavior_detection.TestRulesEngine -v

Specific test method:
    python -m unittest test_behavior_detection.TestTracker.test_track_creation -v
"""

# =============================================================================
# DOCUMENTATION VIEWERS
# =============================================================================

"""
View documentation:
    python QUICKSTART.py
    python PROJECT_SUMMARY.py
    python CONFIG.py
    python INDEX.py
    python VERIFICATION_CHECKLIST.py

Read files directly:
    cat README.md (Linux/macOS)
    type README.md (Windows)
    less README.md (Less pager)
    more README.md (More pager)
    python -c "import webbrowser; webbrowser.open('README.md')" (with proper formatting)
"""

# =============================================================================
# PYTHON API ENTRY POINTS
# =============================================================================

"""
Direct Python API usage:

Phase 1 - Object Detection:
    from yolo_object_detection.detectors import YoloDetector
    from yolo_object_detection.utils import FPSMeter, draw_detections
    
    detector = YoloDetector(confidence_threshold=0.5)
    fps_meter = FPSMeter()
    
    detections, annotated_frame = detector.run_detection(frame)
    fps_meter.update()
    annotated = draw_detections(annotated_frame, detections)

Phase 2 - Behavior Detection:
    from behaviour_detection.pipeline import BehaviourPipeline
    from behaviour_detection.tracker import Tracker
    from behaviour_detection.rules import RulesEngine
    from yolo_object_detection.detectors import YoloDetector
    
    detector = YoloDetector()
    tracker = Tracker()
    rules = RulesEngine(zones={"zone1": (100, 100, 300, 300)})
    
    pipeline = BehaviourPipeline(detector, tracker, rules)
    pipeline.process_stream("video.mp4", show=True, save_dir="output")
    
    events = rules.get_all_events()
    for event in events:
        print(f"{event['type']}: {event['track_id']} at {event['timestamp']}")

See behaviour_detection/pipeline.py for full API documentation.
"""

# =============================================================================
# CUSTOMIZATION ENTRY POINTS
# =============================================================================

"""
Customize behavior detection:
    1. Edit CONFIG.py for quick changes:
       - RUN_SPEED_THRESHOLD = 100.0 (was 150)
       - LOITER_TIME_THRESHOLD = 5.0 (was 10)
       - Add zones: zones["my_zone"] = (x1, y1, x2, y2)
    
    2. Edit run_behaviour.py for zone configuration:
       - Modify zones dictionary
       - Adjust cfg thresholds
    
    3. Edit behaviour_detection/rules.py for new behaviors:
       - Add new _check_* methods
       - Implement custom logic

Customize object detection:
    1. Edit yolo_object_detection/detectors.py:
       - Change model: YoloDetector("yolov8m.pt")
       - Modify detection processing
    
    2. Edit yolo_object_detection/utils.py:
       - Change drawing colors/styles
       - Modify FPS meter behavior

Customize tracker:
    1. Edit behaviour_detection/tracker.py:
       - Change IoU threshold: Tracker(iou_threshold=0.5)
       - Modify association logic
       - Change track persistence: Tracker(max_age=60)
"""

# =============================================================================
# DEVELOPMENT AND DEBUGGING
# =============================================================================

"""
Debug a specific component:
    python -c "
    from behaviour_detection.tracker import Tracker
    t = Tracker()
    detections = [[0, 0, 100, 100, 0.9, 0]]
    tracks = t.update(detections)
    print(tracks)
    "

Test a specific function:
    python -c "
    from behaviour_detection.features import compute_speed
    speed = compute_speed((0, 0), (100, 0), 1.0)
    print(f'Speed: {speed} px/sec')
    "

Profile performance:
    python -m cProfile -s cumulative run_behaviour.py --source 0 --show

Memory usage:
    python -m memory_profiler run_behaviour.py --source 0

Interactive debugging:
    python -i run_behaviour.py  # Drops to Python REPL after exit
"""

# =============================================================================
# BATCH PROCESSING
# =============================================================================

"""
Process multiple videos:
    for video in videos/*.mp4; do
        python run_behaviour.py --source "$video" --events-csv "results/${video%.mp4}.csv"
    done

Process all videos in folder (Windows PowerShell):
    Get-ChildItem "videos" -Filter "*.mp4" | ForEach-Object {
        python run_behaviour.py --source $_.FullName --events-csv "results\\$($_.BaseName).csv"
    }

Extract all events to single file:
    python -c "
    import glob
    import csv
    import json
    
    all_events = []
    for csv_file in glob.glob('results/*.csv'):
        with open(csv_file) as f:
            all_events.extend(csv.DictReader(f))
    
    with open('all_events.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=all_events[0].keys())
        writer.writeheader()
        writer.writerows(all_events)
    "
"""

# =============================================================================
# PERFORMANCE TUNING
# =============================================================================

"""
For real-time webcam (optimize for speed):
    python run_behaviour.py --source 0 --show --conf 0.4
    
For batch video processing (optimize for accuracy):
    python run_behaviour.py --source video.mp4 --conf 0.7 --save-dir output
    
For headless server (no display):
    python run_behaviour.py --source video.mp4 --events-csv events.csv

With GPU acceleration (if available):
    - Will auto-detect and use GPU
    - No changes needed to code
    - ~2-5x faster than CPU
"""

# =============================================================================
# ERROR SCENARIOS
# =============================================================================

"""
If you get "ModuleNotFoundError":
    pip install -r requirements.txt

If webcam not found:
    python run_behaviour.py --source 1 --show  (try camera index 1, 2, etc.)
    
If very slow FPS:
    python run_behaviour.py --source 0 --conf 0.3  (lower confidence)
    
If YOLO model fails to download:
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
    
If display is unavailable:
    Don't use --show, just save to CSV
    python run_behaviour.py --source video.mp4 --events-csv events.csv
"""

# =============================================================================
# DEPLOYMENT
# =============================================================================

"""
Deploy as Docker container:
    1. Create Dockerfile with Python 3.10+
    2. Copy project files
    3. Run: pip install -r requirements.txt
    4. ENTRYPOINT: python run_behaviour.py --source 0 --events-csv events.csv

Deploy as Windows Service:
    - Use NSSM (Non-Sucking Service Manager)
    - Configure to run: python run_behaviour.py --source 0

Deploy as systemd service (Linux):
    - Create .service file
    - Enable and start with systemctl

Deploy as background job:
    python run_behaviour.py --source video.mp4 --events-csv events.csv &

Deploy with monitoring:
    - Run in screen/tmux session
    - Monitor events.csv for anomalies
    - Alert on specific behaviors
"""

# =============================================================================
# QUICK REFERENCE
# =============================================================================

print("""
QUICK START (30 seconds):

1. Install:
   pip install -r requirements.txt

2. Run on webcam:
   python run_behaviour.py --source 0 --show

3. See the behaviors detected in real-time!

DOCUMENTATION:
   - README.md: Full reference
   - QUICKSTART.py: Usage examples
   - CONFIG.py: Configuration

HELP:
   python -c "import run_behaviour; help()"
""")
