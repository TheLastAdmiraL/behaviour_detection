"""
QUICK START GUIDE - Behavior Detection System

=== FIRST TIME SETUP ===

1. Open PowerShell in the project directory
2. Run the setup commands from INSTALLATION.txt

3. Once setup is complete, test with:
   python run_behaviour.py --source 0 --show

=== PHASE 1: OBJECT DETECTION EXAMPLES ===

Example 1: Detect objects in a single image
    python yolo_object_detection/main.py --source path/to/image.jpg --show

Example 2: Detect objects in a video and save results
    python yolo_object_detection/main.py --source video.mp4 --show --save-dir runs/video

Example 3: Real-time webcam detection
    python yolo_object_detection/main.py --source 0 --show

Example 4: Process entire folder
    python yolo_object_detection/main.py --source images/ --save-dir runs/batch

Example 5: Lower confidence threshold (more detections)
    python yolo_object_detection/main.py --source 0 --conf 0.3 --show

=== PHASE 2: BEHAVIOR DETECTION EXAMPLES ===

Example 1: Real-time webcam with behavior analysis
    python run_behaviour.py --source 0 --show

Example 2: Analyze video and save behavior events
    python run_behaviour.py --source video.mp4 --events-csv results/events.csv --save-dir runs/behavior

Example 3: Analyze video without display (faster)
    python run_behaviour.py --source video.mp4 --events-csv results/events.csv

Example 4: Save annotated frames for inspection
    python run_behaviour.py --source video.mp4 --show --save-dir runs/annotated

=== UNDERSTANDING THE OUTPUT ===

Bounding Boxes and Track IDs:
    - GREEN box with "ID X" = person being tracked

Behavior Labels:
    - RED box with "RUNNING" = person moving fast (>150 px/sec)
    - BLUE box with "LOITER" = person staying in zone too long (>10 sec)
    - ORANGE box with "FALL" = person has fallen

Event CSV Format:
    timestamp,type,track_id,zone_name,centroid_x,centroid_y
    
Rows are: timestamp (seconds), behavior type, person ID, zone name, position

=== CUSTOMIZATION ===

To change behavior thresholds, edit run_behaviour.py around line 45:

    cfg = {
        "RUN_SPEED_THRESHOLD": 150.0,          # Higher = less sensitive
        "LOITER_TIME_THRESHOLD": 10.0,         # Higher = need to stay longer
        "FALL_VERTICAL_RATIO_DROP": 0.4,       # Lower = more sensitive to falls
    }

To add custom loitering zones, edit run_behaviour.py around line 45:

    zones = {
        "entrance": (0, 0, 200, 480),          # Left side
        "exit": (400, 0, 640, 480),            # Right side
    }

=== TROUBLESHOOTING ===

Q: "ModuleNotFoundError: No module named 'scipy'"
A: Run: pip install -r requirements.txt

Q: "ImportError: libGL.so.1"
A: On Linux with no display, add to run_behaviour.py:
   import os
   os.environ['DISPLAY'] = ''
   Then use --show flag with caution

Q: Very slow FPS
A: - Lower confidence threshold (--conf 0.3)
   - Use GPU if available
   - Process at lower resolution

Q: Webcam not found
A: Try: python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
   If False, check camera permissions

Q: Track IDs changing rapidly
A: Increase IoU threshold in tracker.py (line ~90)
   Change: iou_threshold=0.5  (was 0.3)

=== PERFORMANCE TIPS ===

For real-time webcam (30 FPS target):
- Use confidence=0.5 (default)
- Don't save frames to disk (slower)
- Keep display window small

For video processing (speed not critical):
- Lower confidence (0.3) for better detection
- Save frames to disk for later review
- Don't display (use --no-show)

For long duration monitoring:
- Increase max_age in Tracker (persistence)
- Use larger loitering zones
- Increase thresholds (less false positives)

=== PROGRAMMATIC USE ===

See behaviour_detection/pipeline.py for the API:

from behaviour_detection.pipeline import BehaviourPipeline
from yolo_object_detection.detectors import YoloDetector

detector = YoloDetector(confidence_threshold=0.5)
pipeline = BehaviourPipeline(detector=detector)
pipeline.process_stream("video.mp4", show=True, save_dir="output")

=== FILES TO KNOW ===

Key files:
- run_behaviour.py ............ Main behavior detection CLI
- yolo_object_detection/main.py ... Object detection CLI
- behaviour_detection/rules.py ...... Behavior rules (edit to customize)
- behaviour_detection/tracker.py .... Tracking algorithm
- requirements.txt ............. Python dependencies

Output directories:
- runs/                  ........ Default output folder
- runs/events.csv       ........ Behavior event log

=== NEXT STEPS ===

1. Test with webcam: python run_behaviour.py --source 0 --show
2. Record a short test video
3. Analyze the video: python run_behaviour.py --source test.mp4 --events-csv results.csv --show
4. Review events.csv to understand the format
5. Customize thresholds in run_behaviour.py based on your needs
6. Deploy to your use case!

=== SUPPORT ===

See README.md for full documentation
Check test_behavior_detection.py for unit test examples
Review behaviour_detection/pipeline.py for advanced usage
"""

print(__doc__)
