"""
Configuration settings for behavior detection system.

Modify this file to customize detection parameters for your use case.
"""

# ============================================================================
# DETECTION PARAMETERS
# ============================================================================

# Confidence threshold for YOLOv8 detector (0.0 to 1.0)
# Higher = more confident detections, fewer false positives, may miss objects
# Lower = more detections, more false positives, slower processing
DETECTION_CONFIDENCE = 0.5

# ============================================================================
# BEHAVIOR DETECTION THRESHOLDS
# ============================================================================

# Running: Speed above this threshold (pixels per second)
# Typical values: 80-300 depending on camera distance
RUN_SPEED_THRESHOLD = 150.0

# Loitering: Time in zone above this threshold (seconds)
# Typical values: 5-30 depending on use case
LOITER_TIME_THRESHOLD = 10.0

# Loitering: Speed must be below this to count as loitering (pixels/sec)
# Typical values: 30-100
LOITER_SPEED_THRESHOLD = 50.0

# Fall Detection: Aspect ratio drop threshold (0.0 to 1.0)
# Ratio = new_height/prev_height. Below this = fall detected
# Typical values: 0.3-0.6
FALL_VERTICAL_RATIO_DROP = 0.4

# Fall Detection: Minimum downward centroid movement (pixels)
# Typical values: 10-50
FALL_DOWNWARD_DISTANCE = 20.0

# Number of frames to average for running detection
RUN_WINDOW_FRAMES = 5

# ============================================================================
# TRACKER PARAMETERS
# ============================================================================

# Maximum frames a track can exist without a detection
# Higher = more persistent tracking, may accumulate errors
# Lower = cleaner tracks, more ID switches
TRACKER_MAX_AGE = 30

# IoU threshold for associating detections to tracks (0.0 to 1.0)
# Higher = stricter matching, more ID switches
# Lower = looser matching, may merge different objects
TRACKER_IOU_THRESHOLD = 0.3

# ============================================================================
# LOITERING ZONES
# ============================================================================

# Define rectangular zones where loitering is detected
# Format: "zone_name": (x1, y1, x2, y2) in pixels
# Origin (0, 0) is top-left, x increases right, y increases down
# 
# To define zones:
# 1. Note your frame dimensions (usually 640x480 or 1920x1080)
# 2. Identify areas of interest
# 3. Define rectangles for each area

ZONES = {
    # Center zone - 400x400 in the middle
    "center": (120, 40, 520, 440),
    
    # Uncomment and customize for your use case:
    # "entrance": (0, 0, 200, 480),          # Left side
    # "exit": (400, 0, 640, 480),            # Right side
    # "checkout_counter": (300, 200, 500, 400),  # Specific area
}

# ============================================================================
# OUTPUT SETTINGS
# ============================================================================

# Default confidence for Phase 1 (object detection)
DEFAULT_CONFIDENCE = 0.5

# Show FPS counter on annotated frames
SHOW_FPS = True

# Draw tracking information
DRAW_TRACK_IDS = True
DRAW_TRACK_CENTROIDS = True

# ============================================================================
# PERFORMANCE TUNING
# ============================================================================

# For faster processing (lower quality):
FAST_MODE = False
# When True: Use smaller model, lower confidence, skip some checks

# For GPU acceleration (if available):
USE_GPU = True

# Frame skip for video processing (1 = process all, 2 = process every 2nd)
FRAME_SKIP = 1

# ============================================================================
# VALIDATION AND USAGE NOTES
# ============================================================================

"""
CALIBRATING THRESHOLDS FOR YOUR SCENARIO:

1. RUNNING_SPEED_THRESHOLD
   - Record a video of people walking, jogging, running
   - Run detection and check FPS output
   - Walk = ~50-100 px/sec, Jog = ~150-200 px/sec, Run = >200 px/sec
   - Adjust based on your camera angle and distance

2. LOITER_TIME_THRESHOLD & SPEED
   - Adjust time based on what you consider "loitering"
   - Speed threshold filters out slow walking
   - Consider using different thresholds for different zones

3. FALL DETECTION
   - Test with actual fall videos
   - Adjust VERTICAL_RATIO_DROP if missing falls
   - Increase DOWNWARD_DISTANCE for false positive filtering

4. ZONES
   - Run in --show mode to see frame coordinates
   - Use Python to calculate zone coordinates:
     x1 = left edge, y1 = top edge
     x2 = right edge, y2 = bottom edge
   - Test zones in --show mode before production deployment

EXAMPLE ZONE CALCULATIONS for 640x480 frame:
  - Full frame: (0, 0, 640, 480)
  - Center 200x200: (220, 140, 420, 340)
  - Left half: (0, 0, 320, 480)
  - Top half: (0, 0, 640, 240)
"""

# ============================================================================
# ADVANCED SETTINGS
# ============================================================================

# Motion history buffer size per track
MOTION_HISTORY_SIZE = 30

# Enable detailed logging
DEBUG_MODE = False

# Save detailed tracking statistics
SAVE_TRACKING_STATS = False

# ============================================================================
# EXPERIMENT WITH THESE VALUES
# ============================================================================

# If you're getting too many RUN detections, increase:
# RUN_SPEED_THRESHOLD = 200.0

# If you're missing RUNNING people, decrease:
# RUN_SPEED_THRESHOLD = 100.0

# If LOITER events not triggered, decrease time:
# LOITER_TIME_THRESHOLD = 5.0

# If too many LOITER false positives, increase speed threshold:
# LOITER_SPEED_THRESHOLD = 100.0

# If FALL not detected, decrease ratio drop:
# FALL_VERTICAL_RATIO_DROP = 0.3

# If FALL too sensitive, increase ratio drop or downward distance:
# FALL_VERTICAL_RATIO_DROP = 0.5
# FALL_DOWNWARD_DISTANCE = 30.0
