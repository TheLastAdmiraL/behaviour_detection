"""Utility functions for YOLO detection: drawing, FPS calculation, and CLI parsing."""

import argparse
import time
import cv2
import numpy as np
from pathlib import Path


class FPSMeter:
    """Calculate and display smoothed FPS."""
    
    def __init__(self, smoothing_factor=0.9):
        """
        Initialize FPS meter.
        
        Args:
            smoothing_factor: Exponential smoothing factor (0-1). Higher = smoother but slower response.
        """
        self.smoothing_factor = smoothing_factor
        self.fps = 0.0
        self.prev_time = time.time()
    
    def update(self):
        """Update FPS based on elapsed time since last call."""
        current_time = time.time()
        dt = current_time - self.prev_time
        self.prev_time = current_time
        
        if dt > 0:
            instantaneous_fps = 1.0 / dt
            self.fps = (self.smoothing_factor * self.fps + 
                       (1 - self.smoothing_factor) * instantaneous_fps)
        
        return self.fps
    
    def get_fps(self):
        """Get current smoothed FPS."""
        return self.fps


def draw_detections(frame, detections):
    """
    Draw bounding boxes, class labels, and confidence scores on frame.
    
    Args:
        frame: Input frame (BGR image)
        detections: List of detections, each as (x1, y1, x2, y2, conf, class_id, class_name)
    
    Returns:
        Annotated frame
    """
    annotated = frame.copy()
    
    for detection in detections:
        x1, y1, x2, y2, conf, class_id, class_name = detection
        
        # Convert to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Draw bounding box
        color = (0, 255, 0)  # Green
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label
        label = f"{class_name} {conf:.2f}"
        
        # Draw label background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        
        label_x = x1
        label_y = y1 - 10
        bg_x1 = label_x
        bg_y1 = label_y - text_size[1] - 5
        bg_x2 = label_x + text_size[0] + 5
        bg_y2 = label_y + 5
        
        cv2.rectangle(annotated, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        cv2.putText(annotated, label, (label_x, label_y), font, font_scale, 
                   (0, 0, 0), thickness)
    
    return annotated


def draw_fps(frame, fps_meter):
    """
    Draw FPS counter on frame.
    
    Args:
        frame: Input frame
        fps_meter: FPSMeter instance
    
    Returns:
        Frame with FPS drawn
    """
    fps = fps_meter.get_fps()
    fps_text = f"FPS: {fps:.1f}"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color = (0, 255, 0)
    
    text_size = cv2.getTextSize(fps_text, font, font_scale, thickness)[0]
    x = frame.shape[1] - text_size[0] - 10
    y = 30
    
    # Background
    cv2.rectangle(frame, (x - 5, y - text_size[1] - 5), 
                  (x + text_size[0] + 5, y + 5), (0, 0, 0), -1)
    cv2.putText(frame, fps_text, (x, y), font, font_scale, color, thickness)
    
    return frame


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-time YOLOv8 object detection on images, videos, and webcam."
    )
    
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to image, video, folder, or 0 for webcam."
    )
    
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for detections (default: 0.5)."
    )
    
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display annotated frames in a window. Press 'q' to quit."
    )
    
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save annotated outputs."
    )
    
    return parser.parse_args()


def get_file_type(path_str):
    """
    Determine the type of input.
    
    Returns:
        'webcam', 'image', 'video', 'folder', or None
    """
    if path_str == "0" or path_str == 0:
        return "webcam"
    
    path = Path(path_str)
    
    if not path.exists():
        return None
    
    if path.is_file():
        suffix = path.suffix.lower()
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}
        
        if suffix in image_extensions:
            return "image"
        elif suffix in video_extensions:
            return "video"
        else:
            return None
    
    if path.is_dir():
        return "folder"
    
    return None


def is_valid_source(source):
    """Check if source is valid."""
    file_type = get_file_type(source)
    return file_type is not None
