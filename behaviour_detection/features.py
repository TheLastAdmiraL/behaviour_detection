"""Feature extraction for behavior detection."""

import math
import numpy as np


def compute_speed(prev_centroid, curr_centroid, dt):
    """
    Compute speed (pixels per second) between two centroids.
    
    Args:
        prev_centroid: Previous position (x, y)
        curr_centroid: Current position (x, y)
        dt: Time elapsed in seconds
    
    Returns:
        Speed in pixels/second
    """
    if dt <= 0:
        return 0.0
    
    prev_x, prev_y = prev_centroid
    curr_x, curr_y = curr_centroid
    
    distance = math.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
    speed = distance / dt
    
    return speed


def compute_velocity_vector(prev_centroid, curr_centroid):
    """
    Compute velocity vector (direction and magnitude).
    
    Args:
        prev_centroid: Previous position (x, y)
        curr_centroid: Current position (x, y)
    
    Returns:
        Tuple: (dx, dy, magnitude)
    """
    prev_x, prev_y = prev_centroid
    curr_x, curr_y = curr_centroid
    
    dx = curr_x - prev_x
    dy = curr_y - prev_y
    magnitude = math.sqrt(dx ** 2 + dy ** 2)
    
    return dx, dy, magnitude


def get_bbox_aspect_ratio(bbox):
    """
    Get aspect ratio of bounding box (height / width).
    
    Args:
        bbox: Bounding box (x1, y1, x2, y2)
    
    Returns:
        Aspect ratio
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    
    if width <= 0:
        return 0.0
    
    return height / width


def is_vertical_to_horizontal_change(prev_bbox, curr_bbox, ratio_threshold=0.5):
    """
    Detect if a person has changed from vertical (standing) to horizontal (lying).
    
    This is used as a heuristic for fall detection.
    
    Args:
        prev_bbox: Previous bounding box
        curr_bbox: Current bounding box
        ratio_threshold: Threshold for aspect ratio drop
    
    Returns:
        Boolean indicating if a fall-like change occurred
    """
    prev_aspect = get_bbox_aspect_ratio(prev_bbox)
    curr_aspect = get_bbox_aspect_ratio(curr_bbox)
    
    if prev_aspect <= 0 or curr_aspect <= 0:
        return False
    
    # Check if aspect ratio dropped significantly (tall to short)
    ratio_change = curr_aspect / prev_aspect if prev_aspect > 0 else 1.0
    
    return ratio_change < ratio_threshold


def is_moving_downward(prev_centroid, curr_centroid, min_distance=5):
    """
    Check if centroid is moving downward significantly.
    
    Args:
        prev_centroid: Previous position (x, y)
        curr_centroid: Current position (x, y)
        min_distance: Minimum pixels to move
    
    Returns:
        Boolean
    """
    prev_x, prev_y = prev_centroid
    curr_x, curr_y = curr_centroid
    
    dy = curr_y - prev_y
    return dy > min_distance


def get_centroid_displacement(prev_centroid, curr_centroid):
    """
    Get displacement vector from previous to current centroid.
    
    Args:
        prev_centroid: Previous position (x, y)
        curr_centroid: Current position (x, y)
    
    Returns:
        Displacement (dx, dy)
    """
    dx = curr_centroid[0] - prev_centroid[0]
    dy = curr_centroid[1] - prev_centroid[1]
    return dx, dy


def is_point_in_zone(point, zone_rect):
    """
    Check if a point is inside a rectangular zone.
    
    Args:
        point: (x, y) coordinate
        zone_rect: Zone as (x1, y1, x2, y2)
    
    Returns:
        Boolean
    """
    x, y = point
    x1, y1, x2, y2 = zone_rect
    
    return x1 <= x <= x2 and y1 <= y <= y2


def compute_distance_to_zone(centroid, zone_rect):
    """
    Compute minimum distance from centroid to zone.
    
    Args:
        centroid: (x, y) coordinate
        zone_rect: Zone as (x1, y1, x2, y2)
    
    Returns:
        Minimum distance in pixels
    """
    x, y = centroid
    x1, y1, x2, y2 = zone_rect
    
    # Closest point in zone
    closest_x = max(x1, min(x, x2))
    closest_y = max(y1, min(y, y2))
    
    distance = math.sqrt((x - closest_x) ** 2 + (y - closest_y) ** 2)
    
    return distance


class MotionHistory:
    """Maintains motion history for a tracked object."""
    
    def __init__(self, max_history=30):
        """
        Initialize motion history.
        
        Args:
            max_history: Maximum number of frames to keep in history
        """
        self.max_history = max_history
        self.centroids = []
        self.timestamps = []
        self.speeds = []
    
    def add_frame(self, centroid, timestamp):
        """Add a new frame to the history."""
        self.centroids.append(centroid)
        self.timestamps.append(timestamp)
        
        # Keep only recent history
        if len(self.centroids) > self.max_history:
            self.centroids.pop(0)
            self.timestamps.pop(0)
            self.speeds.pop(0)
        
        # Compute speed if we have previous frame
        if len(self.centroids) >= 2:
            dt = self.timestamps[-1] - self.timestamps[-2]
            speed = compute_speed(self.centroids[-2], self.centroids[-1], dt)
            self.speeds.append(speed)
        
        # Pad speeds list
        if len(self.speeds) < len(self.centroids):
            self.speeds.insert(0, 0.0)
    
    def get_average_speed(self, window=5):
        """Get average speed over last N frames."""
        if not self.speeds:
            return 0.0
        
        window = min(window, len(self.speeds))
        return np.mean(self.speeds[-window:]) if window > 0 else 0.0
    
    def get_max_speed(self, window=5):
        """Get max speed over last N frames."""
        if not self.speeds:
            return 0.0
        
        window = min(window, len(self.speeds))
        return np.max(self.speeds[-window:]) if window > 0 else 0.0
    
    def get_instant_speed(self):
        """Get speed from last frame."""
        return self.speeds[-1] if self.speeds else 0.0
    
    def get_time_in_zone(self, zone_rect):
        """
        Get cumulative time spent in zone.
        
        Args:
            zone_rect: Zone as (x1, y1, x2, y2)
        
        Returns:
            Time in seconds
        """
        if len(self.centroids) < 2:
            return 0.0
        
        time_in_zone = 0.0
        for i, centroid in enumerate(self.centroids):
            if is_point_in_zone(centroid, zone_rect):
                if i > 0:
                    dt = self.timestamps[i] - self.timestamps[i - 1]
                    time_in_zone += dt
        
        return time_in_zone
