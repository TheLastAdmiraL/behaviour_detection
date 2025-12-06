"""Behavior rules engine for detecting running, loitering, and falls."""

import time
from collections import defaultdict
import csv
from pathlib import Path

from behaviour_detection.features import (
    MotionHistory, compute_speed, is_vertical_to_horizontal_change,
    is_moving_downward, is_point_in_zone
)


class RulesEngine:
    """Rules engine for detecting behaviors."""
    
    def __init__(self, zones=None, cfg=None):
        """
        Initialize rules engine.
        
        Args:
            zones: Dictionary of zone_name -> (x1, y1, x2, y2)
            cfg: Configuration dictionary with thresholds
        """
        self.zones = zones or {}
        
        # Default configuration
        self.cfg = {
            "RUN_SPEED_THRESHOLD": 150.0,  # pixels/sec
            "LOITER_TIME_THRESHOLD": 10.0,  # seconds
            "LOITER_SPEED_THRESHOLD": 50.0,  # pixels/sec (must be below this)
            "FALL_VERTICAL_RATIO_DROP": 0.4,  # aspect ratio drop threshold
            "FALL_DOWNWARD_DISTANCE": 20.0,  # pixels
            "RUN_WINDOW_FRAMES": 5,  # frames to check for running
        }
        
        if cfg:
            self.cfg.update(cfg)
        
        # Per-track state
        self.track_history = defaultdict(MotionHistory)
        self.track_loiter_time = defaultdict(float)
        self.track_loiter_zone = defaultdict(str)
        self.track_in_zone = defaultdict(bool)
        self.track_creation_time = {}
        self.track_last_event_time = defaultdict(float)
        self.track_last_bbox = {}
        self.track_previous_frames_data = defaultdict(list)
        
        # Event log
        self.events = []
    
    def step(self, tracks, dt):
        """
        Process tracks and detect behaviors.
        
        Args:
            tracks: List of track dictionaries
            dt: Time elapsed since last step (seconds)
        
        Returns:
            List of event dictionaries
        """
        current_time = time.time()
        events = []
        
        # Update track histories and get active track IDs
        active_ids = set()
        for track in tracks:
            track_id = track["id"]
            active_ids.add(track_id)
            centroid = track["centroid"]
            
            self.track_history[track_id].add_frame(centroid, current_time)
            self.track_creation_time.setdefault(track_id, current_time)
            
            # Store previous data for fall detection
            if "bbox" in track:
                if track_id not in self.track_last_bbox:
                    self.track_last_bbox[track_id] = track["bbox"]
        
        # Check each track for behaviors
        for track in tracks:
            track_id = track["id"]
            
            # Check for running
            if self._check_running(track_id):
                event = {
                    "type": "RUN",
                    "track_id": track_id,
                    "zone_name": None,
                    "timestamp": current_time,
                    "centroid": track["centroid"],
                    "speed": self.track_history[track_id].get_instant_speed(),
                }
                events.append(event)
                self.events.append(event)
            
            # Check for fall
            if self._check_fall(track_id, track):
                event = {
                    "type": "FALL",
                    "track_id": track_id,
                    "zone_name": None,
                    "timestamp": current_time,
                    "centroid": track["centroid"],
                }
                events.append(event)
                self.events.append(event)
                # Update last bbox after fall detection
                self.track_last_bbox[track_id] = track.get("bbox", self.track_last_bbox.get(track_id))
            
            # Check for loitering in zones
            for zone_name, zone_rect in self.zones.items():
                if self._check_loitering(track_id, track, zone_rect, zone_name, current_time):
                    event = {
                        "type": "LOITER",
                        "track_id": track_id,
                        "zone_name": zone_name,
                        "timestamp": current_time,
                        "centroid": track["centroid"],
                        "time_in_zone": self.track_history[track_id].get_time_in_zone(zone_rect),
                    }
                    events.append(event)
                    self.events.append(event)
        
        # Clean up history for lost tracks
        for track_id in list(self.track_history.keys()):
            if track_id not in active_ids:
                # Keep for a few frames in case track reappears
                if track_id in self.track_creation_time:
                    time_since_last_seen = current_time - self.track_history[track_id].timestamps[-1] if self.track_history[track_id].timestamps else 0
                    if time_since_last_seen > 5.0:  # Clean up after 5 seconds
                        del self.track_history[track_id]
                        self.track_loiter_time[track_id] = 0.0
                        self.track_loiter_zone[track_id] = ""
        
        return events
    
    def _check_running(self, track_id):
        """Check if a track is running."""
        history = self.track_history.get(track_id)
        if not history or not history.speeds:
            return False
        
        # Check if average speed is above threshold
        avg_speed = history.get_average_speed(self.cfg["RUN_WINDOW_FRAMES"])
        return avg_speed > self.cfg["RUN_SPEED_THRESHOLD"]
    
    def _check_fall(self, track_id, track):
        """Check if a track has fallen."""
        history = self.track_history.get(track_id)
        if not history or len(history.centroids) < 2:
            return False
        
        curr_bbox = track.get("bbox")
        prev_bbox = self.track_last_bbox.get(track_id)
        
        if not curr_bbox or not prev_bbox:
            return False
        
        # Check 1: Aspect ratio change from tall to wide
        if not is_vertical_to_horizontal_change(prev_bbox, curr_bbox, 
                                                 self.cfg["FALL_VERTICAL_RATIO_DROP"]):
            return False
        
        # Check 2: Centroid moving downward
        prev_centroid = history.centroids[-2]
        curr_centroid = history.centroids[-1]
        
        if not is_moving_downward(prev_centroid, curr_centroid, 
                                   self.cfg["FALL_DOWNWARD_DISTANCE"]):
            return False
        
        return True
    
    def _check_loitering(self, track_id, track, zone_rect, zone_name, current_time):
        """Check if a track is loitering in a zone."""
        centroid = track["centroid"]
        history = self.track_history.get(track_id)
        
        if not history:
            return False
        
        # Check if in zone
        in_zone = is_point_in_zone(centroid, zone_rect)
        
        if not in_zone:
            # Left zone, reset timer
            self.track_loiter_time[track_id] = 0.0
            self.track_loiter_zone[track_id] = ""
            self.track_in_zone[track_id] = False
            return False
        
        # Check speed while in zone
        speed = history.get_instant_speed()
        if speed > self.cfg["LOITER_SPEED_THRESHOLD"]:
            # Moving too fast, not loitering
            return False
        
        # Accumulate time in zone
        if self.track_in_zone[track_id]:
            # Already in zone, add frame time
            if history.timestamps and len(history.timestamps) >= 2:
                dt = history.timestamps[-1] - history.timestamps[-2]
                self.track_loiter_time[track_id] += dt
        else:
            # Just entered zone
            self.track_in_zone[track_id] = True
            self.track_loiter_time[track_id] = 0.0
            self.track_loiter_zone[track_id] = zone_name
        
        # Check if loitering threshold exceeded
        if self.track_loiter_time[track_id] >= self.cfg["LOITER_TIME_THRESHOLD"]:
            # Emit event only once per LOITER_TIME_THRESHOLD
            time_since_last_event = current_time - self.track_last_event_time[f"{track_id}_{zone_name}"]
            if time_since_last_event >= self.cfg["LOITER_TIME_THRESHOLD"]:
                self.track_last_event_time[f"{track_id}_{zone_name}"] = current_time
                return True
        
        return False
    
    def get_all_events(self):
        """Get all recorded events."""
        return self.events
    
    def clear_events(self):
        """Clear event log."""
        self.events = []
    
    def save_events_to_csv(self, filepath):
        """
        Save events to CSV file.
        
        Args:
            filepath: Path to output CSV file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'timestamp', 'type', 'track_id', 'zone_name', 'centroid_x', 'centroid_y'
            ])
            writer.writeheader()
            
            for event in self.events:
                centroid = event.get("centroid", (0, 0))
                writer.writerow({
                    'timestamp': event['timestamp'],
                    'type': event['type'],
                    'track_id': event['track_id'],
                    'zone_name': event.get('zone_name', ''),
                    'centroid_x': centroid[0],
                    'centroid_y': centroid[1],
                })
