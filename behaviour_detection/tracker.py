"""Multi-object tracker for behavior detection."""

import time
import numpy as np
from scipy.optimize import linear_sum_assignment


class Track:
    """Represents a single tracked object."""
    
    def __init__(self, track_id, bbox, conf, class_id):
        """
        Initialize a track.
        
        Args:
            track_id: Unique identifier for this track
            bbox: Bounding box (x1, y1, x2, y2)
            conf: Confidence score
            class_id: Class ID from detector
        """
        self.id = track_id
        self.bbox = bbox
        self.conf = conf
        self.class_id = class_id
        self.centroid = self._compute_centroid(bbox)
        self.age = 1
        self.consecutive_frames_without_detection = 0
        self.creation_time = time.time()
    
    @staticmethod
    def _compute_centroid(bbox):
        """Compute centroid from bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def update(self, bbox, conf, class_id):
        """Update track with new detection."""
        self.bbox = bbox
        self.conf = conf
        self.class_id = class_id
        self.centroid = self._compute_centroid(bbox)
        self.age += 1
        self.consecutive_frames_without_detection = 0
    
    def mark_missed(self):
        """Mark that this track was not detected in the current frame."""
        self.consecutive_frames_without_detection += 1
    
    def get_dict(self):
        """Return track as a dictionary."""
        x1, y1, x2, y2 = self.bbox
        cx, cy = self.centroid
        return {
            "id": self.id,
            "bbox": self.bbox,
            "centroid": self.centroid,
            "conf": self.conf,
            "class_id": self.class_id,
            "age": self.age,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "cx": cx,
            "cy": cy,
            "width": x2 - x1,
            "height": y2 - y1,
        }


class Tracker:
    """Multi-object tracker using IoU-based association."""
    
    def __init__(self, max_age=30, iou_threshold=0.3):
        """
        Initialize tracker.
        
        Args:
            max_age: Maximum frames a track can exist without a detection
            iou_threshold: Minimum IoU to associate a detection with a track
        """
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_track_id = 1
        self.frame_count = 0
    
    def update(self, detections, frame=None):
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detections [x1, y1, x2, y2, conf, class_id] or 
                       [x1, y1, x2, y2, conf, class_id, class_name]
            frame: Current frame (optional, not used but kept for API compatibility)
        
        Returns:
            List of active tracks as dictionaries
        """
        self.frame_count += 1
        
        # Convert detections to standardized format
        detection_list = []
        for det in detections:
            if len(det) >= 6:
                x1, y1, x2, y2, conf, class_id = det[:6]
                detection_list.append({
                    "bbox": (x1, y1, x2, y2),
                    "conf": conf,
                    "class_id": int(class_id)
                })
        
        # Match detections to tracks using IoU
        matched_indices, unmatched_detections, unmatched_tracks = self._match_detections(
            detection_list
        )
        
        # Update matched tracks
        for track_idx, det_idx in matched_indices:
            self.tracks[track_idx].update(
                detection_list[det_idx]["bbox"],
                detection_list[det_idx]["conf"],
                detection_list[det_idx]["class_id"]
            )
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            new_track = Track(
                self.next_track_id,
                detection_list[det_idx]["bbox"],
                detection_list[det_idx]["conf"],
                detection_list[det_idx]["class_id"]
            )
            self.tracks.append(new_track)
            self.next_track_id += 1
        
        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # Remove dead tracks
        self.tracks = [
            t for t in self.tracks 
            if t.consecutive_frames_without_detection <= self.max_age
        ]
        
        # Return active tracks as dictionaries
        return [t.get_dict() for t in self.tracks]
    
    def _match_detections(self, detections):
        """
        Match detections to existing tracks using IoU.
        
        Returns:
            tuple: (matched_indices, unmatched_detections, unmatched_tracks)
        """
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        
        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._compute_iou(track.bbox, det["bbox"])
        
        # Use Hungarian algorithm to find best matches
        track_indices, det_indices = linear_sum_assignment(-iou_matrix)
        
        matched_indices = []
        matched_tracks = set()
        matched_dets = set()
        
        for t_idx, d_idx in zip(track_indices, det_indices):
            if iou_matrix[t_idx, d_idx] >= self.iou_threshold:
                matched_indices.append((t_idx, d_idx))
                matched_tracks.add(t_idx)
                matched_dets.add(d_idx)
        
        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in matched_tracks]
        unmatched_detections = [i for i in range(len(detections)) if i not in matched_dets]
        
        return matched_indices, unmatched_detections, unmatched_tracks
    
    @staticmethod
    def _compute_iou(bbox1, bbox2):
        """Compute Intersection over Union between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Compute intersection
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0
        
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Compute union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union <= 0:
            return 0.0
        
        return intersection / union
    
    def get_active_tracks(self):
        """Get all active tracks as dictionaries."""
        return [t.get_dict() for t in self.tracks]
