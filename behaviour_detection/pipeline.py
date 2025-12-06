"""End-to-end behavior detection pipeline."""

import time
import cv2
from pathlib import Path
from tqdm import tqdm

from yolo_object_detection.detectors import YoloDetector
from behaviour_detection.tracker import Tracker
from behaviour_detection.rules import RulesEngine


class BehaviourPipeline:
    """End-to-end pipeline for behavior detection and annotation."""
    
    def __init__(self, detector=None, tracker=None, rules_engine=None, save_events_to=None):
        """
        Initialize behavior pipeline.
        
        Args:
            detector: YoloDetector instance (created if None)
            tracker: Tracker instance (created if None)
            rules_engine: RulesEngine instance (created if None)
            save_events_to: Path to save events CSV (optional)
        """
        self.detector = detector or YoloDetector(confidence_threshold=0.5)
        self.tracker = tracker or Tracker()
        self.rules_engine = rules_engine or RulesEngine()
        self.save_events_to = save_events_to
        
        self.prev_time = time.time()
    
    def process_stream(self, source, show=True, save_dir=None):
        """
        Process a video stream or image.
        
        Args:
            source: 0 for webcam, path to video file, or path to image
            show: Whether to display frames
            save_dir: Directory to save annotated frames
        """
        # Determine source type
        if isinstance(source, int) or (isinstance(source, str) and source == "0"):
            self._process_webcam(show, save_dir)
        else:
            source_path = Path(source)
            if source_path.is_file():
                if source_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    self._process_image(source_path, show, save_dir)
                else:
                    self._process_video(source_path, show, save_dir)
            else:
                print(f"Error: Invalid source {source}")
    
    def _process_image(self, image_path, show, save_dir):
        """Process a single image."""
        print(f"Processing image: {image_path}")
        
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Error: Could not read image {image_path}")
            return
        
        # Run pipeline
        detections, tracked, events, annotated = self._run_pipeline_step(frame)
        
        # Display if requested
        if show:
            cv2.imshow("Behavior Detection", annotated)
            print("Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Save if requested
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            output_file = save_path / f"{image_path.stem}_annotated.jpg"
            cv2.imwrite(str(output_file), annotated)
            print(f"Saved to {output_file}")
    
    def _process_video(self, video_path, show, save_dir):
        """Process a video file."""
        print(f"Processing video: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        save_path = None
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        
        frame_idx = 0
        with tqdm(total=total_frames, desc="Processing") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                detections, tracked, events, annotated = self._run_pipeline_step(frame)
                
                # Save frame
                if save_path:
                    output_file = save_path / f"frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(output_file), annotated)
                
                # Display
                if show:
                    cv2.imshow("Behavior Detection", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Interrupted by user")
                        break
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"Processed {frame_idx} frames")
        
        # Save events
        if self.save_events_to:
            self.rules_engine.save_events_to_csv(self.save_events_to)
            print(f"Events saved to {self.save_events_to}")
    
    def _process_webcam(self, show, save_dir):
        """Process webcam stream."""
        print("Starting webcam... (Press 'q' to quit)")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        save_path = None
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        
        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read from webcam")
                    break
                
                detections, tracked, events, annotated = self._run_pipeline_step(frame)
                
                # Save frame
                if save_path:
                    output_file = save_path / f"frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(output_file), annotated)
                
                # Display
                if show:
                    cv2.imshow("Behavior Detection - Webcam", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Stopped by user")
                        break
                
                frame_idx += 1
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        print(f"Captured {frame_idx} frames")
        
        # Save events
        if self.save_events_to:
            self.rules_engine.save_events_to_csv(self.save_events_to)
            print(f"Events saved to {self.save_events_to}")
    
    def _run_pipeline_step(self, frame):
        """
        Run one pipeline step: detect -> track -> rules -> annotate.
        
        Returns:
            tuple: (detections, tracked, events, annotated_frame)
        """
        # Detect
        detections, _ = self.detector.run_detection(frame)
        
        # Track
        tracked = self.tracker.update(detections, frame)
        
        # Compute time delta
        current_time = time.time()
        dt = current_time - self.prev_time
        self.prev_time = current_time
        
        if dt <= 0:
            dt = 0.033  # Default to ~30 FPS
        
        # Check behaviors
        events = self.rules_engine.step(tracked, dt)
        
        # Annotate
        annotated = self._annotate_frame(frame, tracked, events)
        
        return detections, tracked, events, annotated
    
    def _annotate_frame(self, frame, tracked, events):
        """
        Annotate frame with tracking and behavior information.
        
        Args:
            frame: Input frame
            tracked: List of tracked objects
            events: List of events detected
        
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Draw tracks
        for track in tracked:
            track_id = track["id"]
            x1, y1, x2, y2 = track["bbox"]
            cx, cy = track["centroid"]
            
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cx, cy = int(cx), int(cy)
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw centroid
            cv2.circle(annotated, (cx, cy), 3, color, -1)
            
            # Draw track ID
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(annotated, f"ID {track_id}", (x1, y1 - 10), 
                       font, 0.5, color, 2)
        
        # Draw events
        for event in events:
            track_id = event["track_id"]
            event_type = event["type"]
            
            # Find track to get position
            for track in tracked:
                if track["id"] == track_id:
                    x1, y1, x2, y2 = track["bbox"]
                    y_offset = int(y1 - 30)
                    
                    # Color based on event type
                    if event_type == "RUN":
                        color = (0, 0, 255)  # Red
                        label = "RUNNING"
                    elif event_type == "FALL":
                        color = (0, 165, 255)  # Orange
                        label = "FALL"
                    elif event_type == "LOITER":
                        color = (255, 0, 0)  # Blue
                        label = f"LOITER ({event.get('zone_name', 'zone')})"
                    else:
                        color = (255, 255, 255)
                        label = event_type
                    
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(annotated, label, (int(x1), int(max(20, y_offset))), 
                               font, 0.7, color, 2)
                    break
        
        # Draw zones
        for zone_name, zone_rect in self.rules_engine.zones.items():
            x1, y1, x2, y2 = zone_rect
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (200, 200, 0), 2)
            cv2.putText(annotated, zone_name, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
        
        # Draw FPS
        fps = 1.0 / (self.prev_time - time.time() + 0.001)
        if fps > 0:
            cv2.putText(annotated, f"FPS: {fps:.1f}", 
                       (annotated.shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated
