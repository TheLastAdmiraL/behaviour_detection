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
    
    # COCO class IDs for dangerous objects
    # knife=43, scissors=76
    # Add more if using custom model (e.g., gun)
    DANGEROUS_OBJECTS = {
        43: "KNIFE",
        76: "SCISSORS",
        # Add custom dangerous object IDs here if using custom model
        # Example: 80: "GUN",
    }
    
    def __init__(self, detector=None, tracker=None, rules_engine=None, save_events_to=None, debug=False, violence_classifier=None):
        """
        Initialize behavior pipeline.
        
        Args:
            detector: YoloDetector instance (created if None)
            tracker: Tracker instance (created if None)
            rules_engine: RulesEngine instance (created if None)
            save_events_to: Path to save events CSV (optional)
            debug: Print all detected objects for debugging
            violence_classifier: ViolenceClassifier instance for Phase 3 (optional)
        """
        self.detector = detector or YoloDetector(confidence_threshold=0.5)
        self.tracker = tracker or Tracker()
        self.rules_engine = rules_engine or RulesEngine()
        self.save_events_to = save_events_to
        self.debug = debug
        self.violence_classifier = violence_classifier  # Phase 3
        
        self.prev_time = time.time()
        self.dangerous_detected = []  # Track dangerous objects
        self.violence_result = None  # Current violence classification result
    
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
        
        # Debug: print all detected objects
        if self.debug and len(detections) > 0:
            print(f"[DEBUG] Detected {len(detections)} objects:")
            for det in detections:
                class_name = det[6] if len(det) > 6 else f"class_{int(det[5])}"
                print(f"  - {class_name} (class_id={int(det[5])}, conf={det[4]:.2f})")
        
        # Check for dangerous objects in all detections
        self.dangerous_detected = []
        for det in detections:
            x1, y1, x2, y2, conf, class_id = det[:6]
            class_id = int(class_id)
            if class_id in self.DANGEROUS_OBJECTS:
                self.dangerous_detected.append({
                    "bbox": (x1, y1, x2, y2),
                    "conf": conf,
                    "class_id": class_id,
                    "name": self.DANGEROUS_OBJECTS[class_id]
                })
                if self.debug:
                    print(f"  [!] DANGEROUS OBJECT DETECTED: {self.DANGEROUS_OBJECTS[class_id]}")
        
        # Track (only people for behavior detection)
        tracked = self.tracker.update(detections, frame)
        
        # Associate dangerous objects with nearby persons
        self.armed_persons = {}  # track_id -> weapon name
        for danger in self.dangerous_detected:
            dx1, dy1, dx2, dy2 = danger["bbox"]
            danger_cx = (dx1 + dx2) / 2
            danger_cy = (dy1 + dy2) / 2
            
            for track in tracked:
                px1, py1, px2, py2 = track["bbox"]
                track_id = track["id"]
                
                # Check if dangerous object center is inside person bbox (with margin)
                margin = 50  # pixels
                if (px1 - margin <= danger_cx <= px2 + margin and 
                    py1 - margin <= danger_cy <= py2 + margin):
                    self.armed_persons[track_id] = danger["name"]
                    if self.debug:
                        print(f"  [!] Person ID {track_id} is HOLDING {danger['name']}!")
        
        # Compute time delta
        current_time = time.time()
        dt = current_time - self.prev_time
        self.prev_time = current_time
        
        if dt <= 0:
            dt = 0.033  # Default to ~30 FPS
        
        # Check behaviors
        events = self.rules_engine.step(tracked, dt)
        
        # Add dangerous object events (only if not held by a person)
        for danger in self.dangerous_detected:
            danger_event = {
                "type": "DANGER",
                "track_id": -1,  # No track ID for objects
                "zone_name": danger["name"],
                "timestamp": current_time,
                "centroid": ((danger["bbox"][0] + danger["bbox"][2]) / 2,
                            (danger["bbox"][1] + danger["bbox"][3]) / 2),
            }
            events.append(danger_event)
            self.rules_engine.events.append(danger_event)
        
        # Add ARMED PERSON events (person holding dangerous object)
        for track_id, weapon_name in self.armed_persons.items():
            # Find the track to get centroid
            for track in tracked:
                if track["id"] == track_id:
                    armed_event = {
                        "type": "ARMED_PERSON",
                        "track_id": track_id,
                        "zone_name": weapon_name,
                        "timestamp": current_time,
                        "centroid": track["centroid"],
                    }
                    events.append(armed_event)
                    self.rules_engine.events.append(armed_event)
                    break
        
        # Run violence classification (Phase 3)
        self.violence_result = None
        if self.violence_classifier is not None:
            self.violence_result = self.violence_classifier.predict(frame)
            
            # Add VIOLENCE event if detected
            if self.violence_result['is_violent']:
                violence_event = {
                    "type": "VIOLENCE",
                    "track_id": -1,
                    "zone_name": f"{self.violence_result['violence_prob']:.1%}",
                    "timestamp": current_time,
                    "centroid": (frame.shape[1] / 2, frame.shape[0] / 2),
                }
                events.append(violence_event)
                self.rules_engine.events.append(violence_event)
                
                if self.debug:
                    print(f"  [!] VIOLENCE DETECTED: {self.violence_result['violence_prob']:.1%}")
        
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
            
            # Check if this person is holding a dangerous object
            is_armed = track_id in self.armed_persons
            weapon_name = self.armed_persons.get(track_id, "")
            
            # Draw bounding box - RED if armed, GREEN otherwise
            if is_armed:
                color = (0, 0, 255)  # Red for armed person
                thickness = 3
            else:
                color = (0, 255, 0)  # Green
                thickness = 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Draw centroid
            cv2.circle(annotated, (cx, cy), 3, color, -1)
            
            # Draw track ID (and weapon if armed)
            font = cv2.FONT_HERSHEY_SIMPLEX
            if is_armed:
                label = f"ARMED: {weapon_name}"
                # Draw warning background
                text_size = cv2.getTextSize(label, font, 0.7, 2)[0]
                cv2.rectangle(annotated, (x1, y1 - text_size[1] - 10), 
                             (x1 + text_size[0] + 10, y1), (0, 0, 255), -1)
                cv2.putText(annotated, label, (x1 + 5, y1 - 5), 
                           font, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(annotated, f"ID {track_id}", (x1, y1 - 10), 
                           font, 0.5, color, 2)
        
        # Draw DANGEROUS OBJECTS with red boxes and warning
        for danger in self.dangerous_detected:
            x1, y1, x2, y2 = danger["bbox"]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            name = danger["name"]
            conf = danger["conf"]
            
            # Bright RED color for dangerous objects
            danger_color = (0, 0, 255)  # BGR Red
            
            # Draw thick red bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), danger_color, 3)
            
            # Draw red filled background for label
            label = f"DANGER: {name} {conf:.0%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            
            # Label background
            cv2.rectangle(annotated, 
                         (x1, y1 - text_size[1] - 10), 
                         (x1 + text_size[0] + 10, y1), 
                         danger_color, -1)
            
            # Label text (white on red)
            cv2.putText(annotated, label, (x1 + 5, y1 - 5), 
                       font, font_scale, (255, 255, 255), thickness)
            
            # Draw warning corners (flashing effect visual)
            corner_len = 20
            cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), danger_color, 4)
            cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), danger_color, 4)
            cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), danger_color, 4)
            cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), danger_color, 4)
            cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), danger_color, 4)
            cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), danger_color, 4)
            cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), danger_color, 4)
            cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), danger_color, 4)
        
        # Show danger alert banner if any dangerous objects detected
        if self.dangerous_detected:
            banner_text = f"WARNING: {len(self.dangerous_detected)} DANGEROUS OBJECT(S) DETECTED!"
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(banner_text, font, 0.8, 2)[0]
            
            # Red banner at top of frame
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 40), (0, 0, 200), -1)
            text_x = (annotated.shape[1] - text_size[0]) // 2
            cv2.putText(annotated, banner_text, (text_x, 28), 
                       font, 0.8, (255, 255, 255), 2)
        
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
        
        # Draw violence classification result (Phase 3)
        if self.violence_result is not None:
            violence_prob = self.violence_result['violence_prob']
            is_violent = self.violence_result['is_violent']
            
            # Violence probability bar at bottom
            bar_x = 10
            bar_y = annotated.shape[0] - 50
            bar_width = 200
            bar_height = 25
            
            # Background
            cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
            
            # Fill based on violence probability
            fill_width = int(violence_prob * bar_width)
            bar_color = (0, 0, 255) if is_violent else (0, 200, 0)
            cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), bar_color, -1)
            
            # Border
            cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 1)
            
            # Label
            cv2.putText(annotated, f"Violence: {violence_prob:.0%}", (bar_x, bar_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # If violent, show big warning banner
            if is_violent:
                banner_y = 50 if self.dangerous_detected else 0  # Offset if danger banner exists
                cv2.rectangle(annotated, (0, banner_y), (annotated.shape[1], banner_y + 50), (0, 0, 180), -1)
                cv2.putText(annotated, "!!! VIOLENCE DETECTED !!!", 
                           (annotated.shape[1]//2 - 180, banner_y + 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Draw FPS
        fps = 1.0 / (self.prev_time - time.time() + 0.001)
        if fps > 0:
            cv2.putText(annotated, f"FPS: {fps:.1f}", 
                       (annotated.shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated
