"""YOLOv8 detector wrapper for object detection."""

from ultralytics import YOLO


class YoloDetector:
    """Wrapper for YOLOv8 object detection."""
    
    def __init__(self, model_name="yolov8n.pt", confidence_threshold=0.5):
        """
        Initialize YOLO detector.
        
        Args:
            model_name: Name of the YOLO model (default: yolov8n.pt for nano model)
            confidence_threshold: Minimum confidence for detections
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.model = YOLO(model_name)
    
    def run_detection(self, frame, dangerous_objects_conf=0.05):
        """
        Run detection on a single frame.
        
        Args:
            frame: Input frame (BGR image from OpenCV)
            dangerous_objects_conf: Lower confidence threshold for dangerous objects (default: 0.15)
        
        Returns:
            tuple: (detections, annotated_frame)
                detections: list of [x1, y1, x2, y2, conf, class_id, class_name]
                annotated_frame: frame with drawn detections
        """
        # Run inference with lower confidence to catch dangerous objects
        min_conf = min(self.confidence_threshold, dangerous_objects_conf)
        results = self.model(frame, conf=min_conf, verbose=False)
        
        # Dangerous object class IDs (knife and scissors)
        DANGEROUS_CLASSES = {43, 76}  # KNIFE=43, SCISSORS=76
        
        # Extract detections
        detections = []
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]
                    
                    # Apply different confidence thresholds
                    if class_id in DANGEROUS_CLASSES:
                        # Lower threshold for dangerous objects
                        if conf >= dangerous_objects_conf:
                            detections.append([x1, y1, x2, y2, conf, class_id, class_name])
                    else:
                        # Normal threshold for other objects
                        if conf >= self.confidence_threshold:
                            detections.append([x1, y1, x2, y2, conf, class_id, class_name])
        
        # Get annotated frame from YOLO
        annotated_frame = results[0].plot() if results else frame
        
        # Convert from RGB (YOLO uses RGB) back to BGR for OpenCV
        annotated_frame = annotated_frame[..., ::-1]  # RGB to BGR
        
        return detections, annotated_frame
    
    def get_class_names(self):
        """Get dictionary of class IDs to class names."""
        return self.model.names
