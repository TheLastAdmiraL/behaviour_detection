"""
Violence Classifier Module
===========================

Wrapper for the YOLOv8 classification model trained on violence detection.
Provides easy integration with the behavior detection pipeline.
"""

from ultralytics import YOLO
from pathlib import Path


class ViolenceClassifier:
    """
    Violence detection classifier using YOLOv8-Classification.
    
    Usage:
        classifier = ViolenceClassifier("runs/violence_cls/train/weights/best.pt")
        result = classifier.predict(frame)
        
        print(f"Violence probability: {result['violence_prob']:.1%}")
        print(f"Is violent: {result['is_violent']}")
    """
    
    def __init__(self, model_path, threshold=0.5):
        """
        Initialize the violence classifier.
        
        Args:
            model_path: Path to trained classification model weights
            threshold: Probability threshold for violence detection (default: 0.5)
        """
        self.model_path = Path(model_path)
        self.threshold = threshold
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"Loading violence classifier: {model_path}")
        self.model = YOLO(str(model_path))
        
        # Get class names and find violence index
        self.class_names = self.model.names
        print(f"Classes: {self.class_names}")
        
        # Determine which index is 'violence'
        # Alphabetically: 'nonviolence' (0) comes before 'violence' (1)
        self.violence_idx = None
        for idx, name in self.class_names.items():
            if 'violence' in name.lower() and 'non' not in name.lower():
                self.violence_idx = idx
                break
        
        if self.violence_idx is None:
            print("Warning: 'violence' class not found, using index 1")
            self.violence_idx = 1
        
        print(f"Violence class index: {self.violence_idx}")
    
    def predict(self, frame):
        """
        Predict violence probability for a frame.
        
        Args:
            frame: Input frame (BGR image from OpenCV)
        
        Returns:
            dict: {
                'violence_prob': float (0-1),
                'nonviolence_prob': float (0-1),
                'is_violent': bool,
                'confidence': float,
                'class_name': str
            }
        """
        # Run inference
        results = self.model.predict(frame, verbose=False)
        
        # Get probabilities
        probs = results[0].probs
        
        # Extract violence probability
        violence_prob = probs.data[self.violence_idx].item()
        nonviolence_prob = 1 - violence_prob
        
        # Get top prediction
        top1_idx = probs.top1
        top1_conf = probs.top1conf.item()
        class_name = self.class_names[top1_idx]
        
        return {
            'violence_prob': violence_prob,
            'nonviolence_prob': nonviolence_prob,
            'is_violent': violence_prob >= self.threshold,
            'confidence': top1_conf,
            'class_name': class_name
        }
    
    def get_violence_score(self, frame):
        """
        Get just the violence probability score.
        
        Args:
            frame: Input frame
        
        Returns:
            float: Violence probability (0-1)
        """
        result = self.predict(frame)
        return result['violence_prob']


def integrate_with_pipeline(detector, violence_classifier, frame):
    """
    Example of running both object detection and violence classification.
    
    Args:
        detector: YoloDetector instance (object detection)
        violence_classifier: ViolenceClassifier instance
        frame: Input frame
    
    Returns:
        tuple: (detections, violence_result)
    """
    # Run object detection
    detections, annotated_frame = detector.run_detection(frame)
    
    # Run violence classification
    violence_result = violence_classifier.predict(frame)
    
    return detections, violence_result, annotated_frame


# Example usage
if __name__ == "__main__":
    import cv2
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to violence model")
    parser.add_argument("--source", type=str, default="0", help="Video source")
    parser.add_argument("--threshold", type=float, default=0.5, help="Violence threshold")
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = ViolenceClassifier(args.model, threshold=args.threshold)
    
    # Open video source
    cap = cv2.VideoCapture(0 if args.source == "0" else args.source)
    
    print("\nRunning violence detection (press 'q' to quit)...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get prediction
        result = classifier.predict(frame)
        
        # Visualize
        violence_prob = result['violence_prob']
        is_violent = result['is_violent']
        
        # Draw result
        if is_violent:
            # Red warning banner
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (0, 0, 180), -1)
            cv2.putText(frame, "!!! VIOLENCE DETECTED !!!", 
                       (frame.shape[1]//2 - 200, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, f"Confidence: {violence_prob:.1%}", 
                       (frame.shape[1]//2 - 100, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            # Green status
            cv2.putText(frame, f"Normal ({1-violence_prob:.1%})", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Violence probability bar
        bar_x = 10
        bar_y = frame.shape[0] - 40
        bar_width = 200
        bar_height = 20
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
        fill_width = int(violence_prob * bar_width)
        color = (0, 0, 255) if is_violent else (0, 200, 0)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
        cv2.putText(frame, f"Violence: {violence_prob:.1%}", (bar_x, bar_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Violence Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
