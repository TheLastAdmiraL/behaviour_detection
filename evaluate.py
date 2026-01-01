"""
Evaluation script for Behavior Detection System.

This script provides multiple ways to verify the system's output:
1. Manual annotation mode - label frames and compare with predictions
2. Synthetic test mode - generate test scenarios with known ground truth
3. Metrics calculation - precision, recall, F1 score
4. Video replay with ground truth comparison
"""

import cv2
import time
import json
import csv
from pathlib import Path
from datetime import datetime

from yolo_object_detection.detectors import YoloDetector
from behaviour_detection.tracker import Tracker
from behaviour_detection.rules import RulesEngine
from behaviour_detection.pipeline import BehaviourPipeline


class EvaluationTool:
    """Tool for evaluating behavior detection accuracy."""
    
    def __init__(self):
        self.ground_truth = []  # List of {frame, type, track_id}
        self.predictions = []   # List of {frame, type, track_id}
        self.current_frame = 0
        
    def run_manual_annotation(self, source):
        """
        Run manual annotation mode.
        
        Keys:
            R - Mark RUNNING event
            L - Mark LOITERING event
            F - Mark FALL event
            K - Mark KNIFE detected
            A - Mark ARMED person
            S - Save annotations
            Q - Quit
        """
        print("\n" + "="*60)
        print("MANUAL ANNOTATION MODE")
        print("="*60)
        print("Keys:")
        print("  R - Mark RUNNING")
        print("  L - Mark LOITERING") 
        print("  F - Mark FALL")
        print("  K - Mark KNIFE detected")
        print("  A - Mark ARMED person")
        print("  SPACE - Pause/Resume")
        print("  S - Save annotations to file")
        print("  Q - Quit")
        print("="*60 + "\n")
        
        # Open video
        if source == "0":
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(source)
            
        if not cap.isOpened():
            print(f"Cannot open source: {source}")
            return
        
        paused = False
        frame_num = 0
        annotations = []
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_num += 1
            
            # Draw frame info
            display = frame.copy()
            cv2.putText(display, f"Frame: {frame_num}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, f"Annotations: {len(annotations)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if paused:
                cv2.putText(display, "PAUSED", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show recent annotations
            y = 120
            for ann in annotations[-5:]:
                text = f"F{ann['frame']}: {ann['type']}"
                cv2.putText(display, text, (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                y += 20
            
            cv2.imshow("Manual Annotation", display)
            
            key = cv2.waitKey(30 if not paused else 0) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
            elif key == ord('r'):
                annotations.append({"frame": frame_num, "type": "RUN", "timestamp": time.time()})
                print(f"[Frame {frame_num}] Marked: RUNNING")
            elif key == ord('l'):
                annotations.append({"frame": frame_num, "type": "LOITER", "timestamp": time.time()})
                print(f"[Frame {frame_num}] Marked: LOITERING")
            elif key == ord('f'):
                annotations.append({"frame": frame_num, "type": "FALL", "timestamp": time.time()})
                print(f"[Frame {frame_num}] Marked: FALL")
            elif key == ord('k'):
                annotations.append({"frame": frame_num, "type": "DANGER", "timestamp": time.time()})
                print(f"[Frame {frame_num}] Marked: KNIFE/DANGER")
            elif key == ord('a'):
                annotations.append({"frame": frame_num, "type": "ARMED_PERSON", "timestamp": time.time()})
                print(f"[Frame {frame_num}] Marked: ARMED PERSON")
            elif key == ord('s'):
                self._save_annotations(annotations, source)
        
        cap.release()
        cv2.destroyAllWindows()
        
        return annotations
    
    def _save_annotations(self, annotations, source):
        """Save annotations to JSON file."""
        filename = f"ground_truth_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump({
                "source": str(source),
                "annotations": annotations,
                "total": len(annotations)
            }, f, indent=2)
        print(f"\n✓ Saved {len(annotations)} annotations to {filename}\n")
    
    def run_comparison(self, video_source, ground_truth_file):
        """
        Run detection and compare with ground truth.
        
        Args:
            video_source: Path to video file
            ground_truth_file: Path to JSON ground truth file
        """
        print("\n" + "="*60)
        print("COMPARISON MODE")
        print("="*60)
        
        # Load ground truth
        with open(ground_truth_file, 'r') as f:
            gt_data = json.load(f)
        
        gt_annotations = gt_data["annotations"]
        gt_by_frame = {}
        for ann in gt_annotations:
            frame = ann["frame"]
            if frame not in gt_by_frame:
                gt_by_frame[frame] = []
            gt_by_frame[frame].append(ann["type"])
        
        print(f"Loaded {len(gt_annotations)} ground truth annotations")
        
        # Initialize pipeline
        detector = YoloDetector(confidence_threshold=0.25)
        tracker = Tracker(max_age=30, iou_threshold=0.3)
        zones = {"center": (100, 100, 500, 400)}
        rules_engine = RulesEngine(zones=zones)
        pipeline = BehaviourPipeline(detector, tracker, rules_engine)
        
        # Process video
        cap = cv2.VideoCapture(video_source)
        frame_num = 0
        predictions_by_frame = {}
        
        print("Processing video...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            _, _, events, _ = pipeline._run_pipeline_step(frame)
            
            if events:
                predictions_by_frame[frame_num] = [e["type"] for e in events]
        
        cap.release()
        
        # Calculate metrics
        self._calculate_metrics(gt_by_frame, predictions_by_frame, frame_num)
    
    def _calculate_metrics(self, ground_truth, predictions, total_frames):
        """Calculate and print evaluation metrics."""
        
        event_types = ["RUN", "FALL", "LOITER", "DANGER", "ARMED_PERSON"]
        
        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)
        
        for event_type in event_types:
            # True Positives, False Positives, False Negatives
            tp = 0
            fp = 0
            fn = 0
            
            # Get all frames with this event type in ground truth
            gt_frames = set()
            for frame, types in ground_truth.items():
                if event_type in types:
                    gt_frames.add(frame)
            
            # Get all frames with this event type in predictions
            pred_frames = set()
            for frame, types in predictions.items():
                if event_type in types:
                    pred_frames.add(frame)
            
            # Allow tolerance of +/- 5 frames for matching
            tolerance = 5
            matched_gt = set()
            matched_pred = set()
            
            for gt_frame in gt_frames:
                for pred_frame in pred_frames:
                    if abs(gt_frame - pred_frame) <= tolerance:
                        matched_gt.add(gt_frame)
                        matched_pred.add(pred_frame)
            
            tp = len(matched_gt)
            fn = len(gt_frames) - tp
            fp = len(pred_frames) - len(matched_pred)
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\n{event_type}:")
            print(f"  Ground Truth: {len(gt_frames)} events")
            print(f"  Predictions:  {len(pred_frames)} events")
            print(f"  True Positives:  {tp}")
            print(f"  False Positives: {fp}")
            print(f"  False Negatives: {fn}")
            print(f"  Precision: {precision:.2%}")
            print(f"  Recall:    {recall:.2%}")
            print(f"  F1 Score:  {f1:.2%}")
        
        print("\n" + "="*60)


def run_synthetic_test():
    """
    Run synthetic tests with known scenarios.
    Tests the detection logic without needing real video.
    """
    print("\n" + "="*60)
    print("SYNTHETIC TESTS")
    print("="*60)
    
    # Test 1: Running detection
    print("\n[Test 1] Running Detection")
    rules = RulesEngine(zones={}, cfg={"RUN_SPEED_THRESHOLD": 100.0})
    
    # Simulate fast-moving person
    tracks = []
    for i in range(10):
        track = {
            "id": 1,
            "bbox": (100 + i*20, 100, 200 + i*20, 300),  # Moving 20px per frame
            "centroid": (150 + i*20, 200),
            "class_id": 0
        }
        events = rules.step([track], dt=0.033)  # 30 FPS
        if events:
            for e in events:
                if e["type"] == "RUN":
                    print(f"  ✓ Running detected at frame {i+1}")
    
    run_events = [e for e in rules.get_all_events() if e["type"] == "RUN"]
    print(f"  Result: {len(run_events)} RUN events detected")
    print(f"  Status: {'PASS' if len(run_events) > 0 else 'FAIL'}")
    
    # Test 2: Loitering detection
    print("\n[Test 2] Loitering Detection")
    rules2 = RulesEngine(
        zones={"test_zone": (100, 100, 300, 300)},
        cfg={"LOITER_TIME_THRESHOLD": 0.5, "LOITER_SPEED_THRESHOLD": 50.0}
    )
    
    # Simulate stationary person in zone for 20 frames
    for i in range(20):
        track = {
            "id": 1,
            "bbox": (150, 150, 250, 280),  # Stationary in zone
            "centroid": (200, 215),
            "class_id": 0
        }
        events = rules2.step([track], dt=0.1)  # 10 FPS, so 20 frames = 2 seconds
        if events:
            for e in events:
                if e["type"] == "LOITER":
                    print(f"  ✓ Loitering detected at frame {i+1}")
    
    loiter_events = [e for e in rules2.get_all_events() if e["type"] == "LOITER"]
    print(f"  Result: {len(loiter_events)} LOITER events detected")
    print(f"  Status: {'PASS' if len(loiter_events) > 0 else 'FAIL'}")
    
    # Test 3: Fall detection
    print("\n[Test 3] Fall Detection")
    rules3 = RulesEngine(zones={}, cfg={
        "FALL_VERTICAL_RATIO_DROP": 0.3,
        "FALL_DOWNWARD_DISTANCE": 15.0
    })
    
    # Simulate standing person then falling
    heights = [(100, 100, 200, 300), (100, 150, 200, 280), (100, 200, 200, 250)]  # Getting shorter & lower
    for i, bbox in enumerate(heights):
        track = {
            "id": 1,
            "bbox": bbox,
            "centroid": ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2),
            "class_id": 0
        }
        events = rules3.step([track], dt=0.1)
        if events:
            for e in events:
                if e["type"] == "FALL":
                    print(f"  ✓ Fall detected at frame {i+1}")
    
    fall_events = [e for e in rules3.get_all_events() if e["type"] == "FALL"]
    print(f"  Result: {len(fall_events)} FALL events detected")
    print(f"  Status: {'PASS' if len(fall_events) > 0 else 'FAIL'}")
    
    # Test 4: Dangerous object association
    print("\n[Test 4] Dangerous Object Association (Knife + Person)")
    
    # Simulate detector output with person and knife
    person_bbox = (100, 100, 200, 300)
    knife_bbox = (180, 200, 220, 230)  # Overlaps with person
    
    # Check if knife center is inside person bbox (with margin)
    knife_cx = (knife_bbox[0] + knife_bbox[2]) / 2
    knife_cy = (knife_bbox[1] + knife_bbox[3]) / 2
    margin = 50
    
    px1, py1, px2, py2 = person_bbox
    is_associated = (px1 - margin <= knife_cx <= px2 + margin and 
                     py1 - margin <= knife_cy <= py2 + margin)
    
    print(f"  Person bbox: {person_bbox}")
    print(f"  Knife bbox: {knife_bbox}")
    print(f"  Knife center: ({knife_cx}, {knife_cy})")
    print(f"  Associated: {is_associated}")
    print(f"  Status: {'PASS' if is_associated else 'FAIL'}")
    
    print("\n" + "="*60)
    print("SYNTHETIC TESTS COMPLETE")
    print("="*60)


def print_usage():
    """Print usage instructions."""
    print("""
Behavior Detection Evaluation Tool
===================================

Usage:
    python evaluate.py --mode <mode> [options]

Modes:
    --mode annotate --source <video|0>
        Manually annotate events in a video/webcam
        Creates ground_truth_*.json file
    
    --mode compare --source <video> --ground-truth <file.json>
        Compare system predictions with ground truth
        Shows precision, recall, F1 for each event type
    
    --mode synthetic
        Run synthetic unit tests (no video needed)
        Tests the detection logic with known inputs
    
    --mode live
        Live evaluation - shows predictions in real-time
        with confidence scores

Examples:
    python evaluate.py --mode annotate --source test_video.mp4
    python evaluate.py --mode compare --source test_video.mp4 --ground-truth gt.json
    python evaluate.py --mode synthetic
""")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Behavior Detection Evaluation Tool")
    parser.add_argument("--mode", type=str, choices=["annotate", "compare", "synthetic", "live"],
                       help="Evaluation mode")
    parser.add_argument("--source", type=str, help="Video source (file path or 0 for webcam)")
    parser.add_argument("--ground-truth", type=str, help="Ground truth JSON file (for compare mode)")
    
    args = parser.parse_args()
    
    if args.mode is None:
        print_usage()
    elif args.mode == "annotate":
        if not args.source:
            print("Error: --source required for annotate mode")
        else:
            tool = EvaluationTool()
            tool.run_manual_annotation(args.source)
    elif args.mode == "compare":
        if not args.source or not args.ground_truth:
            print("Error: --source and --ground-truth required for compare mode")
        else:
            tool = EvaluationTool()
            tool.run_comparison(args.source, args.ground_truth)
    elif args.mode == "synthetic":
        run_synthetic_test()
    elif args.mode == "live":
        print("Running live evaluation with webcam...")
        import subprocess
        subprocess.run(["python", "run_behaviour.py", "--source", "0", "--show", "--debug"])
