"""CLI runner for behavior detection system."""

import argparse
import sys
from pathlib import Path

from yolo_object_detection.detectors import YoloDetector
from behaviour_detection.tracker import Tracker
from behaviour_detection.rules import RulesEngine
from behaviour_detection.pipeline import BehaviourPipeline


def create_default_zones(frame_width, frame_height):
    """Create default loitering zones."""
    # Create one zone in the center of the frame
    center_x = frame_width / 2
    center_y = frame_height / 2
    zone_size = 200  # pixels
    
    zones = {
        "center": (
            int(center_x - zone_size),
            int(center_y - zone_size),
            int(center_x + zone_size),
            int(center_y + zone_size),
        )
    }
    
    return zones


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AI-Powered Behaviour Detection System (running, loitering, falls)"
    )
    
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="0 for webcam or path to video/image file"
    )
    
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)"
    )
    
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display annotated frames"
    )
    
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save annotated frames"
    )
    
    parser.add_argument(
        "--events-csv",
        type=str,
        default=None,
        help="Path to save events log (CSV format)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Validate source
    if args.source != "0":
        source_path = Path(args.source)
        if not source_path.exists():
            print(f"Error: Source '{args.source}' does not exist")
            sys.exit(1)
    
    # Initialize components
    print("Loading YOLOv8 model...")
    detector = YoloDetector(confidence_threshold=args.conf)
    print("Model loaded")
    
    print("Initializing tracker...")
    tracker = Tracker(max_age=30, iou_threshold=0.3)
    
    print("Initializing behavior detection engine...")
    
    # Create default zones (will be adjusted based on first frame)
    zones = {"center": (100, 100, 300, 300)}
    
    cfg = {
        "RUN_SPEED_THRESHOLD": 150.0,
        "LOITER_TIME_THRESHOLD": 10.0,
        "LOITER_SPEED_THRESHOLD": 50.0,
        "FALL_VERTICAL_RATIO_DROP": 0.4,
        "FALL_DOWNWARD_DISTANCE": 20.0,
    }
    
    rules_engine = RulesEngine(zones=zones, cfg=cfg)
    
    # Create pipeline
    pipeline = BehaviourPipeline(
        detector=detector,
        tracker=tracker,
        rules_engine=rules_engine,
        save_events_to=args.events_csv
    )
    
    # Process stream
    try:
        print("Starting behavior detection...")
        if args.show:
            print("(Press 'q' to quit)")
        
        pipeline.process_stream(
            source=args.source,
            show=args.show if args.source != "0" else True,
            save_dir=args.save_dir
        )
        
        print("\nBehavior detection completed")
        
        # Print statistics
        events = rules_engine.get_all_events()
        if events:
            print(f"\nDetected {len(events)} behavior events:")
            
            run_count = sum(1 for e in events if e["type"] == "RUN")
            fall_count = sum(1 for e in events if e["type"] == "FALL")
            loiter_count = sum(1 for e in events if e["type"] == "LOITER")
            
            print(f"  - Running: {run_count}")
            print(f"  - Falls: {fall_count}")
            print(f"  - Loitering: {loiter_count}")
        else:
            print("\nNo behavior events detected")
        
        if args.events_csv:
            print(f"Events saved to {args.events_csv}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
