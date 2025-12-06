"""Main CLI entry point for YOLO object detection."""

import sys
import cv2
from pathlib import Path
from tqdm import tqdm

from yolo_object_detection.utils import (
    parse_arguments, get_file_type, is_valid_source, 
    FPSMeter, draw_detections, draw_fps
)
from yolo_object_detection.detectors import YoloDetector


def process_image(detector, image_path, output_dir=None, show=False):
    """
    Process a single image.
    
    Args:
        detector: YoloDetector instance
        image_path: Path to input image
        output_dir: Directory to save annotated image
        show: Whether to display the image
    """
    print(f"Processing image: {image_path}")
    
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    detections, annotated_frame = detector.run_detection(frame)
    
    # Draw FPS (static image, just label)
    annotated_frame = draw_fps(annotated_frame, FPSMeter())
    
    print(f"Detected {len(detections)} objects")
    
    # Save if output_dir is provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        input_name = Path(image_path).stem
        output_path = output_dir / f"{input_name}_annotated.jpg"
        cv2.imwrite(str(output_path), annotated_frame)
        print(f"Saved annotated image to {output_path}")
    
    # Show if requested
    if show:
        cv2.imshow("YOLO Detection", annotated_frame)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video(detector, video_path, output_dir=None, show=False, confidence=0.5):
    """
    Process a video file.
    
    Args:
        detector: YoloDetector instance
        video_path: Path to input video
        output_dir: Directory to save annotated frames
        show: Whether to display frames
        confidence: Confidence threshold
    """
    print(f"Processing video: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_input = cap.get(cv2.CAP_PROP_FPS)
    
    fps_meter = FPSMeter()
    frame_idx = 0
    
    output_dir_path = None
    if output_dir:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Total frames: {frame_count}, FPS: {fps_input:.1f}")
    
    with tqdm(total=frame_count, desc="Processing") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            fps_meter.update()
            detections, annotated_frame = detector.run_detection(frame)
            annotated_frame = draw_fps(annotated_frame, fps_meter)
            
            # Save frame if output_dir is provided
            if output_dir_path:
                frame_name = f"frame_{frame_idx:06d}.jpg"
                frame_path = output_dir_path / frame_name
                cv2.imwrite(str(frame_path), annotated_frame)
            
            # Display if requested
            if show:
                cv2.imshow("YOLO Detection - Video", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Interrupted by user")
                    break
            
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Processed {frame_idx} frames")
    if output_dir_path:
        print(f"Saved {frame_idx} annotated frames to {output_dir_path}")


def process_webcam(detector, output_dir=None, show=True):
    """
    Process webcam stream.
    
    Args:
        detector: YoloDetector instance
        output_dir: Directory to save annotated frames
        show: Whether to display frames (should be True for webcam)
    """
    print("Starting webcam capture (Press 'q' to quit)...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    fps_meter = FPSMeter()
    frame_idx = 0
    
    output_dir_path = None
    if output_dir:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read from webcam")
                break
            
            fps_meter.update()
            detections, annotated_frame = detector.run_detection(frame)
            annotated_frame = draw_fps(annotated_frame, fps_meter)
            
            # Save frame if output_dir is provided
            if output_dir_path:
                frame_name = f"frame_{frame_idx:06d}.jpg"
                frame_path = output_dir_path / frame_name
                cv2.imwrite(str(frame_path), annotated_frame)
            
            # Display
            if show:
                cv2.imshow("YOLO Detection - Webcam", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Stopped by user")
                    break
            
            frame_idx += 1
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    print(f"Captured {frame_idx} frames")
    if output_dir_path:
        print(f"Saved {frame_idx} annotated frames to {output_dir_path}")


def process_folder(detector, folder_path, output_dir=None, show=False):
    """
    Process all images and videos in a folder.
    
    Args:
        detector: YoloDetector instance
        folder_path: Path to folder
        output_dir: Directory to save annotated outputs
        show: Whether to display images
    """
    print(f"Processing folder: {folder_path}")
    
    folder = Path(folder_path)
    if not folder.is_dir():
        print(f"Error: {folder_path} is not a directory")
        return
    
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}
    
    # Find all image and video files
    files = []
    for ext in image_extensions | video_extensions:
        files.extend(folder.glob(f"*{ext}"))
        files.extend(folder.glob(f"*{ext.upper()}"))
    
    files = sorted(set(files))  # Remove duplicates and sort
    
    if not files:
        print(f"No image or video files found in {folder_path}")
        return
    
    print(f"Found {len(files)} media files")
    
    for file_path in files:
        if file_path.suffix.lower() in image_extensions:
            process_image(detector, file_path, output_dir, show)
        elif file_path.suffix.lower() in video_extensions:
            process_video(detector, file_path, output_dir, show)


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Validate source
    if not is_valid_source(args.source):
        print(f"Error: Invalid source '{args.source}'")
        print("Source must be: path to image, video, folder, or 0 for webcam.")
        sys.exit(1)
    
    # Initialize detector
    print("Loading YOLOv8 model...")
    detector = YoloDetector(confidence_threshold=args.conf)
    print("Model loaded successfully")
    
    # Determine source type and process
    source_type = get_file_type(args.source)
    
    try:
        if source_type == "image":
            process_image(detector, args.source, args.save_dir, args.show)
        elif source_type == "video":
            process_video(detector, args.source, args.save_dir, args.show, args.conf)
        elif source_type == "webcam":
            process_webcam(detector, args.save_dir, args.show)
        elif source_type == "folder":
            process_folder(detector, args.source, args.save_dir, args.show)
        else:
            print(f"Error: Unknown source type")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
