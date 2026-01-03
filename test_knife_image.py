"""Test knife/weapon detection on images."""

import sys
from pathlib import Path
from ultralytics import YOLO
import cv2

def test_image(image_path, model_path="runs/weapon_det/weights/best.pt"):
    """Test knife detection on a single image."""
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}")
        return
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print("Using default YOLOv8n model instead...")
        model_path = "yolov8n.pt"
    
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    print(f"Testing image: {image_path}")
    print(f"Model classes: {model.names}")
    print("-" * 50)
    
    # Run detection with LOW confidence to catch everything
    results = model(image_path, conf=0.1, verbose=True)
    
    # Print results
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        print(f"\nDetected {len(results[0].boxes)} objects:")
        for i, box in enumerate(results[0].boxes):
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            conf = float(box.conf[0])
            print(f"  {i+1}. {class_name} - Confidence: {conf:.1%}")
    else:
        print("No objects detected.")
    
    # Show annotated image
    print("\nDisplaying annotated image... (Press any key to close)")
    annotated = results[0].plot()
    cv2.imshow(f"Detection Results - {Path(image_path).name}", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save result
    output_path = Path(image_path).parent / f"result_{Path(image_path).name}"
    cv2.imwrite(str(output_path), annotated)
    print(f"\nSaved annotated image to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_knife_image.py <image_path> [model_path]")
        print("\nExample:")
        print("  python test_knife_image.py test_images/knife_test_1.jpg")
        print("  python test_knife_image.py test_images/knife_test_1.jpg runs/weapon_det/weights/best.pt")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "runs/weapon_det/weights/best.pt"
    
    test_image(image_path, model_path)
