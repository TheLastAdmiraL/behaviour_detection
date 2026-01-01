"""Quick test to see what YOLO detects - especially knives."""

import cv2
from ultralytics import YOLO

# Load model
print("Loading YOLOv8 model...")
model = YOLO("yolov8n.pt")

# Print knife class info
print(f"\nCOCO class 43 = {model.names[43]}")
print(f"COCO class 76 = {model.names[76]}")

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

print("\n--- Press 'q' to quit ---")
print("Hold a knife in front of the camera and see if it's detected.\n")

# Very low confidence to catch anything
CONF = 0.1

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run detection with very low confidence
    results = model(frame, conf=CONF, verbose=False)
    
    frame_count += 1
    
    # Print all detections every 30 frames
    if frame_count % 30 == 0 and results[0].boxes is not None:
        print(f"\n[Frame {frame_count}] Detected objects:")
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            conf = float(box.conf[0])
            print(f"  - {class_name} (id={class_id}, conf={conf:.1%})")
            
            # Highlight if it's knife or scissors
            if class_id in [43, 76]:
                print(f"    ^^^ DANGEROUS OBJECT DETECTED! ^^^")
    
    # Draw detections
    annotated = results[0].plot()
    
    # Add instruction text
    cv2.putText(annotated, "Hold knife/scissors in view. Press 'q' to quit.", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(annotated, f"Confidence threshold: {CONF:.0%}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    cv2.imshow("Knife Detection Test", annotated)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nTest complete.")
