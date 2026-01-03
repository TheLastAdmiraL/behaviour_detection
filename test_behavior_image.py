"""Test behavior detection (violence, armed persons, dangerous objects) on images."""

import argparse
import sys
from pathlib import Path
import cv2
from ultralytics import YOLO


def detect_armed_persons(detections, iou_threshold=0.3):
    """
    Check if any person is holding a weapon (overlapping bounding boxes).
    
    Returns:
        list of tuples: [(person_box, weapon_type), ...]
    """
    armed_persons = []
    
    # Separate persons and weapons
    persons = []
    weapons = []
    
    for det in detections:
        class_id = int(det['class_id'])
        class_name = det['class_name']
        box = det['bbox']
        
        # Check if it's a person (adjust class_id based on your model)
        # For COCO: person is class 0
        # For weapon_det model: person is class 3
        if class_name.lower() == 'person':
            persons.append(det)
        # Check if it's a weapon
        elif class_name.lower() in ['knife', 'pistol', 'rifle', 'scissors']:
            weapons.append(det)
    
    # Check for overlaps
    for person in persons:
        px1, py1, px2, py2 = person['bbox']
        for weapon in weapons:
            wx1, wy1, wx2, wy2 = weapon['bbox']
            
            # Calculate intersection over union
            x1 = max(px1, wx1)
            y1 = max(py1, wy1)
            x2 = min(px2, wx2)
            y2 = min(py2, wy2)
            
            if x1 < x2 and y1 < y2:
                intersection = (x2 - x1) * (y2 - y1)
                weapon_area = (wx2 - wx1) * (wy2 - wy1)
                
                # If weapon box overlaps significantly with person
                if weapon_area > 0 and intersection / weapon_area > iou_threshold:
                    armed_persons.append((person, weapon['class_name'].upper()))
    
    return armed_persons


def test_image(image_path, weapon_model_path=None, violence_model_path=None, 
               conf=0.1, violence_threshold=0.5, save_output=True):
    """Test behavior detection on a single image."""
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}")
        return
    
    print(f"Testing image: {image_path}")
    print("=" * 70)
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Failed to load image")
        return
    
    h, w = img.shape[:2]
    annotated = img.copy()
    
    # --- WEAPON DETECTION ---
    weapon_detections = []
    armed_persons = []
    
    if weapon_model_path and Path(weapon_model_path).exists():
        print(f"\n[1/2] Loading weapon detection model: {weapon_model_path}")
        weapon_model = YOLO(weapon_model_path)
        print(f"      Classes: {weapon_model.names}")
        
        print(f"      Running weapon detection (conf={conf})...")
        results = weapon_model(image_path, conf=conf, verbose=False)
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            print(f"      ✓ Detected {len(results[0].boxes)} objects")
            
            # Parse detections
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = weapon_model.names[class_id]
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()
                
                detection = {
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': bbox
                }
                weapon_detections.append(detection)
                
                print(f"        - {class_name}: {confidence:.1%}")
            
            # Check for armed persons
            armed_persons = detect_armed_persons(weapon_detections, iou_threshold=0.2)
            
            # Draw detections
            for det in weapon_detections:
                x1, y1, x2, y2 = map(int, det['bbox'])
                class_name = det['class_name']
                conf = det['confidence']
                
                # Check if this is a weapon
                is_weapon = class_name.lower() in ['knife', 'pistol', 'rifle', 'scissors']
                
                # Color: RED for weapons, GREEN for others
                color = (0, 0, 255) if is_weapon else (0, 255, 0)
                
                # Draw box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Label
                if is_weapon:
                    label = f"DANGER: {class_name.upper()}"
                else:
                    label = f"{class_name}"
                
                # Background for label
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(annotated, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw armed persons
            if armed_persons:
                print(f"\n      🚨 ARMED PERSONS DETECTED: {len(armed_persons)}")
                for person, weapon_type in armed_persons:
                    x1, y1, x2, y2 = map(int, person['bbox'])
                    
                    # Draw thick RED box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 4)
                    
                    # Label
                    label = f"ARMED: {weapon_type}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0] + 10, y1), (0, 0, 255), -1)
                    cv2.putText(annotated, label, (x1 + 5, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    print(f"        - Person with {weapon_type}")
        else:
            print("      ✗ No weapons or persons detected")
    else:
        print(f"\n[1/2] Weapon model not found at: {weapon_model_path}")
        print("      Skipping weapon detection")
    
    # --- VIOLENCE CLASSIFICATION ---
    violence_prob = 0.0
    violence_detected = False
    
    if violence_model_path and Path(violence_model_path).exists():
        print(f"\n[2/2] Loading violence classification model: {violence_model_path}")
        violence_model = YOLO(violence_model_path)
        
        print(f"      Running violence classification...")
        results = violence_model(image_path, verbose=False)
        
        # Get probabilities
        probs = results[0].probs
        if probs is not None:
            # Assuming class 1 is "violence" and class 0 is "nonviolence"
            violence_prob = float(probs.data[1]) if len(probs.data) > 1 else 0.0
            violence_detected = violence_prob > violence_threshold
            
            print(f"      Violence probability: {violence_prob:.1%}")
            if violence_detected:
                print(f"      🚨 VIOLENCE DETECTED! (threshold: {violence_threshold:.0%})")
            else:
                print(f"      ✓ No violence detected (threshold: {violence_threshold:.0%})")
        
        # Draw violence bar
        bar_height = 30
        bar_y = h - bar_height - 10
        bar_width = w - 20
        
        # Background
        cv2.rectangle(annotated, (10, bar_y), (10 + bar_width, bar_y + bar_height), 
                     (50, 50, 50), -1)
        
        # Violence bar
        if violence_prob > 0:
            bar_fill = int(bar_width * violence_prob)
            color = (0, 0, 255) if violence_detected else (0, 165, 255)
            cv2.rectangle(annotated, (10, bar_y), (10 + bar_fill, bar_y + bar_height), 
                         color, -1)
        
        # Text
        text = f"Violence: {violence_prob:.1%}"
        cv2.putText(annotated, text, (20, bar_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Banner if violence detected
        if violence_detected:
            banner_height = 60
            cv2.rectangle(annotated, (0, 0), (w, banner_height), (0, 0, 255), -1)
            text = "!!! VIOLENCE DETECTED !!!"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(annotated, text, (text_x, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    else:
        print(f"\n[2/2] Violence model not found at: {violence_model_path}")
        print("      Skipping violence classification")
    
    # --- SUMMARY ---
    print("\n" + "=" * 70)
    print("DETECTION SUMMARY:")
    print("-" * 70)
    
    if weapon_detections:
        weapons = [d for d in weapon_detections if d['class_name'].lower() in ['knife', 'pistol', 'rifle', 'scissors']]
        persons = [d for d in weapon_detections if d['class_name'].lower() == 'person']
        
        if weapons:
            print(f"⚠️  Dangerous Objects: {len(weapons)}")
            for w in weapons:
                print(f"    - {w['class_name'].upper()} ({w['confidence']:.1%})")
        
        if armed_persons:
            print(f"🚨 Armed Persons: {len(armed_persons)}")
            for _, weapon_type in armed_persons:
                print(f"    - Person with {weapon_type}")
        
        if not weapons and not armed_persons:
            print("✓ No dangerous objects or armed persons")
    else:
        print("✓ No objects detected")
    
    if violence_model_path and Path(violence_model_path).exists():
        if violence_detected:
            print(f"🚨 Violence: DETECTED ({violence_prob:.1%})")
        else:
            print(f"✓ Violence: Not detected ({violence_prob:.1%})")
    
    print("=" * 70)
    
    # --- DISPLAY & SAVE ---
    print("\nDisplaying annotated image... (Press any key to close)")
    cv2.imshow(f"Behavior Detection - {Path(image_path).name}", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if save_output:
        output_path = Path(image_path).parent / f"result_{Path(image_path).name}"
        cv2.imwrite(str(output_path), annotated)
        print(f"✓ Saved annotated image to: {output_path}")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Test behavior detection (violence, armed persons, weapons) on images"
    )
    
    parser.add_argument(
        "image",
        type=str,
        help="Path to image file to test"
    )
    
    parser.add_argument(
        "--weapon-model",
        type=str,
        default="runs/weapon_det/weights/best.pt",
        help="Path to weapon detection model (default: runs/weapon_det/weights/best.pt)"
    )
    
    parser.add_argument(
        "--violence-model",
        type=str,
        default=None,
        help="Path to violence classification model (e.g., runs/violence_cls/train/weights/best.pt)"
    )
    
    parser.add_argument(
        "--conf",
        type=float,
        default=0.1,
        help="Confidence threshold for weapon detection (default: 0.1)"
    )
    
    parser.add_argument(
        "--violence-threshold",
        type=float,
        default=0.5,
        help="Violence detection threshold (default: 0.5)"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save annotated output image"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    test_image(
        image_path=args.image,
        weapon_model_path=args.weapon_model,
        violence_model_path=args.violence_model,
        conf=args.conf,
        violence_threshold=args.violence_threshold,
        save_output=not args.no_save
    )
