"""
Phase 3: Violence Classification - Training Script
===================================================

Train a YOLOv8 Classification model on the prepared violence dataset.

Prerequisites:
    1. Run prepare_violence_data.py first to extract frames
    2. Dataset should be in datasets/violence_classification/

Usage:
    python train_violence_cls.py
    python train_violence_cls.py --epochs 50 --imgsz 224
    python train_violence_cls.py --model yolov8s-cls.pt  # Use larger model
"""

import os
import argparse
from pathlib import Path
from ultralytics import YOLO


def train_violence_classifier(
    data_dir="datasets/violence_classification",
    model_name="yolov8n-cls.pt",
    epochs=30,
    imgsz=224,
    batch=32,
    project="runs/violence_cls",
    name="train"
):
    """
    Train YOLOv8 classification model for violence detection.
    
    Args:
        data_dir: Path to dataset with train/val folders
        model_name: Base YOLO classification model
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
        project: Output project directory
        name: Experiment name
    """
    print("\n" + "="*60)
    print("VIOLENCE CLASSIFICATION TRAINING")
    print("="*60)
    
    # Verify dataset exists
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: Dataset not found at {data_path}")
        print("Run prepare_violence_data.py first!")
        return None
    
    # Check train/val structure
    train_dir = data_path / "train"
    val_dir = data_path / "val"
    
    if not train_dir.exists() or not val_dir.exists():
        print(f"Error: Expected train/ and val/ folders in {data_path}")
        return None
    
    # Count images
    train_violence = len(list((train_dir / "violence").glob("*.jpg")))
    train_nonviolence = len(list((train_dir / "nonviolence").glob("*.jpg")))
    val_violence = len(list((val_dir / "violence").glob("*.jpg")))
    val_nonviolence = len(list((val_dir / "nonviolence").glob("*.jpg")))
    
    print(f"\nDataset: {data_path}")
    print(f"Training images: {train_violence + train_nonviolence:,}")
    print(f"  - Violence: {train_violence:,}")
    print(f"  - Non-Violence: {train_nonviolence:,}")
    print(f"Validation images: {val_violence + val_nonviolence:,}")
    print(f"  - Violence: {val_violence:,}")
    print(f"  - Non-Violence: {val_nonviolence:,}")
    
    print(f"\nModel: {model_name}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    
    # Load pretrained classification model
    print(f"\nLoading {model_name}...")
    model = YOLO(model_name)
    
    # Train
    print("\nStarting training...")
    print("="*60)
    
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        patience=10,  # Early stopping
        save=True,
        plots=True,
        verbose=True,
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    # Find best model path
    best_model = Path(project) / name / "weights" / "best.pt"
    last_model = Path(project) / name / "weights" / "last.pt"
    
    print(f"\nBest model: {best_model}")
    print(f"Last model: {last_model}")
    
    # Print usage instructions
    print("\n" + "="*60)
    print("HOW TO USE THE TRAINED MODEL")
    print("="*60)
    print(f"""
1. Test the model:
   python train_violence_cls.py --test {best_model}

2. Use in behavior detection:
   python run_behaviour.py --source 0 --show --violence-model {best_model}

3. Python code:
   from ultralytics import YOLO
   
   model = YOLO("{best_model}")
   results = model.predict(frame)
   
   # Get violence probability
   probs = results[0].probs
   violence_prob = probs.data[0].item()  # Index 0 = violence
   nonviolence_prob = probs.data[1].item()  # Index 1 = nonviolence
   
   print(f"Violence: {{violence_prob:.1%}}")
""")
    
    return results


def test_violence_classifier(model_path, source="0"):
    """
    Test the trained violence classifier.
    
    Args:
        model_path: Path to trained model weights
        source: Video source (0 for webcam, or video path)
    """
    import cv2
    
    print(f"\nLoading model: {model_path}")
    model = YOLO(model_path)
    
    # Get class names
    class_names = model.names
    print(f"Classes: {class_names}")
    
    # Open video source
    if source == "0":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Cannot open source {source}")
        return
    
    print("\nRunning violence detection (press 'q' to quit)...")
    print("="*60)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run classification
        results = model.predict(frame, verbose=False)
        
        # Get probabilities
        probs = results[0].probs
        top1_idx = probs.top1
        top1_conf = probs.top1conf.item()
        class_name = class_names[top1_idx]
        
        # Get violence probability specifically
        # Note: Class order depends on folder alphabetical order
        # 'nonviolence' (0) comes before 'violence' (1) alphabetically
        violence_idx = 1 if 'violence' in class_names.get(1, '').lower() else 0
        violence_prob = probs.data[violence_idx].item()
        
        # Draw results
        if violence_prob > 0.5:
            color = (0, 0, 255)  # Red for violence
            label = f"VIOLENCE: {violence_prob:.1%}"
            # Draw warning banner
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 200), -1)
            cv2.putText(frame, "!!! VIOLENCE DETECTED !!!", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        else:
            color = (0, 255, 0)  # Green for non-violence
            label = f"Normal: {1-violence_prob:.1%}"
        
        # Draw probability bar
        bar_width = int(violence_prob * 200)
        cv2.rectangle(frame, (10, 70), (210, 100), (100, 100, 100), -1)
        cv2.rectangle(frame, (10, 70), (10 + bar_width, 100), (0, 0, 255), -1)
        cv2.putText(frame, f"Violence: {violence_prob:.1%}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.imshow("Violence Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 violence classification model"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="datasets/violence_classification",
        help="Path to classification dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n-cls.pt",
        help="Base YOLO classification model (yolov8n-cls.pt, yolov8s-cls.pt, etc.)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=224,
        help="Image size for training"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Test a trained model (provide path to weights)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Source for testing (0 for webcam, or video path)"
    )
    
    args = parser.parse_args()
    
    if args.test:
        test_violence_classifier(args.test, args.source)
    else:
        train_violence_classifier(
            data_dir=args.data,
            model_name=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch
        )
