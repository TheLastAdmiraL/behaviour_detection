"""
Custom Dataset Training for YOLOv8
==================================

This script helps you:
1. Prepare a custom dataset
2. Train YOLOv8 on your data
3. Use the trained model in the behavior detection system

Dataset Sources:
- Roboflow: https://roboflow.com (search for "knife detection", "weapon detection")
- Kaggle: https://kaggle.com/datasets
- Open Images: https://storage.googleapis.com/openimages/web/index.html
"""

import os
import shutil
from pathlib import Path
from ultralytics import YOLO


def create_dataset_structure():
    """Create the folder structure for a custom dataset."""
    
    base = Path("datasets/custom")
    
    folders = [
        base / "train" / "images",
        base / "train" / "labels",
        base / "val" / "images",
        base / "val" / "labels",
    ]
    
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        print(f"Created: {folder}")
    
    # Create data.yaml
    yaml_content = """# Custom Dataset Configuration
path: ./datasets/custom
train: train/images
val: val/images

# Classes - modify these for your dataset
names:
  0: knife
  1: gun
  2: scissors

# Number of classes
nc: 3
"""
    
    yaml_path = base / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"Created: {yaml_path}")
    
    print("\n" + "="*60)
    print("Dataset structure created!")
    print("="*60)
    print("""
Next steps:
1. Add your images to:
   - datasets/custom/train/images/  (80% of images)
   - datasets/custom/val/images/    (20% of images)

2. Add corresponding label files to:
   - datasets/custom/train/labels/
   - datasets/custom/val/labels/

3. Label format (YOLO format):
   <class_id> <x_center> <y_center> <width> <height>
   All values normalized 0-1

   Example label file (image001.txt):
   0 0.45 0.52 0.12 0.08

4. Edit datasets/custom/data.yaml to match your classes

5. Run training:
   python train_custom.py --train
""")


def download_roboflow_dataset(api_key, workspace, project, version):
    """
    Download a dataset from Roboflow.
    
    Get your API key from: https://app.roboflow.com/settings/api
    
    Example:
        download_roboflow_dataset(
            api_key="YOUR_API_KEY",
            workspace="your-workspace",
            project="knife-detection",
            version=1
        )
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Installing roboflow...")
        os.system("pip install roboflow")
        from roboflow import Roboflow
    
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project)
    dataset = project.version(version).download("yolov8")
    
    print(f"Dataset downloaded to: {dataset.location}")
    return dataset.location


def train_model(data_yaml="datasets/custom/data.yaml", epochs=50, imgsz=640, batch=16):
    """
    Train YOLOv8 on custom dataset.
    
    Args:
        data_yaml: Path to data.yaml file
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
    """
    print("\n" + "="*60)
    print("TRAINING YOLOv8 ON CUSTOM DATASET")
    print("="*60)
    
    # Load pretrained model (transfer learning)
    model = YOLO("yolov8n.pt")  # Start from pretrained nano model
    
    print(f"Data config: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    
    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name="custom_detector",
        patience=10,  # Early stopping
        save=True,
        plots=True,
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best model saved to: runs/detect/custom_detector/weights/best.pt")
    print("\nTo use this model in behavior detection:")
    print('  python run_behaviour.py --source 0 --show --model runs/detect/custom_detector/weights/best.pt')
    
    return results


def test_custom_model(model_path, source="0"):
    """Test the custom trained model."""
    
    print(f"\nLoading custom model: {model_path}")
    model = YOLO(model_path)
    
    print("Running detection (press 'q' to quit)...")
    results = model(source, show=True, conf=0.25)


def update_pipeline_model(model_path):
    """
    Update the detector configuration to use custom model.
    
    This modifies the default model in the detector.
    """
    print(f"\nTo use your custom model, run:")
    print(f'  python run_behaviour.py --source 0 --show --model "{model_path}"')
    
    # Or modify detector directly
    print("\nOr modify yolo_object_detection/detectors.py:")
    print(f'  Change: model_name="yolov8n.pt"')
    print(f'  To:     model_name="{model_path}"')


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Custom Dataset Training for YOLOv8")
    parser.add_argument("--setup", action="store_true", help="Create dataset folder structure")
    parser.add_argument("--train", action="store_true", help="Train model on custom dataset")
    parser.add_argument("--test", type=str, help="Test a trained model (provide model path)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--data", type=str, default="datasets/custom/data.yaml", help="Data config file")
    parser.add_argument("--roboflow", action="store_true", help="Download from Roboflow")
    parser.add_argument("--api-key", type=str, help="Roboflow API key")
    parser.add_argument("--workspace", type=str, help="Roboflow workspace")
    parser.add_argument("--project", type=str, help="Roboflow project")
    parser.add_argument("--version", type=int, default=1, help="Roboflow dataset version")
    
    args = parser.parse_args()
    
    if args.setup:
        create_dataset_structure()
    elif args.train:
        train_model(data_yaml=args.data, epochs=args.epochs)
    elif args.test:
        test_custom_model(args.test)
    elif args.roboflow:
        if not all([args.api_key, args.workspace, args.project]):
            print("Error: --api-key, --workspace, and --project required for Roboflow download")
        else:
            download_roboflow_dataset(args.api_key, args.workspace, args.project, args.version)
    else:
        print(__doc__)
        print("""
Commands:
=========

1. Create dataset structure:
   python train_custom.py --setup

2. Download from Roboflow:
   python train_custom.py --roboflow --api-key YOUR_KEY --workspace your-ws --project knife-detection

3. Train model:
   python train_custom.py --train --epochs 50

4. Test trained model:
   python train_custom.py --test runs/detect/custom_detector/weights/best.pt

Popular Roboflow Datasets for Weapons:
- "knife-detection" 
- "weapon-detection"
- "dangerous-objects"
""")
