"""
Weapon Detection Model Training Script
Trains YOLOv8 Nano model on cleaned weapon detection dataset
Safe for RTX 3060 6GB VRAM
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train_weapon_detector(epochs=50, imgsz=640, batch=16):
    """
    Train YOLOv8 Nano model for weapon detection
    
    Args:
        epochs: Number of training epochs (default: 50)
        imgsz: Image size for training (default: 640)
        batch: Batch size - FIXED at 16 for 6GB VRAM safety
    """
    print("="*60)
    print("WEAPON DETECTION TRAINING")
    print("="*60)
    print(f"Configuration:")
    print(f"  Model: YOLOv8 Nano (yolov8n.pt)")
    print(f"  Dataset: datasets/weapon_detection_clean/data.yaml")
    print(f"  Epochs: {epochs}")
    print(f"  Image Size: {imgsz}")
    print(f"  Batch Size: {batch} (FIXED for 6GB VRAM)")
    print(f"  Output: runs/weapon_det/")
    print("="*60)
    
    # Verify dataset exists
    data_yaml = Path("datasets/weapon_detection_clean/data.yaml")
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {data_yaml}\n"
            f"Please run prepare_weapons.py first!"
        )
    
    # Load pretrained YOLOv8 Nano model
    print("\n[1/3] Loading pretrained YOLOv8 Nano model...")
    model = YOLO('yolov8n.pt')
    
    # Train the model
    print("\n[2/3] Starting training...")
    print("Training progress will be displayed below:")
    print("-" * 60)
    
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name='weapon_det',
        project='runs',
        patience=0,            # Disable early stopping (was causing premature stop)
        save=True,             # Save checkpoints
        save_period=10,        # Save checkpoint every 10 epochs
        device=0,              # Use GPU 0 (cuda:0)
        workers=4,             # Number of dataloader workers
        exist_ok=True,         # Overwrite existing run
        pretrained=True,       # Use pretrained weights
        optimizer='AdamW',     # Optimizer
        verbose=True,          # Verbose output
        seed=42,               # Random seed for reproducibility
        deterministic=False,   # Faster training (non-deterministic)
        single_cls=False,      # Multi-class detection
        rect=False,            # Rectangular training (disabled)
        cos_lr=True,           # Cosine learning rate scheduler
        close_mosaic=10,       # Disable mosaic augmentation last N epochs
        resume=False,          # Don't resume from checkpoint
        amp=True,              # Automatic Mixed Precision training
        fraction=1.0,          # Use 100% of dataset
        profile=False,         # Don't profile training
        overlap_mask=True,     # Masks overlap during training
        val=True,              # Validate during training
        plots=True,            # Save training plots
    )
    
    print("-" * 60)
    print("\n[3/3] Training complete!")
    
    # Display training results
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    best_model_path = Path("runs/weapon_det/weights/best.pt")
    last_model_path = Path("runs/weapon_det/weights/last.pt")
    
    if best_model_path.exists():
        print(f"✓ Best model saved: {best_model_path}")
        print(f"  Size: {best_model_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    if last_model_path.exists():
        print(f"✓ Last model saved: {last_model_path}")
        print(f"  Size: {last_model_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    print(f"✓ Training logs: runs/weapon_det/")
    print("="*60)
    
    return results


def validate_weapon_detector():
    """
    Validate the trained weapon detection model on test split
    """
    print("\n" + "="*60)
    print("VALIDATION ON TEST SET")
    print("="*60)
    
    best_model_path = Path("runs/weapon_det/weights/best.pt")
    
    if not best_model_path.exists():
        print("❌ Error: Best model not found!")
        print(f"   Expected: {best_model_path}")
        print("   Please train the model first.")
        return
    
    # Load the best trained model
    print(f"\nLoading best model: {best_model_path}")
    model = YOLO(str(best_model_path))
    
    # Validate on test split
    print("\nRunning validation on test split...")
    print("-" * 60)
    
    # Note: YOLO validation uses 'val' split by default
    # For test split, we need to specify the data yaml
    # and it will use the 'val' split defined in the yaml
    results = model.val(
        data="datasets/weapon_detection_clean/data.yaml",
        split='test',
        batch=16,
        imgsz=640,
        device=0,
        workers=4,
        verbose=True,
        plots=True,
        save_json=True,
        save_hybrid=False,
    )
    
    print("-" * 60)
    print("\nVALIDATION RESULTS:")
    print("="*60)
    
    # Display metrics
    metrics = results.results_dict
    
    print(f"Precision:    {metrics.get('metrics/precision(B)', 0):.4f}")
    print(f"Recall:       {metrics.get('metrics/recall(B)', 0):.4f}")
    print(f"mAP@0.5:      {metrics.get('metrics/mAP50(B)', 0):.4f}")
    print(f"mAP@0.5-0.95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
    
    print("="*60)
    print("Validation complete!")
    
    return results


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Train YOLOv8 Nano for weapon detection (RTX 3060 optimized)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default 50 epochs
  python train_weapons.py
  
  # Train with custom epochs
  python train_weapons.py --epochs 100
  
  # Only validate existing model
  python train_weapons.py --validate-only
        """
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only run validation on existing best.pt model'
    )
    
    args = parser.parse_args()
    
    try:
        if args.validate_only:
            # Only validate existing model
            validate_weapon_detector()
        else:
            # Train model
            train_weapon_detector(epochs=args.epochs)
            
            # Auto-validate after training
            print("\n" + "🔍 Running automatic validation...\n")
            validate_weapon_detector()
            
        print("\n✅ All tasks completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user (Ctrl+C)")
        print("Partial results may be saved in runs/weapon_det/")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
