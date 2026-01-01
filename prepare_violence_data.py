"""
Phase 3: Violence Classification - Data Preparation Script
===========================================================

This script converts the "Real Life Violence Situations" video dataset
into a YOLO-compatible image classification format.

Dataset structure expected:
    datasets/real_life_violence/
    ├── Violence/
    │   ├── video1.mp4
    │   ├── video2.avi
    │   └── ...
    └── NonViolence/
        ├── video1.mp4
        ├── video2.avi
        └── ...

Output structure:
    datasets/violence_classification/
    ├── train/
    │   ├── violence/
    │   └── nonviolence/
    └── val/
        ├── violence/
        └── nonviolence/

Usage:
    python prepare_violence_data.py
    python prepare_violence_data.py --input datasets/real_life_violence --output datasets/violence_classification
    python prepare_violence_data.py --frame-interval 5  # Extract every 5th frame
"""

import os
import cv2
import random
import argparse
from pathlib import Path
from tqdm import tqdm


def extract_frames_from_video(video_path, output_folder, frame_interval=10, prefix=""):
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to the video file
        output_folder: Folder to save extracted frames
        frame_interval: Extract 1 frame every N frames (default: 10)
        prefix: Prefix for output filenames
    
    Returns:
        List of saved frame paths
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"  Warning: Cannot open {video_path}")
        return []
    
    saved_frames = []
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract every Nth frame
        if frame_count % frame_interval == 0:
            # Create filename
            filename = f"{prefix}_frame_{saved_count:05d}.jpg"
            filepath = output_folder / filename
            
            # Resize frame for faster training (optional)
            # frame = cv2.resize(frame, (224, 224))
            
            # Save frame
            cv2.imwrite(str(filepath), frame)
            saved_frames.append(filepath)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    return saved_frames


def prepare_violence_dataset(
    input_dir="datasets/real_life_violence",
    output_dir="datasets/violence_classification",
    frame_interval=10,
    train_split=0.8,
    seed=42
):
    """
    Prepare the violence classification dataset.
    
    Args:
        input_dir: Path to input dataset with Violence/NonViolence folders
        output_dir: Path to output YOLO classification dataset
        frame_interval: Extract 1 frame every N frames
        train_split: Fraction of data for training (0.8 = 80%)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Verify input structure
    violence_dir = input_path / "Violence"
    nonviolence_dir = input_path / "NonViolence"
    
    if not violence_dir.exists():
        print(f"Error: Violence folder not found at {violence_dir}")
        print("Expected structure:")
        print("  datasets/real_life_violence/")
        print("  ├── Violence/")
        print("  └── NonViolence/")
        return False
    
    if not nonviolence_dir.exists():
        print(f"Error: NonViolence folder not found at {nonviolence_dir}")
        return False
    
    # Create output structure
    train_violence = output_path / "train" / "violence"
    train_nonviolence = output_path / "train" / "nonviolence"
    val_violence = output_path / "val" / "violence"
    val_nonviolence = output_path / "val" / "nonviolence"
    
    for folder in [train_violence, train_nonviolence, val_violence, val_nonviolence]:
        folder.mkdir(parents=True, exist_ok=True)
        print(f"Created: {folder}")
    
    # Video file extensions
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    
    # Process Violence videos
    print("\n" + "="*60)
    print("Processing VIOLENCE videos...")
    print("="*60)
    
    violence_videos = [
        f for f in violence_dir.iterdir() 
        if f.suffix.lower() in video_extensions
    ]
    
    # Shuffle and split videos
    random.shuffle(violence_videos)
    split_idx = int(len(violence_videos) * train_split)
    train_violence_videos = violence_videos[:split_idx]
    val_violence_videos = violence_videos[split_idx:]
    
    print(f"Found {len(violence_videos)} violence videos")
    print(f"  Training: {len(train_violence_videos)} videos")
    print(f"  Validation: {len(val_violence_videos)} videos")
    
    # Extract frames from training violence videos
    train_violence_frames = 0
    for video in tqdm(train_violence_videos, desc="Training violence"):
        prefix = video.stem
        frames = extract_frames_from_video(
            video, train_violence, frame_interval, prefix
        )
        train_violence_frames += len(frames)
    
    # Extract frames from validation violence videos
    val_violence_frames = 0
    for video in tqdm(val_violence_videos, desc="Validation violence"):
        prefix = video.stem
        frames = extract_frames_from_video(
            video, val_violence, frame_interval, prefix
        )
        val_violence_frames += len(frames)
    
    # Process NonViolence videos
    print("\n" + "="*60)
    print("Processing NON-VIOLENCE videos...")
    print("="*60)
    
    nonviolence_videos = [
        f for f in nonviolence_dir.iterdir() 
        if f.suffix.lower() in video_extensions
    ]
    
    # Shuffle and split videos
    random.shuffle(nonviolence_videos)
    split_idx = int(len(nonviolence_videos) * train_split)
    train_nonviolence_videos = nonviolence_videos[:split_idx]
    val_nonviolence_videos = nonviolence_videos[split_idx:]
    
    print(f"Found {len(nonviolence_videos)} non-violence videos")
    print(f"  Training: {len(train_nonviolence_videos)} videos")
    print(f"  Validation: {len(val_nonviolence_videos)} videos")
    
    # Extract frames from training non-violence videos
    train_nonviolence_frames = 0
    for video in tqdm(train_nonviolence_videos, desc="Training non-violence"):
        prefix = video.stem
        frames = extract_frames_from_video(
            video, train_nonviolence, frame_interval, prefix
        )
        train_nonviolence_frames += len(frames)
    
    # Extract frames from validation non-violence videos
    val_nonviolence_frames = 0
    for video in tqdm(val_nonviolence_videos, desc="Validation non-violence"):
        prefix = video.stem
        frames = extract_frames_from_video(
            video, val_nonviolence, frame_interval, prefix
        )
        val_nonviolence_frames += len(frames)
    
    # Print summary
    print("\n" + "="*60)
    print("DATASET PREPARATION COMPLETE!")
    print("="*60)
    print(f"\nOutput directory: {output_path}")
    print(f"\nTraining set:")
    print(f"  Violence:     {train_violence_frames:,} frames")
    print(f"  Non-Violence: {train_nonviolence_frames:,} frames")
    print(f"  Total:        {train_violence_frames + train_nonviolence_frames:,} frames")
    print(f"\nValidation set:")
    print(f"  Violence:     {val_violence_frames:,} frames")
    print(f"  Non-Violence: {val_nonviolence_frames:,} frames")
    print(f"  Total:        {val_violence_frames + val_nonviolence_frames:,} frames")
    print(f"\nFrame interval: 1 frame every {frame_interval} frames")
    print(f"Train/Val split: {train_split*100:.0f}%/{(1-train_split)*100:.0f}%")
    
    print("\n" + "="*60)
    print("Next step: Train the model with:")
    print("  python train_violence_cls.py")
    print("="*60)
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare violence classification dataset from videos"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="datasets/real_life_violence",
        help="Input directory with Violence/NonViolence folders"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="datasets/violence_classification",
        help="Output directory for YOLO classification format"
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=10,
        help="Extract 1 frame every N frames (default: 10)"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of data for training (default: 0.8)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    prepare_violence_dataset(
        input_dir=args.input,
        output_dir=args.output,
        frame_interval=args.frame_interval,
        train_split=args.train_split,
        seed=args.seed
    )
