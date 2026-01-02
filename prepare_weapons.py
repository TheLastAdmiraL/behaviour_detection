import os
from datasets import load_dataset
from tqdm import tqdm

# --- CONFIGURATION ---
OUTPUT_DIR = "datasets/weapon_detection_clean"
# This mapping merges the 29 messy classes into 4 clean ones
CLASS_MAPPING = {
    # PISTOL Group
    'Pistol': 0, 'pistol': 0, 'pistols': 0, 'handgun': 0, 'Handgun': 0, 
    'Guns': 0, 'guns': 0, 'weapon': 0, # Assuming generic weapon is likely a gun in this dataset
    
    # KNIFE Group
    'Knife': 1, 'Knife_Deploy': 1, 'Knife_Weapon': 1, 'Stabbing': 1,
    
    # RIFLE/HEAVY Group
    'Rifle': 2, 'rifle': 2, 'Shotgun': 2, 'shotgun': 2, 
    'Long guns': 2, 'Heavy Gun': 2, 'heavyweapon': 2,
    
    # PERSON Group
    'Person': 3, 'person': 3, 'Aggressor': 3, 'Victim': 3
}
FINAL_CLASSES = ['pistol', 'knife', 'rifle', 'person']

def convert_to_yolo():
    print("Downloading dataset (this may take a moment)...")
    dataset = load_dataset("Subh775/WeaponDetection")
    
    # Get the original class names list
    orig_classes = dataset['train'].features['objects']['category'].feature.names
    
    splits = ['train', 'validation', 'test']
    
    for split in splits:
        print(f"Processing {split} data...")
        
        # Create directories
        img_dir = f"{OUTPUT_DIR}/images/{split}"
        lbl_dir = f"{OUTPUT_DIR}/labels/{split}"
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        
        data = dataset[split]
        
        for item in tqdm(data):
            image = item['image']
            image_id = item['image_id']
            objects = item['objects']
            
            # Prepare label content
            label_lines = []
            img_width, img_height = image.size
            
            has_valid_obj = False
            
            for idx, cat_id in enumerate(objects['category']):
                original_name = orig_classes[cat_id]
                
                # Check if this class is in our mapping
                if original_name in CLASS_MAPPING:
                    new_class_id = CLASS_MAPPING[original_name]
                    
                    # Get bbox (HuggingFace format: [x_min, y_min, w, h])
                    x_min, y_min, w, h = objects['bbox'][idx]
                    
                    # Convert to YOLO (Normalized: x_center, y_center, w, h)
                    x_center = (x_min + w / 2) / img_width
                    y_center = (y_min + h / 2) / img_height
                    w_norm = w / img_width
                    h_norm = h / img_height
                    
                    # Clip values to ensure they are between 0 and 1
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    w_norm = max(0, min(1, w_norm))
                    h_norm = max(0, min(1, h_norm))
                    
                    label_lines.append(f"{new_class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
                    has_valid_obj = True
            
            # Only save if there are valid objects
            if has_valid_obj:
                # Save Image
                image.save(f"{img_dir}/{image_id}.jpg")
                
                # Save Label
                with open(f"{lbl_dir}/{image_id}.txt", "w") as f:
                    f.write("\n".join(label_lines))

    # Create data.yaml
    yaml_content = f"""path: ../{OUTPUT_DIR} # Relative path for simplicity
train: images/train
val: images/validation
test: images/test

nc: {len(FINAL_CLASSES)}
names: {FINAL_CLASSES}
"""
    with open(f"{OUTPUT_DIR}/data.yaml", "w") as f:
        f.write(yaml_content)
        
    print(f"\nDone! Data prepared in {OUTPUT_DIR}")
    print(f"data.yaml created at {OUTPUT_DIR}/data.yaml")

if __name__ == "__main__":
    convert_to_yolo()