import random
from pathlib import Path
import cv2
import numpy as np
import albumentations as A
import yaml

# Constants
IMAGE_TYPES = ('.jpg', '.jpeg', '.png')
TARGET_SIZE = (640, 640)          # final image size (width, height)
NUM_SYNTHETIC_PER_TAIL = 5        # how many synthetic images to generate per tail instance

def load_yolo_segmentation_masks(label_path, image_shape):
    """Load YOLO segmentation masks and return list of binary masks and class ids."""
    h, w = image_shape[:2]
    masks = []
    class_ids = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            class_ids.append(int(parts[0]))
            points = []
            for i in range(1, len(parts), 2):
                x = float(parts[i]) * w
                y = float(parts[i+1]) * h
                points.append([x, y])
            points = np.array(points, dtype=np.int32)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [points], 1)
            masks.append(mask)
    return masks, class_ids

def save_yolo_segmentation_masks(masks, class_ids, output_path, image_shape):
    """Save binary masks as YOLO segmentation polygons (normalized)."""
    h, w = image_shape[:2]
    with open(output_path, 'w') as f:
        for mask, class_id in zip(masks, class_ids):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if len(contour) < 3:
                    continue
                norm_points = []
                for point in contour[:, 0, :]:
                    x = point[0] / w
                    y = point[1] / h
                    x = max(0.0, min(1.0, x))
                    y = max(0.0, min(1.0, y))
                    norm_points.extend([x, y])
                line = f"{class_id} " + " ".join([f"{p:.6f}" for p in norm_points])
                f.write(line + '\n')

def extract_tail_object(image, mask):
    """Extract tail region: image pixels where mask==1, background becomes black."""
    tail_img = image.copy()
    tail_img[mask == 0] = 0
    return tail_img

def composite_tail_on_background(tail_img, tail_mask, background, position):
    """
    Place tail_img on background using tail_mask at given (x,y) top-left position.
    Returns composite image and the resulting mask (binary) on the background.
    """
    bh, bw = background.shape[:2]
    th, tw = tail_img.shape[:2]
    x, y = position
    # Ensure within bounds
    x = max(0, min(x, bw - tw))
    y = max(0, min(y, bh - th))

    # Region of interest on background
    roi = background[y:y+th, x:x+tw].copy()
    # Create mask for the roi (same size as tail)
    mask_roi = tail_mask.astype(bool)
    # Blend
    roi[mask_roi] = tail_img[mask_roi]
    composite = background.copy()
    composite[y:y+th, x:x+tw] = roi

    # Resulting mask (binary) on full background
    full_mask = np.zeros((bh, bw), dtype=np.uint8)
    full_mask[y:y+th, x:x+tw][mask_roi] = 1

    return composite, full_mask

def generate_synthetic_dataset(source_dataset, background_dir, output_dir,
                               target_size=TARGET_SIZE,
                               num_per_tail=NUM_SYNTHETIC_PER_TAIL):
    """
    Generate synthetic dataset by pasting tails from source_dataset onto random backgrounds.
    """
    source_path = Path(source_dataset)
    bg_path = Path(background_dir)
    out_path = Path(output_dir)

    # Prepare output directories
    for split in ['train', 'valid']:
        (out_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (out_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Load all background images
    background_images = list(bg_path.glob("*.*"))
    background_images = [p for p in background_images if p.suffix.lower() in IMAGE_TYPES]
    if not background_images:
        raise RuntimeError(f"No background images found in {background_dir}")

    # Define albumentations pipeline for tail transformations
    tail_transform = A.Compose([
        A.RandomRotate90(p=0.3),
        A.Affine(scale=(0.6, 1.4), rotate=(-30, 30), translate_percent=(-0.2, 0.2),
                 shear=(-10, 10), p=0.8),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3)
    ])

    # Process each split (train/valid) from source dataset
    for split in ['train', 'valid']:
        src_images = source_path / split / 'images'
        src_labels = source_path / split / 'labels'
        if not src_images.exists():
            continue

        image_files = list(src_images.glob("*.*"))
        image_files = [f for f in image_files if f.suffix.lower() in IMAGE_TYPES]

        for img_file in image_files:
            label_file = src_labels / f"{img_file.stem}.txt"
            if not label_file.exists():
                continue

            # Load original image and masks
            image = cv2.imread(str(img_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            masks, class_ids = load_yolo_segmentation_masks(label_file, image.shape)

            # For each mask in this image (usually one tail)
            for idx, (mask, cls_id) in enumerate(zip(masks, class_ids)):
                # Extract tail object (cut out from original image)
                tail_img = extract_tail_object(image, mask)
                # Original tail mask is binary
                tail_mask = mask.astype(np.uint8) * 255

                # Generate several synthetic versions
                for synth_id in range(num_per_tail):
                    # Apply albumentations to tail and its mask together
                    transformed = tail_transform(image=tail_img, mask=tail_mask)
                    aug_tail = transformed['image']
                    aug_mask = transformed['mask']  # still binary (0/255)

                    # Ensure mask is binary
                    aug_mask = (aug_mask > 128).astype(np.uint8) * 255

                    # If mask becomes empty, skip
                    if np.sum(aug_mask) == 0:
                        continue

                    # Resize tail to fit within target size (optional, keep aspect ratio?)
                    # We'll resize to a random size between 0.1 and 0.4 of target size
                    th, tw = aug_mask.shape
                    scale = random.uniform(0.1, 0.4)
                    new_h = max(10, int(th * scale))
                    new_w = max(10, int(tw * scale))
                    aug_tail = cv2.resize(aug_tail, (new_w, new_h))
                    aug_mask = cv2.resize(aug_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

                    # Load a random background and resize to target size
                    bg_img_path = random.choice(background_images)
                    background = cv2.imread(str(bg_img_path))
                    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
                    background = cv2.resize(background, target_size)

                    # Random position (ensure tail fits)
                    max_x = target_size[0] - new_w
                    max_y = target_size[1] - new_h
                    if max_x < 0 or max_y < 0:
                        continue  # tail too large after resize, skip
                    pos_x = random.randint(0, max_x)
                    pos_y = random.randint(0, max_y)

                    # Composite onto background
                    composite_img, composite_mask = composite_tail_on_background(
                        aug_tail, aug_mask, background, (pos_x, pos_y)
                    )

                    # Convert composite mask back to YOLO polygon format
                    # composite_mask is binary (0/255)
                    # We need to find its contours and save
                    contours, _ = cv2.findContours(composite_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not contours:
                        continue
                    # Use the largest contour (should be the tail)
                    largest_contour = max(contours, key=cv2.contourArea)
                    # Create a new mask from that contour to ensure single polygon
                    final_mask = np.zeros_like(composite_mask)
                    cv2.drawContours(final_mask, [largest_contour], -1, 1, -1)
                    # Save mask and image
                    out_img_name = f"{img_file.stem}_tail{idx}_synth{synth_id}.jpg"
                    out_lbl_name = out_img_name.replace('.jpg', '.txt')
                    out_img_path = out_path / split / 'images' / out_img_name
                    out_lbl_path = out_path / split / 'labels' / out_lbl_name

                    # Save composite image (convert back to BGR)
                    composite_img_bgr = cv2.cvtColor(composite_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(out_img_path), composite_img_bgr)

                    # Save YOLO label
                    save_yolo_segmentation_masks([final_mask], [cls_id], out_lbl_path, composite_img.shape[:2])

                    print(f"Generated: {out_img_path}")

def create_dataset_yaml(output_path, class_names):
    """Create dataset.yaml for the synthetic dataset."""
    path = Path(output_path).absolute()
    config = {
        'path': str(path),
        'train': 'train/images',
        'val': 'valid/images',
        'nc': len(class_names),
        'names': class_names
    }
    yaml_path = path / 'dataset.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

if __name__ == "__main__":
    # Example usage – adjust paths to your environment
    source_dataset = "../datasets/cats_yolo"          # original dataset with tail masks
    background_dir = "../datasets/backgrounds"        # folder with random background images (any scenes)
    output_dataset = "../datasets/cats_yolo_synthetic"

    generate_synthetic_dataset(source_dataset, background_dir, output_dataset,
                               target_size=(640, 640), num_per_tail=5)
    create_dataset_yaml(output_dataset, class_names=['tail'])
    print("Synthetic dataset generation complete.")