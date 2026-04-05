import json
from pathlib import Path
import shutil
import yaml

# SPLITS = ['train', 'valid', 'test']
SPLITS = ['train', 'valid']


def coco_to_yolo(dataset_root_path, output_root_path):
    """ Конвертирует COCO JSON в формат YOLO segmentation"""

    Path(output_root_path).mkdir(parents=True, exist_ok=True)
    all_class_names = []

    for split_folder in SPLITS:
        split_path = Path(dataset_root_path) / split_folder

        if not split_path.exists():
            continue

        coco_json_path = split_path / "_annotations.coco.json"

        if not coco_json_path.exists():
            continue

        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)

        # category_id в class_id
        category_to_class = {}
        class_names = []
        for idx, category in enumerate(coco_data['categories']):
            category_to_class[category['id']] = idx
            class_names.append(category['name'])

        if not all_class_names:
            all_class_names = class_names

        # Словари image_id в (ширина, высота, имя) и аннотации
        image_info = {img['id']: img for img in coco_data['images']}

        annotations_by_image = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)

        # Создаем папки
        output_split_path = Path(output_root_path) / split_folder
        labels_dir = output_split_path / 'labels'
        images_dir = output_split_path / 'images'
        labels_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)

        converted_count = 0
        for img_id, annotations in annotations_by_image.items():
            img = image_info[img_id]
            img_width, img_height = img['width'], img['height']

            txt_path = labels_dir / f"{Path(img['file_name']).stem}.txt"

            with open(txt_path, 'w') as f:
                for ann in annotations:
                    if not ann['segmentation']:
                        continue

                    class_id = category_to_class[ann['category_id']]
                    segmentation = ann['segmentation'][0]

                    normalized_points = []
                    for i in range(0, len(segmentation), 2):
                        x = max(0.0, min(1.0, segmentation[i] / img_width))
                        y = max(0.0, min(1.0, segmentation[i + 1] / img_height))
                        normalized_points.extend([x, y])

                    f.write(f"{class_id} " + " ".join([f"{p:.6f}" for p in normalized_points]) + '\n')

            converted_count += 1

            # Копируем изображение
            src_image_path = split_path / img['file_name']
            dst_image_path = images_dir / img['file_name']
            if src_image_path.exists():
                shutil.copy2(src_image_path, dst_image_path)

    # Создаем dataset.yaml
    abs_path = Path(output_root_path).absolute()
    dataset_config = {
        'path': str(abs_path),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(all_class_names),
        'names': all_class_names
    }

    yaml_path = abs_path / 'dataset.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


if __name__ == "__main__":
    dataset_root_path = r"../../data/cats.v9i.coco-segmentation"
    output_path = r"../../data/cats_yolo"

    coco_to_yolo(dataset_root_path, output_path)