import shutil
from pathlib import Path
import albumentations as A
import cv2
import numpy as np
import yaml

IMAGE_TYPES = ['.jpg', '.jpeg', '.png']
# SPLITS = ['train', 'valid', 'test']
SPLITS = ['train', 'valid']

def augment_dataset(input_path, output_path, aug_per_image=3):
    """ Аугментация датасета """

    transform = A.Compose([

        # Аффинное преобразование
        A.Affine(
            translate_percent=(-0.1, 0.1),  # сдвиг
            rotate=(-20, 20),  # поворот
            shear=(-10, 10),   # наклон
            scale=(0.8, 1.2),  # масштаб
            p=0.5
        ),

        # Яркость
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.8
        ),

        # Тон
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.5
        )
    ])

    # Проходим по сплитам
    for split in SPLITS:
        input_split = Path(input_path) / split
        output_split = Path(output_path) / split

        if not input_split.exists():
            continue

        # Создаем структуру папок
        images_output = output_split / 'images'
        labels_output = output_split / 'labels'
        images_output.mkdir(parents=True, exist_ok=True)
        labels_output.mkdir(parents=True, exist_ok=True)

        # Копируем файлы
        input_images = input_split / 'images'
        input_labels = input_split / 'labels'

        # Копируем оригинальные файлы и создаём аугментированные версии
        for img_file in input_images.glob("*.*"):
            if img_file.suffix.lower() not in IMAGE_TYPES:
                continue

            # Копируем изображение
            shutil.copy2(img_file, images_output / img_file.name)

            # Копируем label
            label_file = input_labels / f"{img_file.stem}.txt"
            if label_file.exists():
                shutil.copy2(label_file, labels_output / f"{img_file.stem}.txt")

            # Загружаем изображение
            image = cv2.imread(str(img_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Загружаем маски из YOLO формата
            if not label_file.exists():
                continue

            masks, class_ids = load_yolo_segmentation_masks(label_file, image.shape[:2])

            # Генерируем несколько аугментированных версий
            for aug_idx in range(aug_per_image):
                try:
                    # Применяем аугментации
                    augmented = transform(image=image, masks=masks)
                    aug_image = augmented['image']
                    aug_masks = augmented['masks']

                    # Сохраняем аугментированное изображение
                    aug_filename = f"{img_file.stem}_aug{aug_idx + 1}{img_file.suffix}"
                    aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(images_output / aug_filename), aug_image_bgr)

                    # Сохраняем аугментированные маски в YOLO формате
                    save_yolo_segmentation_masks(
                        aug_masks, class_ids,
                        labels_output / f"{img_file.stem}_aug{aug_idx + 1}.txt",
                        aug_image.shape[:2]
                    )

                except Exception as e:
                    print(f"Ошибка при аугментации {img_file.name}: {e}")
                    continue


def load_yolo_segmentation_masks(label_path, image_shape):
    """ Загружает YOLO segmentation маски и преобразует в бинарные маски """

    h, w = image_shape[:2]
    masks = []
    class_ids = []

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # class + минимум 3 точки
                continue

            class_ids.append(int(parts[0]))

            points = []
            for i in range(1, len(parts), 2):
                x = float(parts[i]) * w
                y = float(parts[i + 1]) * h
                points.append([x, y])

            points = np.array(points, dtype=np.int32)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [points], 1)
            masks.append(mask)

    return masks, class_ids


def save_yolo_segmentation_masks(masks, class_ids, output_path, image_shape):
    """ Сохраняет бинарные маски в YOLO segmentation формат """
    h, w = image_shape[:2]

    with open(output_path, 'w') as f:
        for mask, class_id in zip(masks, class_ids):
            # Находим контуры
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if len(contour) < 3:
                    continue

                # Нормализуем координаты
                normalized_points = []
                for point in contour[:, 0, :]:
                    x = point[0] / w
                    y = point[1] / h
                    x = max(0.0, min(1.0, x))
                    y = max(0.0, min(1.0, y))
                    normalized_points.extend([x, y])

                line = f"{class_id} " + " ".join([f"{p:.6f}" for p in normalized_points])
                f.write(line + '\n')


def create_dataset_yaml(output_path, class_names):
    """ Создает dataset.yaml для датасета """

    path = Path(output_path).absolute()

    dataset_config = {
        'path': str(path),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(class_names),
        'names': class_names
    }

    yaml_path = path / 'dataset.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


if __name__ == "__main__":

    # исходный датасет
    input_dataset = "../datasets/cats_yolo"
    # аугментированный датасет
    output_dataset = "../datasets/cats_yolo_augmented"

    # Аугментация
    augment_dataset(
        input_path=input_dataset,
        output_path=output_dataset,
        aug_per_image=5
    )

    # Создаем dataset.yaml
    create_dataset_yaml(output_dataset, class_names=['tail'])