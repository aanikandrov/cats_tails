from pathlib import Path
from ultralytics import YOLO

from utils.utils import print_sep, choose_device


def train(model_path, data_yaml, device="cpu"):

    model_path = Path(model_path)
    if not model_path.exists():
        YOLO(model_path.name).export()
        model_path.parent.mkdir(parents=True, exist_ok=True)
        Path(model_path.name).rename(model_path)

    model = YOLO(str(model_path))

    # заморозка backbone
    freeze_layers = list(range(10))

    model.train(
        data=str(data_yaml),
        name='model_cat_',
        project='../models',

        imgsz=640,
        rect=True,

        epochs=120,
        save_period=10,

        batch=16,
        dropout=0.3,
        freeze=freeze_layers,

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        degrees=30,
        translate=0.2,
        scale=0.5,
        shear=10,
        fliplr=0.5,

        workers=4,
        device=device,
        verbose=True,
    )



if __name__ == "__main__":
    print_sep()

    model_path = '../models/pretrained/yolov8n-seg.pt'
    data_path = '../data/cats_yolo/dataset.yaml'
    device = choose_device()

    train(model_path, data_path, device)

    print_sep()