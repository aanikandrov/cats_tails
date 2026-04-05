from pathlib import Path

from ultralytics import YOLO

from scripts.utils.utils import choose_device, print_sep


def valid(model_path, data_yaml, device="cpu"):

    model_path = Path(model_path)
    if not model_path.exists():
        print("Модель не найдена")
        return 0

    model = YOLO(str(model_path))

    val_results = model.val(
        data=data_yaml,
        device=device,
        split='val'
    )

    return val_results


if __name__ == "__main__":
    print_sep()
    device = choose_device()
    print_sep()

    model_path = '../../models/final/best.pt'
    data_path = '../../data/cats_yolo/dataset.yaml'

    result = valid(model_path, data_path, device)

    print_sep()

    if hasattr(result, 'seg'):
        print(f"mAP50-95: {result.seg.map:.2f}")

    print_sep()