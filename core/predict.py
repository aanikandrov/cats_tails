from ultralytics import YOLO

from utils.utils import print_sep

def predict(model_path, image_path, conf=0.25):
    model = YOLO(model_path)
    results = model(image_path, conf=conf)
    return results


def main():
    model_path = r'../models/final/best.pt'
    image_path = r'../data/cats_another/1.jpg'

    print_sep()

    predict(model_path, image_path)

    print_sep()


if __name__ == "__main__":
    main()