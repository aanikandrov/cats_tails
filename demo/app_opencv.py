import cv2

from core.predict import predict
from utils.mask import apply_mask_and_contour


def predict_and_show(model_path, image_path, conf=0.25):
    """ Выполняет инференс и показывает результат в окне OpenCV """

    original_image = cv2.imread(image_path)

    results = predict(model_path, image_path, conf=conf)

    image = results[0].plot()

    result_img = apply_mask_and_contour(original_image, results)

    cv2.namedWindow('Tail Segmentation', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Tail Segmentation', 800, 800)

    cv2.imshow('Tail Segmentation', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    model_path = r'../models/final/best.pt'
    image_path = r'../data/cats_another/3.jpg'
    conf = 0.3

    predict_and_show(model_path, image_path, conf)


if __name__ == "__main__":
    main()