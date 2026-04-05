import cv2
import numpy as np

def apply_mask_and_contour(image, results, contour_color=(0, 255, 0), contour_thickness=2):
    """ Накладывает полупрозрачную маску и рисует контур на изображении """
    img_copy = image.copy()

    if results[0].masks is None:
        return img_copy

    # Получаем маски
    masks = results[0].masks.data.cpu().numpy()

    for mask in masks:
        mask_binary = (mask > 0.5).astype(np.uint8) * 255

        if mask_binary.shape != img_copy.shape[:2]:
            mask_binary = cv2.resize(mask_binary, (img_copy.shape[1], img_copy.shape[0]))

        color_mask = np.zeros_like(img_copy)
        color_mask[:, :, 1] = mask_binary

        # Накладываем полупрозрачную маску
        alpha = 0.5
        mask_3channel = cv2.cvtColor(mask_binary, cv2.COLOR_GRAY2BGR) / 255.0
        img_copy = (img_copy * (1 - alpha * mask_3channel) + alpha * color_mask).astype(np.uint8)

        # Находим и рисуем контур
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_copy, contours, -1, contour_color, contour_thickness)

    return img_copy