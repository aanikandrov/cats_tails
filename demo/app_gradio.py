from pathlib import Path

import gradio as gr
from ultralytics import YOLO

from scripts.core.predict import predict
from scripts.utils.mask import apply_mask_and_contour

MODEL_PATH = Path(__file__).parent.parent / "models/final/best.pt"

def predict_for_gradio(model_path, image_array, conf=0.25):
    """ Выполняет инференс из numpy-массива (для Gradio) """

    if image_array is None:
        return None

    original_image = image_array.copy()

    results = predict(model_path, image_array, conf=conf)

    result_img = apply_mask_and_contour(original_image, results)

    return result_img


def segment_tail(image):

    return predict_for_gradio(str(MODEL_PATH), image)


with gr.Blocks(title="Tail Segmentation") as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Загрузите фото", type="numpy")

        with gr.Column():
            output_image = gr.Image(label="Результат")

    input_image.change(
        fn=segment_tail,
        inputs=input_image,
        outputs=output_image
    )

if __name__ == "__main__":
    demo.launch()