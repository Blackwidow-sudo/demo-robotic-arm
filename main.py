import gradio as gr
import torch
from PIL import Image
from ultralytics import YOLO

MODEL_NAME = 'drogerie_8_yolov8s-seg_8k_v1.pt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def clear():
    input_image.value = None
    output_image.value = None

def predict(image):
    model = YOLO(MODEL_NAME).to(device)
    results = model.predict(image)

    for r in results:
        image_array = r.plot(boxes=True)
        pil_image = Image.fromarray(image_array[..., ::-1])

    return pil_image

with gr.Blocks(css='footer {visibility: hidden}') as demo:
    with gr.Row():
        input_image = gr.Image(type='pil', label='Input Image')
        output_image = gr.Image(type='pil', label='Output Image')

    with gr.Row():
        button_clear = gr.Button(value='Clear')
        button_submit = gr.Button(value='Submit', variant='primary')

    button_clear.click(fn=clear)
    button_submit.click(fn=predict, inputs=[input_image], outputs=[output_image])

    with gr.Accordion('Calibration', open=False):
        with gr.Row():
            with gr.Row():
                gr.Number(label='Left-Top X', value=0)
                gr.Number(label='Left-Top Y', value=0)

            with gr.Row():
                gr.Number(label='Right-Top X', value=0)
                gr.Number(label='Right-Top Y', value=0)

        with gr.Row():
            with gr.Row():
                gr.Number(label='Left-Bottom X', value=0)
                gr.Number(label='Left-Bottom Y', value=0)

            with gr.Row():
                gr.Number(label='Right-Bottom X', value=0)
                gr.Number(label='Right-Bottom Y', value=0)

        with gr.Row():
            gr.Number(label='Width', value=0)
            gr.Number(label='Height', value=0)

if __name__ == '__main__':
    demo.queue().launch(root_path='/video')
