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
            gr.Dataframe(label='Left-Top Corner', datatype='number', interactive=True, col_count=2, row_count=1, headers=['X', 'Y'], value=[[0, 0]])
            gr.Dataframe(label='Right-Top Corner', datatype='number', interactive=True, col_count=2, row_count=1, headers=['X', 'Y'], value=[[0, 0]])

        with gr.Row():
            gr.Dataframe(label='Left-Bottom Corner', datatype='number', interactive=True, col_count=2, row_count=1, headers=['X', 'Y'], value=[[0, 0]])
            gr.Dataframe(label='Right-Bottom Corner', datatype='number', interactive=True, col_count=2, row_count=1, headers=['X', 'Y'], value=[[0, 0]])

if __name__ == '__main__':
    demo.queue().launch(root_path='/video')
