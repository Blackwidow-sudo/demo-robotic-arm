import config
import gradio as gr
import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def clear():
    input_image.value = None
    output_image.value = None

def predict(image, left_top_x, left_top_y, right_top_x, right_top_y, left_bottom_x, left_bottom_y, right_bottom_x, right_bottom_y, width, height):
    model = YOLO(config.get('MODEL_NAME')).to(device)
    results = model.predict(image)

    for r in results:
        image_array = r.plot(boxes=True)
        pil_image = Image.fromarray(image_array[..., ::-1])

    draw = ImageDraw.Draw(pil_image)

    draw.line([(left_top_x, left_top_y), (right_top_x, right_top_y)], fill='red', width=2)
    draw.line([(right_top_x, right_top_y), (right_bottom_x, right_bottom_y)], fill='red', width=2)
    draw.line([(right_bottom_x, right_bottom_y), (left_bottom_x, left_bottom_y)], fill='red', width=2)
    draw.line([(left_bottom_x, left_bottom_y), (left_top_x, left_top_y)], fill='red', width=2)

    return pil_image, [left_top_x, left_top_y, right_top_x, right_top_y, left_bottom_x, left_bottom_y, right_bottom_x, right_bottom_y, width, height]

with gr.Blocks(css='footer {visibility: hidden}') as demo:
    with gr.Row():
        input_image = gr.Image(type='pil', label='Input Image')
        output_image = gr.Image(type='pil', label='Output Image')

    with gr.Accordion('Calibration', open=False):
        with gr.Row():
            with gr.Row():
                left_top_x = gr.Number(label='Left-Top X', value=10)
                left_top_y = gr.Number(label='Left-Top Y', value=10)

            with gr.Row():
                right_top_x = gr.Number(label='Right-Top X', value=100)
                right_top_y = gr.Number(label='Right-Top Y', value=10)

        with gr.Row():
            with gr.Row():
                left_bottom_x = gr.Number(label='Left-Bottom X', value=10)
                left_bottom_y = gr.Number(label='Left-Bottom Y', value=100)

            with gr.Row():
                right_bottom_x = gr.Number(label='Right-Bottom X', value=100)
                right_bottom_y = gr.Number(label='Right-Bottom Y', value=100)

        with gr.Row():
            width = gr.Number(label='Width', value=0)
            height = gr.Number(label='Height', value=0)

        output_text = gr.Textbox(label='Output Text')

    with gr.Row():
        button_clear = gr.Button(value='Clear')
        button_submit = gr.Button(value='Submit', variant='primary')

    button_clear.click(fn=clear)
    button_submit.click(fn=predict, inputs=[input_image, left_top_x, left_top_y, right_top_x, right_top_y, left_bottom_x, left_bottom_y, right_bottom_x, right_bottom_y, width, height], outputs=[output_image, output_text])

if __name__ == '__main__':
    demo.queue().launch(root_path='/video')
