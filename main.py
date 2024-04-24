import config
import cv2 as cv
import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def predict(image, left_top_x, left_top_y, right_top_x, right_top_y, left_bottom_x, left_bottom_y, right_bottom_x, right_bottom_y, width, height):
    model = YOLO(config.get('MODEL_NAME')).to(device)
    results = model.predict(image)

    for r in results:
        image_array = r.plot(boxes=True)
        pil_image = Image.fromarray(image_array[..., ::-1])

    if config.get('DRAW_CALIBRATION'):
        draw = ImageDraw.Draw(pil_image)

        draw.line([(left_top_x, left_top_y), (right_top_x, right_top_y)], fill='red', width=2)
        draw.line([(right_top_x, right_top_y), (right_bottom_x, right_bottom_y)], fill='red', width=2)
        draw.line([(right_bottom_x, right_bottom_y), (left_bottom_x, left_bottom_y)], fill='red', width=2)
        draw.line([(left_bottom_x, left_bottom_y), (left_top_x, left_top_y)], fill='red', width=2)

    source_points = np.array([[left_top_x, left_top_y], [right_top_x, right_top_y], [right_bottom_x, right_bottom_y], [left_bottom_x, left_bottom_y]])
    destination_points = np.array([[0, 0], [width, 0], [width, height], [0, height]])

    matrix = cv.getPerspectiveTransform(source_points.astype(np.float32), destination_points.astype(np.float32))
    warped_image = cv.warpPerspective(np.array(pil_image), matrix, (width, height))

    return warped_image if config.get('OUTPUT_WARPED') else pil_image, [left_top_x, left_top_y, right_top_x, right_top_y, left_bottom_x, left_bottom_y, right_bottom_x, right_bottom_y, width, height]

with gr.Blocks(css='footer {visibility: hidden}') as demo:
    with gr.Row():
        input_image = gr.Image(type='pil', label='Input Image', sources=['webcam'])
        output_image = gr.Image(type='pil', label='Output Image')

    with gr.Accordion('Calibration', open=False):
        with gr.Row():
            with gr.Row():
                left_top_x = gr.Number(label='Left-Top X', value=config.get_as('CALIBRATION_LEFT_TOP_X', int))
                left_top_y = gr.Number(label='Left-Top Y', value=config.get_as('CALIBRATION_LEFT_TOP_Y', int))

            with gr.Row():
                right_top_x = gr.Number(label='Right-Top X', value=config.get_as('CALIBRATION_RIGHT_TOP_X', int))
                right_top_y = gr.Number(label='Right-Top Y', value=config.get_as('CALIBRATION_RIGHT_TOP_Y', int))

        with gr.Row():
            with gr.Row():
                left_bottom_x = gr.Number(label='Left-Bottom X', value=config.get_as('CALIBRATION_LEFT_BOTTOM_X', int))
                left_bottom_y = gr.Number(label='Left-Bottom Y', value=config.get_as('CALIBRATION_LEFT_BOTTOM_Y', int))

            with gr.Row():
                right_bottom_x = gr.Number(label='Right-Bottom X', value=config.get_as('CALIBRATION_RIGHT_BOTTOM_X', int))
                right_bottom_y = gr.Number(label='Right-Bottom Y', value=config.get_as('CALIBRATION_RIGHT_BOTTOM_Y', int))

        with gr.Row():
            width = gr.Number(label='Width', value=config.get_as('CALIBRATION_WIDTH', int))
            height = gr.Number(label='Height', value=config.get_as('CALIBRATION_HEIGHT', int))

        output_text = gr.Textbox(label='Output Text')

    with gr.Row():
        button_clear = gr.ClearButton(components=[input_image, output_image, output_text], value='Clear')
        button_submit = gr.Button(value='Submit', variant='primary')

    button_submit.click(fn=predict, inputs=[input_image, left_top_x, left_top_y, right_top_x, right_top_y, left_bottom_x, left_bottom_y, right_bottom_x, right_bottom_y, width, height], outputs=[output_image, output_text])

if __name__ == '__main__':
    demo.queue().launch(root_path='/video')
