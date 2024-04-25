import config
import cv2 as cv
import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO

custom_css = """
    footer {
        visibility: hidden;
    }

    .app.gradio-container {
        max-width: 100% !important;
    }
"""
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


def predict(image, left_top_x, left_top_y, right_top_x, right_top_y, left_bottom_x, left_bottom_y, right_bottom_x, right_bottom_y, width, height):
    model = YOLO(config.get('MODEL_NAME')).to(device)
    results = model.predict(image)

    left_top = (left_top_x, left_top_y)
    right_top = (right_top_x, right_top_y)
    left_bottom = (left_bottom_x, left_bottom_y)
    right_bottom = (right_bottom_x, right_bottom_y)

    for r in results:
        image_array = r.plot(boxes=True)
        pil_image = Image.fromarray(image_array[..., ::-1])

    if config.get_bool('DRAW_CALIBRATION'):
        draw = ImageDraw.Draw(pil_image)

        draw.line([left_top, right_top], fill='red', width=2)
        draw.line([right_top, right_bottom], fill='red', width=2)
        draw.line([right_bottom, left_bottom], fill='red', width=2)
        draw.line([left_bottom, left_top], fill='red', width=2)

    warped_image = unscew_img(pil_image, left_top, right_top, left_bottom, right_bottom)

    return warped_image if config.get_bool('OUTPUT_WARPED') else pil_image, [left_top_x, left_top_y, right_top_x, right_top_y, left_bottom_x, left_bottom_y, right_bottom_x, right_bottom_y, width, height]


def unscew_img(image: Image, top_left, top_right, bottom_left, bottom_right) -> Image:
    """Skew image so that the table box is parallel to the image edges"""
    np_img = np.array(image)
    height, width = np_img.shape[:2]

    src_pts = np.float32([list(top_left), list(top_right), list(bottom_left), list(bottom_right)])
    dest_pts = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    matrix = cv.getPerspectiveTransform(src_pts, dest_pts)
    warped_img = cv.warpPerspective(np_img, matrix, (width, height))

    return Image.fromarray(warped_img)


with gr.Blocks(css=custom_css) as demo:
    with gr.Row():
        input_image = gr.Image(type='pil', label='Input Image', sources=['webcam'], streaming=True)
        output_image = gr.Image(type='pil', label='Output Image')

    with gr.Row():
        input_audio = gr.Audio(label='Input Audio', sources=['microphone'])
        output_text = gr.Textbox(label='Output Text')

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

    with gr.Row():
        button_clear = gr.ClearButton(components=[input_image, output_image, output_text], value='Clear')
        button_submit = gr.Button(value='Submit', variant='primary')

    button_submit.click(fn=predict, inputs=[input_image, left_top_x, left_top_y, right_top_x, right_top_y, left_bottom_x, left_bottom_y, right_bottom_x, right_bottom_y, width, height], outputs=[output_image, output_text])


if __name__ == '__main__':
    demo.queue().launch(root_path='/video')
