import config
import cv2 as cv
import gradio as gr
import json
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import pipeline
import json
from ultralytics import YOLO

custom_css = """
    footer {
        visibility: hidden;
    }

    .app.gradio-container {
        max-width: 100% !important;
    }
"""
device = 'cuda' if torch.cuda.is_available() else 'mps' if config.get('ALLOW_MPS') and torch.backends.mps.is_available() else 'cpu'
transcriber = pipeline('automatic-speech-recognition', model='openai/whisper-small')


def predict(image, audio, left_top_x, left_top_y, right_top_x, right_top_y, left_bottom_x, left_bottom_y, right_bottom_x, right_bottom_y, width, height):
    left_top = (left_top_x, left_top_y)
    right_top = (right_top_x, right_top_y)
    left_bottom = (left_bottom_x, left_bottom_y)
    right_bottom = (right_bottom_x, right_bottom_y)

    image = unscew_img(image, left_top, right_top, left_bottom, right_bottom) if config.get_bool('OUTPUT_WARPED') else image
    model = YOLO(config.get('MODEL_NAME')).to(device)
    results = model.predict(image)

    for r in results:
        image_array = r.plot(boxes=True)
        pil_image = Image.fromarray(image_array[..., ::-1])

    if config.get_bool('DRAW_CALIBRATION'):
        draw = ImageDraw.Draw(pil_image)

        draw.line([left_top, right_top], fill='red', width=2)
        draw.line([right_top, right_bottom], fill='red', width=2)
        draw.line([right_bottom, left_bottom], fill='red', width=2)
        draw.line([left_bottom, left_top], fill='red', width=2)

    json_results = to_json_results(results[0], (pil_image.size[0] / width + pil_image.size[1] / height) / 2)

    if config.get_bool('LOG_JSON'):
        with open('results.json', 'w') as f:
            f.write(json_results)

    audio_text = transcribe(audio)

    return pil_image, audio_text, json_results


def transcribe(audio):
    if audio is None:
        return ''
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    return transcriber({'sampling_rate': sr, 'raw': y})['text']


def unscew_img(image: Image, top_left, top_right, bottom_left, bottom_right) -> Image:
    """Skew image so that the table box is parallel to the image edges"""
    np_img = np.array(image)
    height, width = np_img.shape[:2]

    src_pts = np.float32([list(top_left), list(top_right), list(bottom_left), list(bottom_right)])
    dest_pts = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    matrix = cv.getPerspectiveTransform(src_pts, dest_pts)
    warped_img = cv.warpPerspective(np_img, matrix, (width, height))

    return Image.fromarray(warped_img)


def to_json_results(result, pxl_per_cm) -> str:
    """Generate JSON string from the results of the model prediction"""
    src_json = json.loads(result.tojson())
    result = []

    for detected in src_json:
        box = detected['box']

        result.append({
            'id': detected['name'],
            'confidence': detected['confidence'],
            'area': (box['x2'] - box['x1']) / pxl_per_cm * (box['y2'] - box['y1']) / pxl_per_cm,
            'bb': detected['box'],
            'bb_cm': {
                'x1': box['x1'] / pxl_per_cm,
                'y1': box['y1'] / pxl_per_cm,
                'x2': box['x2'] / pxl_per_cm,
                'y2': box['y2'] / pxl_per_cm
            }
        })

    return json.dumps(result, indent=2)


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
            width = gr.Number(label='Width', value=config.get_as('CALIBRATION_WIDTH', float))
            height = gr.Number(label='Height', value=config.get_as('CALIBRATION_HEIGHT', float))

        with gr.Column():
            output_json = gr.Textbox(label='Output JSON', )

    with gr.Row():
        button_clear = gr.ClearButton(components=[input_image, output_image, output_text], value='Clear')
        button_submit = gr.Button(value='Submit', variant='primary')

    button_submit.click(fn=predict, inputs=[input_image, input_audio, left_top_x, left_top_y, right_top_x, right_top_y, left_bottom_x, left_bottom_y, right_bottom_x, right_bottom_y, width, height], outputs=[output_image, output_text, output_json])


if __name__ == '__main__':
    demo.queue().launch(root_path='/video')
