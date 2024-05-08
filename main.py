import config
import cv2 as cv
import gradio as gr
import json
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import pipeline
from ultralytics import YOLO

device = 'cuda' if torch.cuda.is_available() else 'mps' if config.get_bool('ALLOW_MPS') and torch.backends.mps.is_available() else 'cpu'
transcriber = pipeline('automatic-speech-recognition', model='openai/whisper-small')


def rank_depths(image, result):
    '''Rank the detected objects by depth (brightness) using the depth estimation model'''
    pipe = pipeline(task='depth-estimation', model='LiheYoung/depth-anything-small-hf')
    depth_img = pipe(image)['depth']

    width, height = image.size

    cv_img = cv.cvtColor(np.array(depth_img), cv.COLOR_RGB2BGR)
    resized_cv_img = cv.resize(cv_img, dsize=(width, height))

    ranking = []

    img = np.copy(resized_cv_img)

    # iterate each object contour 
    for ci, contour in enumerate(result):
        label = contour.names[contour.boxes.cls.tolist().pop()]

        binary_mask = np.zeros(img.shape[:2], np.uint8)

        # Create contour mask 
        contour_mask = contour.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
        _ = cv.drawContours(binary_mask, [contour_mask], -1, (255, 255, 255), cv.FILLED)

        # Choose one:

        # OPTION-1: Isolate object with black background
        mask3ch = cv.cvtColor(binary_mask, cv.COLOR_GRAY2BGR)
        isolated = cv.bitwise_and(mask3ch, img)

        # OPTION-2: Isolate object with transparent background (when saved as PNG)
        # isolated = np.dstack([img, binary_mask])

        # OPTIONAL: detection crop (from either OPT1 or OPT2)
        x1, y1, x2, y2 = contour.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
        iso_crop = isolated[y1:y2, x1:x2]

        # Rank the objects by brightest found pixel
        ranking.append((f'{label}_{ci}', np.max(iso_crop)))

    # Sort by brightness
    ranking.sort(key=lambda x: x[1], reverse=True)

    return ranking


def predict(image, audio, sort_order, draw_calibration, output_warped, left_top_x, left_top_y, right_top_x, right_top_y, left_bottom_x, left_bottom_y, right_bottom_x, right_bottom_y, width, height, offset_x, offset_y):
    left_top = (left_top_x, left_top_y)
    right_top = (right_top_x, right_top_y)
    left_bottom = (left_bottom_x, left_bottom_y)
    right_bottom = (right_bottom_x, right_bottom_y)

    if draw_calibration:
        draw = ImageDraw.Draw(image)

        draw.line([left_top, right_top], fill='red', width=2)
        draw.line([right_top, right_bottom], fill='red', width=2)
        draw.line([right_bottom, left_bottom], fill='red', width=2)
        draw.line([left_bottom, left_top], fill='red', width=2)

    image = warp_image(image, left_top, right_top, left_bottom, right_bottom) if output_warped else image
    model = YOLO(config.get('MODEL_NAME')).to(device)
    results = model.predict(image)

    for r in results:
        image_array = r.plot(boxes=True)
        pil_image = Image.fromarray(image_array[..., ::-1])

    json_results = to_json_results(results[0], sort_order, (pil_image.size[0] / width + pil_image.size[1] / height) / 2, offset_x, offset_y)

    depth_ranking = rank_depths(image, results[0])
    print('Depths:', depth_ranking)

    if config.get_bool('LOG_JSON'):
        with open('results.json', 'w') as f:
            f.write(json_results)

    audio_text = transcribe(audio)

    return pil_image, audio_text, json_results


def to_json_results(result, sort_order, pxl_per_cm, offset_x, offset_y) -> str:
    '''Generate JSON string from the results of the model prediction'''
    src_json = json.loads(result.tojson())
    result = []

    for detected in src_json:
        box = detected['box']

        result.append({
            'id': detected['name'],
            'area': (box['x2'] - box['x1']) / pxl_per_cm * (box['y2'] - box['y1']) / pxl_per_cm,
            'bb': detected['box'],
            'bb_cm': {
                'x1': box['x1'] / pxl_per_cm + offset_x,
                'y1': box['y1'] / pxl_per_cm + offset_y,
                'x2': box['x2'] / pxl_per_cm + offset_x,
                'y2': box['y2'] / pxl_per_cm + offset_y,
            },
            'confidence': detected['confidence'],
        })

    sorted_result = sorted(result, key=lambda x: x[sort_order.lower()], reverse=True)

    return json.dumps(sorted_result, indent=2)


def transcribe(audio):
    if audio is None:
        return ''
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    return transcriber({'sampling_rate': sr, 'raw': y})['text']


def warp_image(image: Image, top_left, top_right, bottom_left, bottom_right) -> Image:
    '''Skew image so that the table box is parallel to the image edges'''
    np_img = np.array(image)
    height, width = np_img.shape[:2]

    src_pts = np.float32([list(top_left), list(top_right), list(bottom_left), list(bottom_right)])
    dest_pts = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    matrix = cv.getPerspectiveTransform(src_pts, dest_pts)
    warped_img = cv.warpPerspective(np_img, matrix, (width, height))

    return Image.fromarray(warped_img)


with gr.Blocks(css='style.css') as demo:
    with gr.Row():
        input_image = gr.Image(type='pil', label='Input Image', sources=['webcam'], streaming=True)
        output_image = gr.Image(type='pil', label='Output Image')

    with gr.Row():
        input_audio = gr.Audio(label='Input Audio', sources=['microphone'])
        output_text = gr.Textbox(label='Output Text')

    with gr.Row():
        button_clear = gr.ClearButton(components=[input_image, output_image, output_text], value='Clear')
        button_submit = gr.Button(value='Submit', variant='primary')

    with gr.Accordion('Calibration', open=False):
        with gr.Row():
            sort_order = gr.Dropdown(label='Sort Order', choices=['Area', 'Confidence'], value='Confidence')

        with gr.Row():
            draw_calibration = gr.Checkbox(label='Draw Calibration', value=config.get_bool('DRAW_CALIBRATION'))
            output_warped = gr.Checkbox(label='Output Warped', value=config.get_bool('OUTPUT_WARPED'))

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
            with gr.Row():
                width = gr.Number(label='Width', value=config.get_as('CALIBRATION_WIDTH', float))
                height = gr.Number(label='Height', value=config.get_as('CALIBRATION_HEIGHT', float))

            with gr.Row():
                offset_x = gr.Number(label='Offset X', value=config.get_as('CALIBRATION_OFFSET_X', int))
                offset_y = gr.Number(label='Offset Y', value=config.get_as('CALIBRATION_OFFSET_Y', int))

        with gr.Column():
            mapping = gr.Dataframe(label='Mapping', col_count=2, datatype=['str', 'number'], interactive=True, type='array')
            output_json = gr.Textbox(label='Output JSON', )

    button_submit.click(fn=predict, inputs=[input_image, input_audio, sort_order, draw_calibration, output_warped, left_top_x, left_top_y, right_top_x, right_top_y, left_bottom_x, left_bottom_y, right_bottom_x, right_bottom_y, width, height, offset_x, offset_y], outputs=[output_image, output_text, output_json])


if __name__ == '__main__':
    demo.queue().launch(root_path='/video')
