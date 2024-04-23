import gradio as gr
import torch
from PIL import Image
from ultralytics import YOLO

import config

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def predict(image):
    model = YOLO(config.get('MODEL_NAME')).to(device)
    results = model.predict(image)

    for r in results:
        image_array = r.plot(boxes=True)
        pil_image = Image.fromarray(image_array[..., ::-1])

    return pil_image

app = gr.Interface(
    allow_flagging=False,
    css='footer {visibility: hidden}',
    fn=predict,
    inputs=gr.Image(type='pil', label='Input Image'),
    outputs=gr.Image(type='pil', label='Output Image'),
)

if __name__ == '__main__':
    app.queue().launch(root_path='/video')
