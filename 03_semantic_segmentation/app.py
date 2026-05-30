import os

os.environ['NO_PROXY'] = '127.0.0.1,localhost'
os.environ['no_proxy'] = '127.0.0.1,localhost'

import gradio as gr
from inference import SegmentationModel

seg_model = SegmentationModel()


def predict(image):
    if image is None:
        return None, "Please upload an image."
    img_out, text_out = seg_model.predict(image)
    return img_out, text_out


with gr.Blocks(title="03 Semantic Segmentation - FCN") as demo:
    gr.Markdown("# 03 Semantic Segmentation (FCN-ResNet50)")
    gr.Markdown("Upload an image of a pet (cat or dog). Red area represents the pet, Green is the boundary.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            submit_btn = gr.Button("Segment")
        with gr.Column():
            image_output = gr.Image(type="pil", label="Segmentation Overlay")
            text_output = gr.Textbox(label="Status")

    submit_btn.click(fn=predict, inputs=image_input, outputs=[image_output, text_output])

    if seg_model.model is None:
        gr.Markdown("### ⚠️ Warning: Model weights not found. Please run `python train.py` first.")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=6006, share=False)
