import os

os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

import gradio as gr
from inference import SegmentationModel

seg_model = SegmentationModel()


def predict(image):
    if image is None:
        return None, None, "Please upload an image."
    overlay, mask, status = seg_model.predict(image)
    return overlay, mask, status


with gr.Blocks(title="03 Semantic Segmentation - Pet Foreground") as demo:
    gr.Markdown("# 03 Semantic Segmentation")
    gr.Markdown("Upload a cat or dog image. The model separates pet foreground from background environment.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            submit_btn = gr.Button("Segment")
        with gr.Column():
            image_output = gr.Image(type="pil", label="Overlay")
            mask_output = gr.Image(type="pil", label="Binary Mask")
            text_output = gr.Textbox(label="Status")

    submit_btn.click(fn=predict, inputs=image_input, outputs=[image_output, mask_output, text_output])

    if seg_model.model is None:
        gr.Markdown("### Warning: Model weights not found. Please run `python train.py` first.")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=6006, share=False)
