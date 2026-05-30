import os

os.environ['NO_PROXY'] = '127.0.0.1,localhost'
os.environ['no_proxy'] = '127.0.0.1,localhost'

import gradio as gr
from inference import ImageGenerator

generator = ImageGenerator()


def generate_images(num_images):
    img, text = generator.generate(int(num_images))
    return img, text


with gr.Blocks(title="09 Image Generation - DCGAN") as demo:
    gr.Markdown("# 09 Image Generation (DCGAN)")
    gr.Markdown("Click the button to generate synthetic handwritten digits from random noise.")

    with gr.Row():
        with gr.Column():
            num_images = gr.Slider(minimum=4, maximum=64, step=4, value=16, label="Number of Images to Generate (Grid)")
            submit_btn = gr.Button("Generate")
        with gr.Column():
            image_output = gr.Image(type="pil", label="Generated Output Grid")
            text_output = gr.Textbox(label="Status")

    submit_btn.click(fn=generate_images, inputs=num_images, outputs=[image_output, text_output])

    if generator.model is None:
        gr.Markdown("### ⚠️ Warning: Model weights not found. Please run `python train.py` first.")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=6006, share=False)
