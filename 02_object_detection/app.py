import gradio as gr
from inference import ObjectDetector

detector = ObjectDetector()

def predict(image):
    if image is None:
        return None, "Please upload an image."
    img_out, text_out = detector.predict(image)
    return img_out, text_out

with gr.Blocks(title="02 Object Detection - Faster R-CNN") as demo:
    gr.Markdown("# 02 Object Detection (Faster R-CNN)")
    gr.Markdown("Upload an image to detect pedestrians (trained on PennFudanPed).")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            submit_btn = gr.Button("Detect")
        with gr.Column():
            image_output = gr.Image(type="pil", label="Detection Result")
            text_output = gr.Textbox(label="Status")
            
    submit_btn.click(fn=predict, inputs=image_input, outputs=[image_output, text_output])
    
    if detector.model is None:
        gr.Markdown("### ⚠️ Warning: Model weights not found. Please run `python train.py` first.")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=6006, share=False)
