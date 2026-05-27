import gradio as gr
from inference import ImageClassifier
import os

# Initialize classifier
classifier = ImageClassifier()

def predict(image):
    if image is None:
        return "Please upload an image."
    return classifier.predict(image)

# Define Gradio interface
with gr.Blocks(title="01 Image Classification - CIFAR-10") as demo:
    gr.Markdown("# 01 Image Classification (CIFAR-10)")
    gr.Markdown("Upload an image of one of the 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            submit_btn = gr.Button("Predict")
        with gr.Column():
            label_output = gr.Label(num_top_classes=3, label="Predictions")
            
    submit_btn.click(fn=predict, inputs=image_input, outputs=label_output)
    
    # If a sample model doesn't exist, show warning
    if classifier.model is None:
        gr.Markdown("### ⚠️ Warning: Model weights not found. Please run `python train.py` in this directory first.")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
