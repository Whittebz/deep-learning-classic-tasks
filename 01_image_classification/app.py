import gradio as gr
from inference import ImageClassifier

classifier = ImageClassifier()


def predict(image):
    if image is None:
        return "Please upload an image."
    return classifier.predict(image)


with gr.Blocks(title="01 Image Classification - CIFAR-100") as demo:
    gr.Markdown("# 01 Image Classification (CIFAR-100)")
    gr.Markdown("Upload an image. The model returns top-3 CIFAR-100 class predictions.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            submit_btn = gr.Button("Predict")
        with gr.Column():
            label_output = gr.Label(num_top_classes=3, label="Predictions")

    submit_btn.click(fn=predict, inputs=image_input, outputs=label_output)

    if classifier.model is None:
        gr.Markdown("### Warning: Model weights not found. Please run `python train.py` in this directory first.")


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)
