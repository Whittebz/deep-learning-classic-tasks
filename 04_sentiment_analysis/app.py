import gradio as gr
from inference import SentimentClassifier

clf = SentimentClassifier()

def predict(text):
    if not text.strip():
        return {"Please enter some text": 1.0}
    return clf.predict(text)

with gr.Blocks(title="04 Text Classification - Sentiment Analysis") as demo:
    gr.Markdown("# 04 Text Classification (Sentiment Analysis)")
    gr.Markdown("Enter a movie review in English to classify its sentiment as Positive or Negative (trained on IMDB).")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(lines=5, placeholder="I loved this movie!", label="Input Text")
            submit_btn = gr.Button("Classify")
        with gr.Column():
            label_output = gr.Label(num_top_classes=2, label="Sentiment Probabilities")
            
    submit_btn.click(fn=predict, inputs=text_input, outputs=label_output)
    
    if clf.model is None:
        gr.Markdown("### ⚠️ Warning: Model weights not found. Please run `python train.py` first.")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=6006, share=False)
