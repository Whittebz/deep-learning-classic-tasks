import gradio as gr
from inference import Summarizer

summarizer = Summarizer()

def generate_summary(text):
    if not text.strip():
        return "Please enter an article to summarize."
    return summarizer.summarize(text)

with gr.Blocks(title="07 Text Summarization") as demo:
    gr.Markdown("# 07 Text Summarization (T5-small)")
    gr.Markdown("Enter a long English news article below to generate a short summary.")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(lines=10, placeholder="Paste a news article here...", label="Original Article")
            submit_btn = gr.Button("Summarize")
        with gr.Column():
            text_output = gr.Textbox(lines=5, label="Generated Summary")
            
    submit_btn.click(fn=generate_summary, inputs=text_input, outputs=text_output)
    
    if summarizer.model is None:
        gr.Markdown("### ⚠️ Warning: Model weights not found. Please run `python train.py` first.")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
