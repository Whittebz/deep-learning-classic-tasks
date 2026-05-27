import gradio as gr
from inference import Translator

translator = Translator()

def translate_text(text):
    if not text.strip():
        return "Please enter text to translate."
    return translator.translate(text)

with gr.Blocks(title="05 Machine Translation - English to French") as demo:
    gr.Markdown("# 05 Machine Translation (English to French)")
    gr.Markdown("Enter English text below to translate it into French.")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(lines=5, placeholder="Enter English text...", label="English")
            submit_btn = gr.Button("Translate")
        with gr.Column():
            text_output = gr.Textbox(lines=5, label="French Translation")
            
    submit_btn.click(fn=translate_text, inputs=text_input, outputs=text_output)
    
    if translator.model is None:
        gr.Markdown("### ⚠️ Warning: Model weights not found. Please run `python train.py` first.")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)
