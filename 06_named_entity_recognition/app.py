import gradio as gr
from inference import NERModel

ner = NERModel()

def extract_entities(text):
    if not text.strip():
        return "Please enter text."
    return ner.predict(text)

with gr.Blocks(title="06 Named Entity Recognition") as demo:
    gr.Markdown("# 06 Named Entity Recognition (NER)")
    gr.Markdown("Enter text to extract Persons, Organizations, Locations, and Miscellaneous entities.")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(lines=5, placeholder="Apple is looking at buying U.K. startup for $1 billion, said Tim Cook in London.", label="Input Text")
            submit_btn = gr.Button("Extract")
        with gr.Column():
            text_output = gr.Textbox(lines=5, label="Extracted Entities")
            
    submit_btn.click(fn=extract_entities, inputs=text_input, outputs=text_output)
    
    if ner.nlp is None:
        gr.Markdown("### ⚠️ Warning: Model weights not found. Please run `python train.py` first.")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=6006, share=False)
