import gradio as gr
from inference import SpeechRecognizer

sr = SpeechRecognizer()

def recognize(audio):
    if audio is None:
        return {"Please record or upload audio": 1.0}
    return sr.predict(audio)

with gr.Blocks(title="08 Speech Recognition") as demo:
    gr.Markdown("# 08 Speech Recognition (M5 Audio CNN)")
    gr.Markdown("Upload or record a short audio clip (1 second) of a command word (e.g., 'up', 'down', 'left', 'right').")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="Input Audio")
            submit_btn = gr.Button("Recognize")
        with gr.Column():
            label_output = gr.Label(num_top_classes=3, label="Predicted Command")
            
    submit_btn.click(fn=recognize, inputs=audio_input, outputs=label_output)
    
    if sr.model is None:
        gr.Markdown("### ⚠️ Warning: Model weights not found. Please run `python train.py` first.")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=6006, share=False)
