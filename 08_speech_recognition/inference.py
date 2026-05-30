import os
from pathlib import Path

import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


DEFAULT_MODEL_ID = os.environ.get("ASR_MODEL_ID", "openai/whisper-small")
DEFAULT_CACHE_DIR = os.environ.get("ASR_MODEL_DIR", "models/whisper_small")


class SpeechRecognizer:
    def __init__(self, model_id=DEFAULT_MODEL_ID, cache_dir=DEFAULT_CACHE_DIR):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model_id = model_id
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.processor = None
        self.error_message = None

        try:
            self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=str(self.cache_dir))
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                cache_dir=str(self.cache_dir),
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
            )
            self.model.to(self.device)
            self.model.eval()
            print(f"Loaded ASR model: {model_id}")
        except Exception as exc:
            self.error_message = str(exc)
            print(f"Failed to load ASR model {model_id}: {exc}")

    def _load_audio(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        return waveform.squeeze(0).numpy()

    def predict(self, audio_path):
        if self.model is None or self.processor is None:
            return f"模型未就绪，请先运行 python train.py 预下载中文 ASR 模型。错误信息: {self.error_message}"

        audio_array = self._load_audio(audio_path)
        inputs = self.processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(device=self.device, dtype=self.torch_dtype)

        with torch.inference_mode():
            predicted_ids = self.model.generate(
                input_features,
                language="zh",
                task="transcribe",
            )

        text = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
        )[0].strip()
        return text or "未识别到清晰语音，请重试。"


if __name__ == "__main__":
    recognizer = SpeechRecognizer()
    if recognizer.error_message:
        print(recognizer.error_message)
    else:
        print(f"ASR model ready: {recognizer.model_id}")
