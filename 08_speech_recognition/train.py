import os
from pathlib import Path

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


DEFAULT_MODEL_ID = os.environ.get("ASR_MODEL_ID", "openai/whisper-small")
DEFAULT_CACHE_DIR = os.environ.get("ASR_MODEL_DIR", "models/whisper_small")


def prepare_model(model_id=DEFAULT_MODEL_ID, cache_dir=DEFAULT_CACHE_DIR):
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading ASR processor: {model_id}")
    AutoProcessor.from_pretrained(model_id, cache_dir=str(cache_path))

    print(f"Downloading ASR model weights: {model_id}")
    AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        cache_dir=str(cache_path),
        low_cpu_mem_usage=True,
    )

    print(f"Model prepared in {cache_path.resolve()}")


if __name__ == "__main__":
    prepare_model()
