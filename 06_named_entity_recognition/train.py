import json
import os
import subprocess

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from transformers import AutoConfig, AutoModelForTokenClassification, BertTokenizerFast

MODEL_REPO = "ckiplab/bert-base-chinese-ner"
SAVE_PATH = "models/bert_base_chinese_ner"
MODEL_CACHE_DIR = "models/pretrained_ckip_ner_cache"
MODEL_FILES = [
    "config.json",
    "pytorch_model.bin",
    "vocab.txt",
    "tokenizer_config.json",
    "special_tokens_map.json",
]
MODEL_BASE_URLS = [
    f"{os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com').rstrip('/')}/{MODEL_REPO}/resolve/main",
    f"https://huggingface.co/{MODEL_REPO}/resolve/main",
]


def download_with_curl(url, dest_path):
    tmp_path = dest_path + ".part"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    cmd = [
        "curl",
        "-fL",
        "--retry",
        "5",
        "--retry-delay",
        "2",
        "--connect-timeout",
        "20",
        "--max-time",
        "0",
        "-o",
        tmp_path,
        url,
    ]
    subprocess.run(cmd, check=True)
    os.replace(tmp_path, dest_path)


def ensure_model_files():
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    missing_files = [name for name in MODEL_FILES if not os.path.exists(os.path.join(MODEL_CACHE_DIR, name))]
    if not missing_files:
        print(f"Using cached model files from {MODEL_CACHE_DIR}")
        return MODEL_CACHE_DIR, "local-cache"

    last_exc = None
    for base_url in MODEL_BASE_URLS:
        try:
            print(f"Trying direct model download via: {base_url}")
            for name in missing_files:
                dest_path = os.path.join(MODEL_CACHE_DIR, name)
                if os.path.exists(dest_path):
                    continue
                file_url = f"{base_url}/{name}"
                print(f"Downloading {name} ...")
                download_with_curl(file_url, dest_path)
            print(f"Model files downloaded via: {base_url}")
            return MODEL_CACHE_DIR, base_url
        except Exception as e:
            last_exc = e
            print(f"Failed direct model download via {base_url}: {e}")
    raise RuntimeError(
        "Unable to download the Chinese NER model files from either the configured mirror or huggingface.co."
    ) from last_exc


def load_local_model(model_dir):
    config = AutoConfig.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_config(config)
    weight_path = os.path.join(model_dir, "pytorch_model.bin")
    state_dict = torch.load(weight_path, map_location="cpu")
    state_dict.pop("bert.embeddings.position_ids", None)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Missing keys during state_dict load: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys during state_dict load: {unexpected_keys}")
    return model


def train():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    print("Preparing Chinese NER model for local inference...")
    print(f"Model repo: {MODEL_REPO}")

    model_source_dir, model_download_source = ensure_model_files()
    tokenizer = BertTokenizerFast.from_pretrained(model_source_dir)
    model = load_local_model(model_source_dir)
    model.to(device)
    model.eval()

    os.makedirs(SAVE_PATH, exist_ok=True)
    tokenizer.save_pretrained(SAVE_PATH)
    model.save_pretrained(SAVE_PATH, safe_serialization=True)

    with open(os.path.join(SAVE_PATH, "training_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": "pretrained_model_cache",
                "model_repo": MODEL_REPO,
                "hf_endpoint": os.environ.get("HF_ENDPOINT", ""),
                "task": "chinese_named_entity_recognition",
                "model_cache_dir": MODEL_CACHE_DIR,
                "model_download_source": model_download_source,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Chinese NER model is ready at: {SAVE_PATH}")


if __name__ == "__main__":
    train()
