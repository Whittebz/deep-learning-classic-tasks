import csv
import os
import urllib.request
import zipfile
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

REPO_DIR = Path(__file__).resolve().parent
TMP_ROOT = Path(os.environ.get("TASK4_TMP_ROOT", "/root/autodl-tmp/04_sentiment_analysis"))
HF_HOME = TMP_ROOT / "hf_home"
HF_DATASETS_CACHE = TMP_ROOT / "hf_datasets"
HF_HUB_CACHE = TMP_ROOT / "hf_hub"
TORCH_HOME = TMP_ROOT / "torch_home"
TRAINING_OUTPUT_DIR = TMP_ROOT / "trainer_output"
PRETRAINED_MODEL_DIR = TMP_ROOT / "pretrained" / "bert-base-chinese"
SAVE_PATH = TMP_ROOT / "models" / "bert_chinese_sentiment"
DATA_DIR = REPO_DIR / "data"
DATASET_DIR = DATA_DIR / "chnsenticorp"
DATASET_ARCHIVE = DATA_DIR / "chnsenticorp.zip"
DATASET_URLS = [
    "https://github.com/ymcui/Chinese-BERT-wwm/raw/master/data/chnsenticorp/chnsenticorp.zip",
    "https://raw.githubusercontent.com/ymcui/Chinese-BERT-wwm/master/data/chnsenticorp/chnsenticorp.zip",
]

os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_DATASETS_CACHE))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HUB_CACHE))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HUB_CACHE))
os.environ.setdefault("TORCH_HOME", str(TORCH_HOME))

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "bert-base-chinese"
MODEL_FILE_URLS = {
    "config.json": [
        "https://huggingface.co/bert-base-chinese/resolve/main/config.json",
        "https://hf-mirror.com/bert-base-chinese/resolve/main/config.json",
    ],
    "tokenizer_config.json": [
        "https://huggingface.co/bert-base-chinese/resolve/main/tokenizer_config.json",
        "https://hf-mirror.com/bert-base-chinese/resolve/main/tokenizer_config.json",
    ],
    "vocab.txt": [
        "https://huggingface.co/bert-base-chinese/resolve/main/vocab.txt",
        "https://hf-mirror.com/bert-base-chinese/resolve/main/vocab.txt",
    ],
    "tokenizer.json": [
        "https://huggingface.co/bert-base-chinese/resolve/main/tokenizer.json",
        "https://hf-mirror.com/bert-base-chinese/resolve/main/tokenizer.json",
    ],
    "model.safetensors": [
        "https://huggingface.co/bert-base-chinese/resolve/main/model.safetensors",
        "https://hf-mirror.com/bert-base-chinese/resolve/main/model.safetensors",
    ],
}


def download_file(urls, destination: Path) -> None:
    last_exc = None
    for url in urls:
        try:
            print(f"Downloading ChnSentiCorp from {url} ...")
            urllib.request.urlretrieve(url, destination)
            print(f"Downloaded dataset archive to {destination}")
            return
        except Exception as exc:
            last_exc = exc
            print(f"Failed to download from {url}: {exc}")

    raise RuntimeError(
        "Unable to download ChnSentiCorp dataset archive. "
        f"Please check network access. Last error: {last_exc}"
    )


def extract_dataset(archive_path: Path, extract_dir: Path) -> None:
    print(f"Extracting dataset archive to {extract_dir} ...")
    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)


def ensure_pretrained_model() -> Path:
    PRETRAINED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for filename, urls in MODEL_FILE_URLS.items():
        destination = PRETRAINED_MODEL_DIR / filename
        if destination.exists() and destination.stat().st_size > 0:
            continue
        download_file(urls, destination)
    return PRETRAINED_MODEL_DIR


def find_split_file(split_name: str) -> Path:
    matches = sorted(DATASET_DIR.rglob(f"{split_name}.tsv"))
    if not matches:
        raise FileNotFoundError(f"Could not find {split_name}.tsv under {DATASET_DIR}")
    return matches[0]


def ensure_local_dataset() -> dict[str, Path]:
    expected = {name: find_split_file(name) for name in ("train", "dev", "test")} if DATASET_DIR.exists() else None
    if expected:
        return expected

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DATASET_ARCHIVE.exists():
        download_file(DATASET_URLS, DATASET_ARCHIVE)

    extract_dataset(DATASET_ARCHIVE, DATASET_DIR)
    return {name: find_split_file(name) for name in ("train", "dev", "test")}


def load_tsv_dataset(path: Path) -> Dataset:
    texts, labels = [], []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="	")
        if not reader.fieldnames:
            raise ValueError(f"{path} is missing a TSV header")

        label_key = "label" if "label" in reader.fieldnames else reader.fieldnames[0]
        text_candidates = ["text", "text_a", "sentence", "review"]
        text_key = next((key for key in text_candidates if key in reader.fieldnames), None)
        if text_key is None:
            text_columns = [name for name in reader.fieldnames if name != label_key]
            if not text_columns:
                raise ValueError(f"{path} has no text column")
            text_key = text_columns[0]

        for row in reader:
            text = (row.get(text_key) or "").strip()
            label = (row.get(label_key) or "").strip()
            if not text:
                continue
            if label not in {"0", "1", 0, 1}:
                raise ValueError(f"Invalid label '{label}' in {path}; expected 0 or 1")
            texts.append(text)
            labels.append(int(label))

    if not texts:
        raise ValueError(f"No valid samples found in {path}")

    return Dataset.from_dict({"text": texts, "label": labels})


def load_chinese_sentiment_dataset() -> DatasetDict:
    print("Preparing full ChnSentiCorp dataset...")
    split_files = ensure_local_dataset()
    print("Using local dataset files:")
    for split_name, split_path in split_files.items():
        print(f"  {split_name}: {split_path}")

    return DatasetDict({
        "train": load_tsv_dataset(split_files["train"]),
        "validation": load_tsv_dataset(split_files["dev"]),
        "test": load_tsv_dataset(split_files["test"]),
    })


def train():
    if os.environ.get("OMP_NUM_THREADS") in (None, "", "0"):
        os.environ["OMP_NUM_THREADS"] = "1"

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    print(f"Temporary training storage: {TMP_ROOT}")
    print(f"Final model path: {SAVE_PATH}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRAINING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_chinese_sentiment_dataset()
    pretrained_model_path = ensure_pretrained_model()
    print(f"Using local pretrained model files from: {pretrained_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)

    def preprocess_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    print("Tokenizing datasets...")
    tokenized_train = dataset["train"].map(preprocess_function, batched=True)
    tokenized_eval = dataset["validation"].map(preprocess_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_path, num_labels=2)
    model.to(device)

    common_args = dict(
        output_dir=str(TRAINING_OUTPUT_DIR),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        save_strategy="no",
        logging_steps=50,
        logging_dir=str(TMP_ROOT / "logs"),
        report_to=[],
    )
    try:
        training_args = TrainingArguments(eval_strategy="epoch", **common_args)
    except TypeError:
        training_args = TrainingArguments(evaluation_strategy="epoch", **common_args)

    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            processing_class=tokenizer,
        )
    except TypeError:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            tokenizer=tokenizer,
        )

    print("Starting training...")
    trainer.train()

    print(f"Saving fine-tuned model to {SAVE_PATH}...")
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print("Training completed.")


if __name__ == "__main__":
    train()
