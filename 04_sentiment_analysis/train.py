import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 国内镜像

import tarfile
import urllib.request
from pathlib import Path

import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


def build_acl_imdb_dataset(data_root: str = "./data"):
    """Download and parse Stanford ACL IMDB dataset as non-synthetic fallback."""
    data_dir = Path(data_root)
    data_dir.mkdir(parents=True, exist_ok=True)

    archive_path = data_dir / "aclImdb_v1.tar.gz"
    extracted_root = data_dir / "aclImdb"

    if not extracted_root.exists():
        if not archive_path.exists():
            url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
            print(f"Downloading fallback dataset from: {url}")
            urllib.request.urlretrieve(url, archive_path)

        print("Extracting aclImdb archive...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=data_dir)

    def load_split(split_name: str):
        texts, labels = [], []
        for label_name, label_id in (("pos", 1), ("neg", 0)):
            folder = extracted_root / split_name / label_name
            if not folder.exists():
                continue
            for p in folder.glob("*.txt"):
                texts.append(p.read_text(encoding="utf-8", errors="ignore"))
                labels.append(label_id)
        return Dataset.from_dict({"text": texts, "label": labels})

    train_ds = load_split("train")
    test_ds = load_split("test")
    if len(train_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError("Failed to parse aclImdb dataset from extracted files.")

    return DatasetDict({"train": train_ds, "test": test_ds})


def train():
    if os.environ.get("OMP_NUM_THREADS") in (None, "", "0"):
        os.environ["OMP_NUM_THREADS"] = "1"

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    print("Downloading and preparing IMDB dataset...")
    candidates = ["imdb", "stanfordnlp/imdb", "huggingface/imdb"]
    last_exc = None
    dataset = None
    for name in candidates:
        try:
            dataset = load_dataset(name)
            print(f"Loaded dataset with identifier: {name}")
            break
        except Exception as e:
            last_exc = e
            print(f"Failed to load with identifier '{name}': {e}")

    if dataset is None:
        print("HF IMDB unavailable. Falling back to Stanford ACL IMDB download...")
        if last_exc is not None:
            print(f"Last HF error: {last_exc}")
        dataset = build_acl_imdb_dataset("./data")
        print("Loaded fallback dataset: aclImdb_v1")

    small_train_dataset = dataset["train"].shuffle(seed=42).select(range(min(1000, len(dataset["train"]))))
    small_eval_dataset = dataset["test"].shuffle(seed=42).select(range(min(200, len(dataset["test"]))))

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def preprocess_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    print("Tokenizing datasets...")
    tokenized_train = small_train_dataset.map(preprocess_function, batched=True)
    tokenized_eval = small_eval_dataset.map(preprocess_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    model.to(device)

    common_args = dict(
        output_dir="./models/results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_steps=10,
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

    save_path = "models/distilbert_imdb"
    print(f"Saving fine-tuned model to {save_path}...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Training completed.")


if __name__ == "__main__":
    train()
