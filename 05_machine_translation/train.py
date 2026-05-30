import json
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 国内镜像

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def train():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Prefer a translation-specific checkpoint. If we fall back to a generic
    # seq2seq model, persist the required prefix so inference stays consistent.
    model_candidates = [
        ("Helsinki-NLP/opus-mt-en-fr", ""),
        ("t5-small", "translate English to French: "),
    ]
    model_checkpoint = model_candidates[0][0]
    input_prefix = model_candidates[0][1]

    print("Downloading dataset...")
    candidates = [
        "opus_books",
        "huggingface/opus_books",
        "Helsinki-NLP/opus_books",
        "opus/opus_books",
    ]
    last_exc = None
    for name in candidates:
        try:
            raw_datasets = load_dataset(name, "en-fr")
            print(f"Loaded dataset with identifier: {name}")
            break
        except Exception as e:
            last_exc = e
            print(f"Failed to load with identifier '{name}': {e}")
    else:
        raise last_exc

    dataset = raw_datasets["train"].shuffle(seed=42)
    max_samples = min(len(dataset), 5000)
    subset = dataset.select(range(max_samples))
    split_dataset = subset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    print(f"Using {len(train_dataset)} training samples and {len(eval_dataset)} eval samples.")

    last_exc = None
    for checkpoint_name, prefix in model_candidates:
        try:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_name)
            model_checkpoint = checkpoint_name
            input_prefix = prefix
            print(f"Loaded model checkpoint: {model_checkpoint}")
            break
        except Exception as e:
            last_exc = e
            print(f"Failed to load checkpoint '{checkpoint_name}': {e}")
    else:
        raise last_exc

    model.to(device)

    max_input_length = 128
    max_target_length = 128

    def preprocess_function(examples):
        inputs = [f"{input_prefix}{ex['en']}" for ex in examples["translation"]]
        targets = [ex["fr"] for ex in examples["translation"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
        labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Tokenizing data...")
    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_eval = eval_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    common_args = dict(
        output_dir="./models/results",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=3,
        predict_with_generate=True,
        logging_steps=20,
        save_strategy="epoch",
        generation_max_length=max_target_length,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    try:
        training_args = Seq2SeqTrainingArguments(
            eval_strategy="epoch",
            **common_args,
        )
    except TypeError:
        training_args = Seq2SeqTrainingArguments(
            evaluation_strategy="epoch",
            **common_args,
        )

    try:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            data_collator=data_collator,
            processing_class=tokenizer,
        )
    except TypeError:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

    print("Starting fine-tuning...")
    trainer.train()
    eval_metrics = trainer.evaluate()
    print(f"Final eval metrics: {eval_metrics}")

    save_path = "models/translation_en_fr"
    os.makedirs(save_path, exist_ok=True)
    print(f"Saving model to {save_path}...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    with open(os.path.join(save_path, "task_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_checkpoint": model_checkpoint,
                "input_prefix": input_prefix,
                "source_language": "en",
                "target_language": "fr",
                "max_input_length": max_input_length,
                "max_target_length": max_target_length,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("Training completed.")


if __name__ == "__main__":
    train()
