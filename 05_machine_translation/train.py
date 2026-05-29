import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 国内镜像

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)


def train():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # For machine translation, we'll use a pre-trained model and fine-tune it
    # on a tiny subset of English-French data to demonstrate the pipeline.
    model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
    print("Downloading dataset...")
    # Using 'opus_books' dataset which has en-fr pairs
    # Try multiple dataset identifiers to handle different HuggingFace Hub versions
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
        # If none of the candidates worked, re-raise the last exception for visibility
        raise last_exc

    # Take a tiny subset for rapid demonstration
    small_train = raw_datasets["train"].shuffle(seed=42).select(range(500))
    # Note: opus_books doesn't have a default split, so we split it manually
    split_dataset = small_train.train_test_split(test_size=0.1)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    # Prefer Marian (small and task-aligned). If mirror files are incomplete, fallback to T5.
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    except Exception as e:
        print(f"Primary checkpoint failed: {e}\\nFalling back to t5-small...")
        model_checkpoint = "t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    model.to(device)

    max_input_length = 128
    max_target_length = 128

    def preprocess_function(examples):
        if "t5" in model_checkpoint.lower():
            inputs = [f"translate English to French: {ex['en']}" for ex in examples["translation"]]
        else:
            inputs = [ex["en"] for ex in examples["translation"]]
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
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=1,
        predict_with_generate=True,
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

    save_path = "models/translation_en_fr"
    print(f"Saving model to {save_path}...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Training completed.")


if __name__ == "__main__":
    train()
