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

    model_checkpoint = "t5-small"
    print("Downloading CNN/DailyMail dataset...")
    candidates = [
        ("cnn_dailymail", "3.0.0"),
        ("abisee/cnn_dailymail", "3.0.0"),
        ("ccdv/cnn_dailymail", "3.0.0"),
    ]
    last_exc = None
    for name, cfg in candidates:
        try:
            raw_datasets = load_dataset(name, cfg)
            print(f"Loaded dataset with identifier: {name}")
            break
        except Exception as e:
            last_exc = e
            print(f"Failed to load with identifier '{name}': {e}")
    else:
        raise last_exc

    # Take a very small subset for fast demonstration
    train_dataset = raw_datasets["train"].shuffle(seed=42).select(range(500))
    eval_dataset = raw_datasets["validation"].shuffle(seed=42).select(range(100))

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    model.to(device)

    max_input_length = 512
    max_target_length = 128

    def preprocess_function(examples):
        inputs = ["summarize: " + doc for doc in examples["article"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
        labels = tokenizer(text_target=examples["highlights"], max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Tokenizing data...")
    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_eval = eval_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    common_args = dict(
        output_dir="./models/results",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
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
            processing_class=tokenizer,
            data_collator=data_collator,
        )
    except TypeError:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

    print("Starting fine-tuning...")
    trainer.train()

    save_path = "models/t5_summarization"
    print(f"Saving model to {save_path}...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Training completed.")


if __name__ == "__main__":
    train()
