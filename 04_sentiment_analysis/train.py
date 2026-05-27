import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)

def train():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    print("Downloading and preparing IMDB dataset...")
    # Load IMDB dataset
    dataset = load_dataset("imdb")
    
    # We will take a very small subset for fast demonstration (1000 train, 200 test)
    small_train_dataset = dataset["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = dataset["test"].shuffle(seed=42).select(range(200))
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def preprocess_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    print("Tokenizing datasets...")
    tokenized_train = small_train_dataset.map(preprocess_function, batched=True)
    tokenized_eval = small_eval_dataset.map(preprocess_function, batched=True)
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    model.to(device)
    
    training_args = TrainingArguments(
        output_dir="./models/results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,     # Just 1 epoch for quick demonstration
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10
    )

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
