import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification
)

def train():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    print("Downloading and preparing CoNLL-2003 dataset...")
    # CoNLL-2003 is the classic NER dataset
    dataset = load_dataset("conll2003")
    
    # We will take a very small subset for fast demonstration
    small_train_dataset = dataset["train"].select(range(500))
    small_eval_dataset = dataset["validation"].select(range(100))
    
    model_checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    # NER tags in CoNLL-2003: 0:O, 1:B-PER, 2:I-PER, 3:B-ORG, 4:I-ORG, 5:B-LOC, 6:I-LOC, 7:B-MISC, 8:I-MISC
    label_list = dataset["train"].features[f"ner_tags"].feature.names
    
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples[f"ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    print("Tokenizing datasets...")
    tokenized_train = small_train_dataset.map(tokenize_and_align_labels, batched=True)
    tokenized_eval = small_eval_dataset.map(tokenize_and_align_labels, batched=True)
    
    # Load model
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))
    model.to(device)
    
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    training_args = TrainingArguments(
        output_dir="./models/results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
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
        data_collator=data_collator
    )

    print("Starting training...")
    trainer.train()

    save_path = "models/distilbert_ner"
    print(f"Saving fine-tuned model to {save_path}...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Save label list mapping
    model.config.id2label = {i: label for i, label in enumerate(label_list)}
    model.config.label2id = {label: i for i, label in enumerate(label_list)}
    model.config.save_pretrained(save_path)
    
    print("Training completed.")

if __name__ == "__main__":
    train()
