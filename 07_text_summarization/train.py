import json
import os
import urllib.request

os.environ['OMP_NUM_THREADS'] = os.environ.get('OMP_NUM_THREADS') or '1'

import torch
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    T5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

MODEL_REPO = 'IDEA-CCNL/Randeng-T5-77M-MultiTask-Chinese'
MODEL_DIR = 'models/t5_zh_summarization'
MODEL_CACHE_DIR = 'models/pretrained_randeng_t5_zh_cache'
DEFAULT_HF_ENDPOINT = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com').rstrip('/')
MODEL_BASE_URLS = [
    f'{DEFAULT_HF_ENDPOINT}/{MODEL_REPO}/resolve/main',
    f'https://huggingface.co/{MODEL_REPO}/resolve/main',
]
MODEL_FILES = [
    'config.json',
    'special_tokens_map.json',
    'tokenizer_config.json',
    'spiece.model',
]
OPTIONAL_MODEL_FILES = [
    'model.safetensors',
    'pytorch_model.bin',
]
DATASET_FILES = [
    f'{DEFAULT_HF_ENDPOINT}/datasets/hugcyp/LCSTS/resolve/main/valid.jsonl',
    'https://huggingface.co/datasets/hugcyp/LCSTS/resolve/main/valid.jsonl',
]
MAX_SAMPLES = 1000
TIMEOUT_SECONDS = 60
MAX_INPUT_LENGTH = 384
MAX_TARGET_LENGTH = 96
SUMMARY_PROMPT_PREFIX = '生成式摘要任务：【'
SUMMARY_PROMPT_SUFFIX = '】这篇文章的摘要是什么？'


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def open_remote_dataset():
    last_exc = None
    headers = {'User-Agent': 'Mozilla/5.0'}
    for url in DATASET_FILES:
        try:
            request = urllib.request.Request(url, headers=headers)
            response = urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS)
            print(f'Dataset stream opened: {url}')
            return url, response
        except Exception as e:
            last_exc = e
            print(f'Failed to open dataset URL {url}: {e}')
    raise RuntimeError(
        'Unable to download the remote Chinese summarization dataset file. '
        'Check network access to Hugging Face or HF mirror, then retry.'
    ) from last_exc


def download_file(url, dest_path):
    headers = {'User-Agent': 'Mozilla/5.0'}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response, open(dest_path, 'wb') as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def ensure_model_files():
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    missing_files = [name for name in MODEL_FILES if not os.path.exists(os.path.join(MODEL_CACHE_DIR, name))]
    existing_weight_files = [name for name in OPTIONAL_MODEL_FILES if os.path.exists(os.path.join(MODEL_CACHE_DIR, name))]

    if not missing_files and existing_weight_files:
        print(f'Using cached model files from {MODEL_CACHE_DIR}')
        return MODEL_CACHE_DIR, 'local-cache'

    last_exc = None
    for base_url in MODEL_BASE_URLS:
        try:
            print(f'Trying direct model download via: {base_url}')
            for name in missing_files:
                dest_path = os.path.join(MODEL_CACHE_DIR, name)
                if os.path.exists(dest_path):
                    continue
                file_url = f'{base_url}/{name}'
                print(f'Downloading {name} ...')
                download_file(file_url, dest_path)

            weight_downloaded = bool(existing_weight_files)
            for name in OPTIONAL_MODEL_FILES:
                dest_path = os.path.join(MODEL_CACHE_DIR, name)
                if os.path.exists(dest_path):
                    weight_downloaded = True
                    continue
                try:
                    file_url = f'{base_url}/{name}'
                    print(f'Downloading optional weight file {name} ...')
                    download_file(file_url, dest_path)
                    weight_downloaded = True
                except Exception as e:
                    last_exc = e
                    print(f'Optional weight file unavailable: {name} ({e})')

            if not weight_downloaded:
                raise RuntimeError('No model weight file could be downloaded.')

            print(f'Model files downloaded via: {base_url}')
            return MODEL_CACHE_DIR, base_url
        except Exception as e:
            last_exc = e
            print(f'Failed direct model download via {base_url}: {e}')
    raise RuntimeError(
        'Unable to download the Chinese summarization model files from either the configured mirror or huggingface.co.'
    ) from last_exc


def load_model_with_fallback(model_source_dir):
    safetensors_path = os.path.join(model_source_dir, 'model.safetensors')
    bin_path = os.path.join(model_source_dir, 'pytorch_model.bin')

    if os.path.exists(safetensors_path):
        try:
            print('Trying to load model from safetensors cache...')
            return AutoModelForSeq2SeqLM.from_pretrained(model_source_dir, use_safetensors=True), 'safetensors'
        except Exception as e:
            print(f'Failed to load safetensors checkpoint: {e}')
            print('Removing corrupted safetensors file and falling back to manual PyTorch checkpoint load...')
            os.remove(safetensors_path)

    if os.path.exists(bin_path):
        print('Loading model from PyTorch checkpoint via manual state_dict load...')
        config = AutoConfig.from_pretrained(model_source_dir)
        model = AutoModelForSeq2SeqLM.from_config(config)
        state_dict = torch.load(bin_path, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f'Missing keys during state_dict load: {missing_keys}')
        if unexpected_keys:
            print(f'Unexpected keys during state_dict load: {unexpected_keys}')
        return model, 'pytorch_model.bin'

    raise FileNotFoundError('No usable model weight file found in cache directory.')


def load_samples(max_samples):
    dataset_url, response = open_remote_dataset()
    samples = []
    with response:
        for raw_line in response:
            if len(samples) >= max_samples:
                break
            line = raw_line.decode('utf-8').strip()
            if not line:
                continue
            item = json.loads(line)
            article = str(item.get('text', '')).strip()
            summary = str(item.get('summary', '')).strip()
            if article and summary:
                samples.append({'article': article, 'summary': summary})
    if not samples:
        raise ValueError('No valid samples were loaded from the remote dataset file.')
    return dataset_url, samples


def build_prompt(article):
    return SUMMARY_PROMPT_PREFIX + article + SUMMARY_PROMPT_SUFFIX


def build_features(tokenizer, samples):
    features = []
    for sample in samples:
        encoded = tokenizer(build_prompt(sample['article']), max_length=MAX_INPUT_LENGTH, truncation=True)
        labels = tokenizer(text_target=sample['summary'], max_length=MAX_TARGET_LENGTH, truncation=True)
        encoded['labels'] = labels['input_ids']
        features.append(encoded)
    return features


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Using at most {MAX_SAMPLES} samples to control disk and training cost.')
    print(f'Primary HF endpoint: {DEFAULT_HF_ENDPOINT}')
    print(f'Model repo: {MODEL_REPO}')

    dataset_url, samples = load_samples(MAX_SAMPLES)
    split_at = max(1, int(len(samples) * 0.9))
    train_samples = samples[:split_at]
    eval_samples = samples[split_at:] if split_at < len(samples) else samples[:1]

    model_source_dir, model_download_source = ensure_model_files()
    tokenizer = T5Tokenizer.from_pretrained(model_source_dir, use_fast=False)
    model, weight_source = load_model_with_fallback(model_source_dir)
    model.to(device)

    train_dataset = SimpleDataset(build_features(tokenizer, train_samples))
    eval_dataset = SimpleDataset(build_features(tokenizer, eval_samples))
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    common_args = dict(
        output_dir='./models/results',
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=1e-4,
        num_train_epochs=2,
        save_total_limit=1,
        logging_steps=20,
        predict_with_generate=True,
        report_to='none',
    )
    try:
        training_args = Seq2SeqTrainingArguments(eval_strategy='epoch', **common_args)
    except TypeError:
        training_args = Seq2SeqTrainingArguments(evaluation_strategy='epoch', **common_args)

    try:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
        )
    except TypeError:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

    trainer.train()

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_pretrained(MODEL_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MODEL_DIR)

    with open(os.path.join(MODEL_DIR, 'training_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(
            {
                'dataset_url': dataset_url,
                'sample_count': len(samples),
                'model_repo': MODEL_REPO,
                'model_download_source': model_download_source,
                'model_cache_dir': MODEL_CACHE_DIR,
                'weight_source': weight_source,
                'prompt_prefix': SUMMARY_PROMPT_PREFIX,
                'prompt_suffix': SUMMARY_PROMPT_SUFFIX,
                'max_input_length': MAX_INPUT_LENGTH,
                'max_target_length': MAX_TARGET_LENGTH,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f'Training completed. Saved model to {MODEL_DIR}')
    print(f'Dataset URL: {dataset_url}')
    print(f'Model source: {model_download_source}')
    print(f'Weight source: {weight_source}')
    print(f'Samples used: {len(samples)}')


if __name__ == '__main__':
    train()
