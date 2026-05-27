# 任务 5：机器翻译 (Machine Translation) 完成记录

## 1. 任务概述
- **任务编号**：05
- **使用模型**：Seq2Seq Transformer (`Helsinki-NLP/opus-mt-en-fr`)
- **数据集**：opus_books (英法双语语料)
- **代码结构**：
  - `train.py`：自动下载并预处理 opus_books 数据集，取 500 条样本构建微调环境。利用 Hugging Face 的 `Seq2SeqTrainer` 演示翻译模型的端到端微调流程并保存模型。
  - `inference.py`：封装了 `Translator` 类，实现了带有 Beam Search 的高效 Seq2Seq 文本生成。
  - `app.py`：利用 Gradio 构建了左右分栏的在线机器翻译 UI。

## 2. 数据集获取机制
通过 Hugging Face `load_dataset("opus_books", "en-fr")` 自动拉取，完全免配置免手动下载。

## 3. 完成状态
✅ 已解耦（训练、推理、UI分离）
✅ 数据获取与预处理代码完成
✅ 基于 Hugging Face 的 Seq2Seq 微调流完成
