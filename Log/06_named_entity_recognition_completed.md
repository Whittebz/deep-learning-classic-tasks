# 任务 6：命名实体识别 (NER) 完成记录

## 1. 任务概述
- **任务编号**：06
- **使用模型**：DistilBERT (Token Classification)
- **数据集**：CoNLL-2003 (极小截断集：取前 500 条)
- **代码结构**：
  - `train.py`：自动拉取并解析经典的 CoNLL-2003 数据集。实现了基于 WordPiece Tokenizer 的复杂词对齐逻辑（Token Alignment），确保 B/I 标签正确对应。使用 Hugging Face 的 Trainer 进行 2 个 epoch 的 NER 微调。
  - `inference.py`：加载模型权重，巧妙调用了 Hugging Face 的 `pipeline("ner", aggregation_strategy="simple")` 实现实体词块合并。
  - `app.py`：使用 Gradio 搭建了直观的实体识别 UI。

## 2. 数据集获取机制
通过 `datasets.load_dataset("conll2003")` 一键自动下载。在 `train.py` 中使用了 `select(range(500))` 大幅压缩数据集以实现秒级或分钟级的微调训练。

## 3. 完成状态
✅ 已解耦（训练、推理、UI分离）
✅ 数据获取与 WordPiece 对齐预处理逻辑代码完成
✅ 自动词汇合并 (Aggregation) 渲染实现完成
