# 任务 4：文本情感分类 (Text Classification) 完成记录

## 1. 任务概述
- **任务编号**：04
- **使用模型**：DistilBERT (微调)
- **数据集**：IMDB 影评数据集 (极小样本集截断：训练集 1000 条，测试集 200 条)
- **代码结构**：
  - `train.py`：调用 Hugging Face `datasets` 库自动下载 IMDB 完整数据集并打乱截断。使用 `Trainer` API 进行极速 1 个 epoch 的微调。
  - `inference.py`：封装了 `SentimentClassifier` 类，支持将任意长短的文本转换为 token 并前向计算，输出消极 (Negative) 或积极 (Positive) 情绪的概率分布。
  - `app.py`：使用 Gradio 搭建了直观的交互框，输入一段评论即可显示情感得分。

## 2. 数据集获取机制
通过 Hugging Face 的核心函数 `load_dataset("imdb")` 实现全自动拉取。
为确保在本地快速完成训练，在 `train.py` 中使用了 `select(range(1000))` 对海量数据进行了微型化处理。

## 3. 完成状态
✅ 已解耦（训练、推理、UI分离）
✅ 数据获取与预处理代码完成
✅ 基于 Hugging Face 的 Trainer 极速微调流完成
