# 任务 7：文本摘要 (Summarization) 完成记录

## 1. 任务概述
- **任务编号**：07
- **使用模型**：T5-small
- **数据集**：CNN/DailyMail (极小截断集：取前 500 条)
- **代码结构**：
  - `train.py`：使用 `datasets.load_dataset("cnn_dailymail", "3.0.0")` 下载新闻文本摘要数据集。为适配 T5 模型特点，在处理输入时增加了 `summarize: ` 的前缀提示词。利用 `Seq2SeqTrainer` 极速微调并保存。
  - `inference.py`：封装了 `Summarizer` 类，设置了诸如 `length_penalty`、`num_beams` 等有助于提升生成质量的解码参数。
  - `app.py`：通过 Gradio 创建了左右分栏的长文本到短文本生成的 UI 界面。

## 2. 数据集获取机制
基于 Hugging Face 数据集接口自动下载版本号为 3.0.0 的 `cnn_dailymail`。通过 `select()` 极大压缩了本地实验时的数据量。

## 3. 完成状态
✅ 已解耦（训练、推理、UI分离）
✅ 自动带有 T5 的前缀预处理和 Beam Search 解码逻辑
✅ 训练代码及 GUI 应用编写完成
