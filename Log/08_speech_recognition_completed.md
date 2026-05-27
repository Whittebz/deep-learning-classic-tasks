# 任务 8：语音识别 (Speech Recognition) 完成记录

## 1. 任务概述
- **任务编号**：08
- **使用模型**：M5 (纯 CNN 网络架构)
- **数据集**：SpeechCommands (短语音关键词识别，包含如 up, down, left, right 等命令)
- **代码结构**：
  - `train.py`：自动下载 SpeechCommands 数据集，建立长度对齐（Padding）的数据装载器。基于 PyTorch 官方教程搭建了 M5 纯卷积神经网络结构，直接处理原始的一维音频波形。
  - `inference.py`：加载训练好的模型与生成的标签列表。自动将用户上传的音频通过 `torchaudio.transforms.Resample` 降采样至 16000Hz 并转为单声道后进行前向推理，输出 Top-3 结果。
  - `app.py`：利用 Gradio 封装，支持网页端直接录音或上传 `.wav` 文件测试。

## 2. 数据集获取机制
通过 `torchaudio.datasets.SPEECHCOMMANDS('./data', download=True)` 全自动拉取谷歌的 Speech Commands 数据集。选取了 1000 条样本进行飞速演示。

## 3. 完成状态
✅ 已解耦（训练、推理、UI分离）
✅ 基于波形处理的 M5 音频网络和变长对齐技术实现完毕
✅ 网页端录音测试可视化构建完毕
