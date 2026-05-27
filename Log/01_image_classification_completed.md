# 任务 1：图像分类 (Image Classification) 完成记录

## 1. 任务概述
- **任务编号**：01
- **使用模型**：ResNet18
- **数据集**：CIFAR-10 (微型图片10分类)
- **代码结构**：
  - `train.py`：自动下载并预处理 CIFAR-10 数据集，搭建修改后的 ResNet18 模型并进行 2 个 epoch 的快速训练，保存权重。
  - `inference.py`：使用 `ImageClassifier` 类封装了图像预处理和模型前向推理的逻辑，返回 Top-3 预测结果。
  - `app.py`：基于 Gradio 编写的独立 UI 应用，提供直观的图片上传与预测展示。

## 2. 数据集获取机制
在 `train.py` 中的 `download_and_load_data()` 函数内使用了 `torchvision.datasets.CIFAR10(download=True)` 机制。
在初次运行训练脚本时，若当前目录下不存在数据集，则会自动从官方数据源拉取完整的 CIFAR-10 压缩包并解压，属于**自动获取**。

## 3. 完成状态
✅ 已解耦（训练、推理、UI分离）
✅ 数据获取与预处理代码完成
✅ 支持 CUDA/CPU 自适应

后续可直接进入此目录执行 `python train.py` 测试。
