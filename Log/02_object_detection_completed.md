# 任务 2：目标检测 (Object Detection) 完成记录

## 1. 任务概述
- **任务编号**：02
- **使用模型**：Faster R-CNN (MobileNetV3 Large FPN Backbone)
- **数据集**：PennFudanPed (行人检测小数据集)
- **代码结构**：
  - `train.py`：使用 `urllib` 和 `zipfile` 自动下载并解压官方数据集。定义了用于解析掩码（PedMasks）和转换为边界框的 `PennFudanDataset`。训练 Faster R-CNN 2个 epoch 并保存权重。
  - `inference.py`：载入保存的模型权重，接收用户图片并输出带有红色边界框和置信度分数的处理后图像。
  - `app.py`：Gradio UI，支持左侧上传图片，右侧显示带有检测框的结果。

## 2. 数据集获取机制
在 `train.py` 内部实现了 `download_and_extract_dataset()` 函数。
它会自动从宾夕法尼亚大学的官网 (`https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip`) 下载数据集，并解压至当前目录，完全不需要用户手动下载干预。

## 3. 完成状态
✅ 已解耦（训练、推理、UI分离）
✅ 数据获取与预处理代码完成
✅ 支持自适应使用 CUDA
