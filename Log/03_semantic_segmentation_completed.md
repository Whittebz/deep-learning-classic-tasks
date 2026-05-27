# 任务 3：语义分割 (Semantic Segmentation) 完成记录

## 1. 任务概述
- **任务编号**：03
- **使用模型**：FCN (ResNet50 Backbone)
- **数据集**：Oxford-IIIT Pet (猫狗宠物图像语义分割)
- **代码结构**：
  - `train.py`：自动下载并预处理 Oxford-IIIT Pet 分割数据集，构建同步处理图像和遮罩的转换流 (Transforms v2)，训练修改后的 FCN-ResNet50 并保存权重。
  - `inference.py`：加载模型权重，接受用户输入图片，利用模型输出的分割矩阵，将背景、宠物主体和边界进行着色（红色为主体、绿色为边界），并将彩色遮罩透明叠加在原图上。
  - `app.py`：提供 Gradio UI 进行可视化交互展示。

## 2. 数据集获取机制
在 `train.py` 中的自定义数据集 `PetSegmentationDataset` 内使用了 `torchvision.datasets.OxfordIIITPet(download=True)` 机制。
在初次运行训练脚本时，会自动从牛津大学官方源拉取图像（约 800MB）和分割注释，完全不需要手动寻找数据。

## 3. 完成状态
✅ 已解耦（训练、推理、UI分离）
✅ 数据获取与图像/Mask同步变换处理完成
✅ 渲染展示代码（图层混合算法）完成
