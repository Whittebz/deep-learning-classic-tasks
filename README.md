# 深度学习十大经典任务 (DeepLearning 10 Tasks) - 部署与使用文档

本项目整合了深度学习领域中覆盖计算机视觉(CV)、自然语言处理(NLP)与时间序列分析的10个经典任务。为了便于在各种服务器（特别是配备GPU的服务器）上进行统一管理与展示，我们采用了 **“全局统一环境 + 内部解耦架构”** 的设计。

---

## 1. 任务选型速查表

为了保证能在合理时间内完成 GPU 训练，所有任务都会选用轻量级模型或截断后的小样本经典数据集：

| 任务编号 | 任务名称 | 领域 | 使用模型 | 使用数据集 (Mini版) |
| :---: | :--- | :---: | :--- | :--- |
| **01** | 图像分类 (Image Classification) | CV | ResNet18 | CIFAR-10 (微型图片10分类) |
| **02** | 目标检测 (Object Detection) | CV | Faster R-CNN (MobileNetV3) | PennFudanPed (极小行人检测集) |
| **03** | 语义分割 (Semantic Segmentation) | CV | U-Net / FCN | Oxford-IIIT Pet (裁剪版/单类目) |
| **04** | 文本情感分类 (sentiment_analysis) | NLP | DistilBERT | IMDB (截断取2000条样本极速微调) |
| **05** | 机器翻译 (Machine Translation) | NLP | Seq2Seq (GRU/LSTM) | fra-eng (法译英迷你数据集) |
| **06** | 命名实体识别 (NER) | NLP | DistilBERT (TokenCls) | CoNLL-2003 (极小截断集) |
| **07** | 文本摘要 (Summarization) | NLP | T5-small | CNN/DailyMail (截断小样本演示) |
| **08** | 语音识别 (ASR) | Audio | M5 (纯CNN网络) | SpeechCommands (短语音关键词) |
| **09** | 图像生成 (Image Generation) | CV | DCGAN | MNIST (手写数字生成) |
| **10** | 时间序列预测 (Time Series) | TimeSeries | LSTM | Airline Passengers (航空客运量预测) |

---

## 2. 硬件要求与 AutoDL 镜像选择

本项目推荐在云端 GPU 服务器（如 AutoDL）上运行。由于所有 10 个任务共享**一个大一统的运行环境**，且如果基础环境带有 PyTorch 则无需从头搭建繁杂的 Miniconda，您可以直接租用预装好 PyTorch 的基础镜像，开箱即用！

**🌟 AutoDL 推荐基础镜像配置：**
- **框架**：`PyTorch 2.1.0` (或 2.0.1 以上)
- **Python**：`Python 3.10`
- **CUDA**：`CUDA 12.1` (或 11.8)
- **系统**：`Ubuntu 22.04`

*在 AutoDL 创建实例时，直接在“基础镜像”的 PyTorch 分类下选择符合上述配置的镜像即可。*

---

## 3. 一键环境部署

当您启动 AutoDL 实例并打开终端后，您当前所处的已经是包含了 PyTorch 的底层环境。您只需要补充安装剩余的轻量级依赖即可：

```bash
# 1. 确保在项目根目录
cd deep-learning-classic-tasks

# 2. 安装自然语言处理、语音、时序和UI展示所需的依赖
pip install -r requirements.txt
```

> **说明：** 使用自带 PyTorch 镜像配合 pip 补充依赖，只需要十几秒您的环境就彻底准备完毕，可以直接开始运行所有的训练和展示脚本了。

---

## 4. 项目结构概览

```text
deep-learning-classic-tasks/
├── README.md                    # 本部署文档
├── requirements.txt             # 全局依赖列表
├── start_train.py               # 统一的外部训练脚本
├── start_ui.py                  # 统一的 UI 启动脚手架
│
├── 01_image_classification/     # 任务1：图像分类
│   ├── train.py                 # 核心训练脚本：加载数据、GPU训练、保存权重
│   ├── inference.py             # 独立推理逻辑：加载权重并提供 predict 接口
│   ├── app.py                   # Gradio 极简 UI，内部调用 inference.py
│   └── models/                  # 训练完成后自动生成的权重存放文件夹
│
├── 02_object_detection/         # 任务2：目标检测 (结构同上)
└── ...                          # 其他任务文件夹
```

---

## 5. 如何运行（核心两步走）

为了保证您的显存不会溢出，并且能够直观地看到每个任务的训练 Loss 变化，**建议您逐个任务进行操作。** 操作分为“训练”和“展示”两步。

### 第一步：执行训练生成权重 (`train.py` 或 `start_train.py`)
每个任务下的 `app.py` 都会依赖于训练后保存的模型权重，因此在查看任何 UI 之前，**必须先执行一次该任务的训练脚本**。

**方式一：使用统一脚本（推荐）**
```bash
# 训练指定任务（例如任务1）
python start_train.py --task 1

# 或顺序训练所有任务
python start_train.py --task all
```
> **设计说明：** 为了让您立刻看到效果，所有的训练脚本都默认使用了小样本截断或极少的 epoch。您可以在几分钟内完成一个任务的微调训练。

### 第二步：启动交互式展示界面 (`start_ui.py`)
模型训练完成并保存后，您可以返回到项目**根目录**，使用提供的统一脚本一键启动 Web 界面。

```bash
# 启动特定任务的展示UI，例如启动任务1 (图像分类)
python start_ui.py --task 1
```

执行后，终端会打印出一个本地链接（如：`http://0.0.0.0:7861`）。在浏览器中打开此链接，或通过 AutoDL 提供的“自定义服务”端口映射打开，即可测试您的模型效果。