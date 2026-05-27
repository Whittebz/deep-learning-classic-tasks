# 深度学习十大经典任务 (Deep Learning 10 Classic Tasks)

本项目整合了深度学习领域中覆盖**计算机视觉 (CV)**、**自然语言处理 (NLP)**、**语音处理 (Audio)** 与**时间序列分析 (TimeSeries)** 的 10 个经典任务。

每个任务采用统一的 **"train → inference → app"** 三层解耦架构，训练完成后可独立启动 Gradio Web 界面查看效果。

---

## 1. 任务选型

为了保证能在合理时间内完成 GPU 训练，所有任务都会选用轻量级模型或截断后的小样本经典数据集：

| 编号 | 任务名称 | 领域 | 使用模型 | 数据集 | 训练模式 |
| :---: | :--- | :---: | :--- | :--- | :---: |
| **01** | 图像分类 | CV | ResNet18 | CIFAR-10 (50000张) | 微调 |
| **02** | 目标检测 | CV | Faster R-CNN (MobileNetV3) | PennFudanPed (170张) | 微调 |
| **03** | 语义分割 | CV | FCN-ResNet50 | Oxford-IIIT Pet (取200张) | 微调 |
| **04** | 情感分析 | NLP | DistilBERT | IMDB (取1000条) | 微调 |
| **05** | 机器翻译 | NLP | opus-mt-en-fr (Transformer) | opus_books en-fr (取500条) | 微调 |
| **06** | 命名实体识别 | NLP | DistilBERT (TokenCls) | CoNLL-2003 (取500条) | 微调 |
| **07** | 文本摘要 | NLP | T5-small | CNN/DailyMail (取500条) | 微调 |
| **08** | 语音识别 | Audio | M5 (自定义CNN) | SpeechCommands (取1000条) | 从零训练 |
| **09** | 图像生成 | CV | DCGAN (自定义GAN) | MNIST (取2000张) | 从零训练 |
| **10** | 时间序列预测 | TimeSeries | LSTM (自定义) | Airline Passengers (144条) | 从零训练 |

---

## 2. 训练模式详解

本项目的 10 个任务采用了两种不同的训练方式：

### 基于预训练模型微调 (Fine-tuning) — 任务 01~07

这 7 个任务加载了社区发布的预训练权重（ImageNet / COCO / HuggingFace），然后在一个小数据集上进行**微调 (Fine-tuning)**。

| 编号 | 预训练权重来源 | 微调数据量 | Epochs |
| :---: | :--- | :--- | :---: |
| 01 | `torchvision` ResNet18 (ImageNet) | 全集 50000 张 | 2 |
| 02 | `torchvision` Faster R-CNN MobileNetV3 (COCO) | 120 张 | 2 |
| 03 | `torchvision` FCN-ResNet50 (COCO) | 200 张子集 | 2 |
| 04 | HuggingFace `distilbert-base-uncased` | 1000 条子集 | 1 |
| 05 | HuggingFace `Helsinki-NLP/opus-mt-en-fr` | 450 条子集 | 1 |
| 06 | HuggingFace `distilbert-base-uncased` | 500 条子集 | 2 |
| 07 | HuggingFace `t5-small` | 500 条子集 | 1 |

### 从零开始训练 (From Scratch) — 任务 08、09、10
这 3 个任务使用**自定义网络架构**，没有加载任何预训练权重，是真正的从零训练。
| 编号 | 自定义模型 | 网络类型 | 训练数据量 | Epochs |
| :---: | :--- | :--- | :--- | :---: |
| 08 | M5 | 纯一维 CNN | 1000 条子集 | 2 |
| 09 | DCGAN (Generator + Discriminator) | 对抗生成网络 | 2000 张子集 | 2 |
| 10 | LSTMForecaster | LSTM 循环网络 | 144 条全集 | 150 |

---

## 3. 硬件要求与 AutoDL 镜像选择

AutoDL服务器运行。所有 10 个任务共享一个统一的运行环境。
** AutoDL 推荐配置（基础镜像）：**
- **框架**：`PyTorch 2.5.1`
- **Python**：`Python 3.10`
- **CUDA**：`CUDA 12.1`
- **系统**：`Ubuntu 22.04`

---

## 4. 环境部署

```bash
# 1. 进入项目根目录
cd deep-learning-classic-tasks
# 2. 安装所有依赖
pip install -r requirements.txt
```

---

## 5. 项目结构概览

```text
deep-learning-classic-tasks/
├── README.md                        
├── requirements.txt                 # 全局 pip 依赖列表
├── .gitignore                       
├── start_train.py                   # 统一训练入口脚本
├── start_ui.py                      # 统一 UI 启动脚本
│
├── 01_image_classification/         # 任务1：图像分类 (ResNet18 + CIFAR-10)
│   ├── train.py                     #   训练脚本：数据加载、模型微调、保存权重
│   ├── inference.py                 #   推理逻辑：加载权重、提供 predict 接口
│   ├── app.py                       #   Gradio Web UI，调用 inference.py
│   └── models/                      #   训练产物：权重文件（已被 .gitignore 忽略）
│
├── 02_object_detection/             # 任务2：目标检测 (Faster R-CNN + PennFudanPed)
├── 03_semantic_segmentation/        # 任务3：语义分割 (FCN-ResNet50 + Oxford Pet)
├── 04_sentiment_analysis/           # 任务4：情感分析 (DistilBERT + IMDB)
├── 05_machine_translation/          # 任务5：机器翻译 (opus-mt-en-fr + opus_books)
├── 06_named_entity_recognition/     # 任务6：命名实体识别 (DistilBERT + CoNLL-2003)
├── 07_text_summarization/           # 任务7：文本摘要 (T5-small + CNN/DailyMail)
├── 08_speech_recognition/           # 任务8：语音识别 (M5 CNN + SpeechCommands)
├── 09_image_generation/             # 任务9：图像生成 (DCGAN + MNIST)
├── 10_time_series_forecasting/      # 任务10：时间序列预测 (LSTM + Airline)
│
├── Log/                             # 各任务完成记录
└── Task.md                          # 原始任务需求文档
```

> 每个任务子文件夹 (`01_` ~ `10_`) 内部结构一致，均包含 `train.py`、`inference.py`、`app.py` 三个文件。训练后会自动生成 `models/` 目录存放权重。

---

## 6. 如何运行

### 第一步：训练模型

建议**逐个任务**训练，以避免显存溢出：

```bash
# 训练指定任务（例如任务1）
python start_train.py --task 1

# 或顺序训练全部任务
python start_train.py --task all
```

> **设计说明**：所有训练脚本默认使用小样本截断或极少的 epoch，可在几分钟内完成单个任务的训练。

### 第二步：启动 Web 展示界面

**每个任务训练完成后可立即独立查看效果，不需要等全部任务训练完。**

```bash
# 启动指定任务的展示 UI（例如任务1）
python start_ui.py --task 1
```

执行后，终端会打印本地链接（如 `http://0.0.0.0:7860`），在浏览器中打开即可交互测试模型效果。

如果使用 AutoDL，可通过"自定义服务"端口映射功能访问该链接。

> **提示**：若还未训练该任务，Web 页面会显示 `⚠️ Warning: Model weights not found` 警告，此时需先运行对应的训练脚本。

---

## 7. 训练产物（权重文件）一览

训练完成后，各任务的权重文件保存在各自的 `models/` 子目录下：

| 编号 | 保存路径 | 文件类型 |
| :---: | :--- | :--- |
| 01 | `01_image_classification/models/resnet18_cifar10.pth` | PyTorch state_dict |
| 02 | `02_object_detection/models/faster_rcnn_pennfudan.pth` | PyTorch state_dict |
| 03 | `03_semantic_segmentation/models/fcn_resnet50_pet.pth` | PyTorch state_dict |
| 04 | `04_sentiment_analysis/models/distilbert_imdb/` | HuggingFace 模型目录 |
| 05 | `05_machine_translation/models/translation_en_fr/` | HuggingFace 模型目录 |
| 06 | `06_named_entity_recognition/models/distilbert_ner/` | HuggingFace 模型目录 |
| 07 | `07_text_summarization/models/t5_summarization/` | HuggingFace 模型目录 |
| 08 | `08_speech_recognition/models/m5_speech_commands.pth` | PyTorch state_dict |
| 09 | `09_image_generation/models/dcgan_generator.pth` | PyTorch state_dict |
| 10 | `10_time_series_forecasting/models/lstm_airline.pth` + `scaler.pkl` | PyTorch + Joblib |

> ⚠️ 所有 `models/` 目录和 `data/` 目录均已通过 `.gitignore` 忽略，不会被推送到 Git 仓库。