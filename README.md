# 深度学习十大经典任务 (Deep Learning 10 Classic Tasks)

本项目整合了深度学习领域中覆盖**计算机视觉 (CV)**、**自然语言处理 (NLP)**、**语音处理 (Audio)** 与**时间序列分析 (TimeSeries)** 的 10 个经典任务。

每个任务采用统一的 **train -> inference -> app** 三层解耦架构，训练完成后可独立启动 Gradio Web 界面查看效果。

---

## 1. 任务选型

为了保证能在合理时间内完成 GPU 训练，所有任务都会选用轻量级模型或截断后的小样本经典数据集：

| 编号 | 任务名称 | 领域 | 使用模型 | 数据集 | 训练模式 |
| :---: | :--- | :---: | :--- | :--- | :---: |
| **01** | 图像分类 | CV | ResNet18 | CIFAR-10 (50000张) | 微调 |
| **02** | 目标检测 | CV | Faster R-CNN (MobileNetV3) | PennFudanPed (170张) | 微调 |
| **03** | 语义分割 | CV | FCN-ResNet50 | Oxford-IIIT Pet (取200张) | 微调 |
| **04** | 情感分析 | NLP | BERT-base-chinese | ChnSentiCorp（完整训练/验证/测试集） | 微调 |
| **05** | 机器翻译 | NLP | opus-mt-en-fr (Transformer) | opus_books en-fr (取500条) | 微调 |
| **06** | 命名实体识别 | NLP | DistilBERT (TokenCls) | CoNLL-2003 (取500条) | 微调 |
| **07** | 文本摘要 | NLP | T5-small | CNN/DailyMail (取500条) | 微调 |
| **08** | 语音识别 | Audio | M5 (自定义CNN) | SpeechCommands (取1000条) | 从零训练 |
| **09** | 图像生成 | CV | DCGAN (自定义GAN) | MNIST (取2000张) | 从零训练 |
| **10** | 时间序列预测 | TimeSeries | LSTM (自定义) | Airline Passengers (144条) | 从零训练 |

---

## 2. 训练模式详解

本项目的 10 个任务采用了两种不同的训练方式。

### 基于预训练模型微调 (Fine-tuning) - 任务 01~07

这 7 个任务加载了社区发布的预训练权重（ImageNet / COCO / HuggingFace），然后在一个小数据集上进行微调。

| 编号 | 预训练权重来源 | 微调数据量 | Epochs |
| :---: | :--- | :--- | :---: |
| 01 | `torchvision` ResNet18 (ImageNet) | 全集 50000 张 | 2 |
| 02 | `torchvision` Faster R-CNN MobileNetV3 (COCO) | 120 张 | 2 |
| 03 | `torchvision` FCN-ResNet50 (COCO) | 200 张子集 | 2 |
| 04 | HuggingFace `bert-base-chinese` | ChnSentiCorp 完整训练集 | 1 |
| 05 | HuggingFace `Helsinki-NLP/opus-mt-en-fr` | 450 条子集 | 1 |
| 06 | HuggingFace `distilbert-base-uncased` | 500 条子集 | 2 |
| 07 | HuggingFace `t5-small` | 500 条子集 | 1 |

### 从零开始训练 (From Scratch) - 任务 08、09、10

这 3 个任务使用自定义网络架构，没有加载任何预训练权重。

| 编号 | 自定义模型 | 网络类型 | 训练数据量 | Epochs |
| :---: | :--- | :--- | :--- | :---: |
| 08 | M5 | 纯一维 CNN | 1000 条子集 | 2 |
| 09 | DCGAN (Generator + Discriminator) | 对抗生成网络 | 2000 张子集 | 2 |
| 10 | LSTMForecaster | LSTM 循环网络 | 144 条全集 | 150 |

---

## 3. 硬件要求与 AutoDL 镜像选择

AutoDL 服务器运行。每个任务使用独立的 Conda 虚拟环境，实现依赖完全隔离。

**AutoDL 推荐配置（基础镜像）：**
- **框架**：`Miniconda`
- **Python**：`Python 3.10`（由各任务 `environment.yml` 指定）
- **CUDA**：`CUDA 12.4`
- **系统**：`Ubuntu 22.04`

> 不同任务对 PyTorch、CUDA、以及第三方库版本要求可能相互冲突。使用 Conda 为每个任务创建独立环境，可以稳定隔离依赖。

---

## 4. 环境部署

### 方式一：一键创建所有环境

```bash
cd deep-learning-classic-tasks
bash setup_envs.sh
```

### 方式二：按需创建单个环境

```bash
bash setup_envs.sh --task 1
python start_train.py --task 1 --setup
```

### 环境管理命令

```bash
bash setup_envs.sh --list
bash setup_envs.sh --task 1 --force
bash setup_envs.sh --clean
```

---

## 5. 项目结构概览

```text
deep-learning-classic-tasks/
├── README.md
├── requirements.txt
├── setup_envs.sh
├── start_train.py
├── start_ui.py
├── AutoDL_UI_Usage.md
├── Log/
├── 01_image_classification/
├── 02_object_detection/
├── 03_semantic_segmentation/
├── ...
└── 10_time_series_forecasting/
```

> 每个任务子目录内部结构一致，均包含 `environment.yml`、`train.py`、`inference.py`、`app.py`。训练与展示都通过 `conda run` 在对应环境中执行。

---

## 6. 如何运行

### 第一步：训练模型

```bash
python start_train.py --task 1 --setup
python start_train.py --task 1
python start_train.py --task all --setup
```

### 第二步：启动展示界面

#### 推荐方式：统一控制台

```bash
python start_ui.py
```

说明：
- `start_ui.py` 会自动检查当前环境是否有 `gradio`
- 如果当前环境没有 `gradio`，会自动切换到一个可用的 `dl_taskXX` Conda 环境重新启动自己
- 控制台固定监听 `6008`
- 实际任务页面固定监听 `6006`
- 同一时间只允许一个任务占用 `6006`
- 切换任务时，控制台会先停止上一个任务，并等待 `6006` 端口真正释放，再启动下一个任务

推荐访问顺序：
1. 打开 `6008` 对应的本地地址或 AutoDL 公网映射地址
2. 在控制台里选择要展示的任务
3. 再打开 `6006` 对应地址查看当前任务页面

#### 兼容方式：直接启动单任务

```bash
python start_ui.py --task 4
python start_ui.py --task 10 --setup
```

这会直接启动指定任务，不经过统一控制台。

---

## 7. AutoDL 公网映射与已知问题

### 端口约定
- `6008`：统一控制台
- `6006`：当前实际任务页面

### AutoDL 下的代理问题

在 AutoDL 环境中，Gradio 启动时可能访问：
- `http://localhost:6006/gradio_api/startup-events`
- `http://localhost:6008/gradio_api/startup-events`

如果实例环境中的代理变量影响了 `localhost`，Gradio 可能会返回 `502` 并退出。

当前仓库已经在 `start_ui.py` 和各任务 `app.py` 中内置：

```bash
NO_PROXY=127.0.0.1,localhost
no_proxy=127.0.0.1,localhost
```

用于避免本机回环地址走代理。

### 任务切换时的端口释放问题

任务切换时，旧任务虽然已经收到停止信号，但 `6006` 端口不一定会立刻释放。
如果新任务过早启动，会报：

```text
OSError: Cannot find empty port in range: 6006-6006
```

当前 `start_ui.py` 已修复这一点：会等待 `6006` 真正释放后再启动新任务。

### 任务 04 的特殊点

任务 04 的权重目录位于：

```text
/root/autodl-tmp/04_sentiment_analysis/models/bert_chinese_sentiment
```

因此任务 04 的 `app.py` 额外做了默认权重路径修复。其他任务没有改动模型或数据路径。

更详细的 AutoDL 使用说明见 [AutoDL_UI_Usage.md](./AutoDL_UI_Usage.md)。

---

## 8. Conda 环境一览

| 环境名 | 对应任务 | 核心依赖 |
| :--- | :--- | :--- |
| `dl_task01` | 图像分类 | torch, torchvision, pillow, gradio |
| `dl_task02` | 目标检测 | torch, torchvision, numpy, pillow, gradio |
| `dl_task03` | 语义分割 | torch, torchvision, numpy, pillow, gradio |
| `dl_task04` | 情感分析 | torch, transformers, datasets, gradio |
| `dl_task05` | 机器翻译 | torch, transformers, datasets, sacremoses, gradio |
| `dl_task06` | 命名实体识别 | torch, transformers, datasets, gradio |
| `dl_task07` | 文本摘要 | torch, transformers, datasets, gradio |
| `dl_task08` | 语音识别 | torch, torchaudio, librosa, soundfile, gradio |
| `dl_task09` | 图像生成 | torch, torchvision, matplotlib, gradio |
| `dl_task10` | 时间序列预测 | torch, pandas, scikit-learn, matplotlib, gradio |

---

## 9. 训练产物（权重文件）一览

| 编号 | 保存路径 | 文件类型 |
| :---: | :--- | :--- |
| 01 | `01_image_classification/models/resnet18_cifar10.pth` | PyTorch state_dict |
| 02 | `02_object_detection/models/faster_rcnn_pennfudan.pth` | PyTorch state_dict |
| 03 | `03_semantic_segmentation/models/fcn_resnet50_pet.pth` | PyTorch state_dict |
| 04 | `/root/autodl-tmp/04_sentiment_analysis/models/bert_chinese_sentiment/` | HuggingFace 模型目录 |
| 05 | `05_machine_translation/models/translation_en_fr/` | HuggingFace 模型目录 |
| 06 | `06_named_entity_recognition/models/distilbert_ner/` | HuggingFace 模型目录 |
| 07 | `07_text_summarization/models/t5_summarization/` | HuggingFace 模型目录 |
| 08 | `08_speech_recognition/models/m5_speech_commands.pth` | PyTorch state_dict |
| 09 | `09_image_generation/models/dcgan_generator.pth` | PyTorch state_dict |
| 10 | `10_time_series_forecasting/models/lstm_airline.pth` + `scaler.pkl` | PyTorch + Joblib |

> 所有 `models/` 和 `data/` 目录均已通过 `.gitignore` 忽略，不会被推送到 Git 仓库。
