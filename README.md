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

## 3. 硬件要求与 AutoDL 镜像选择

AutoDL 服务器运行。每个任务使用独立的 Conda 虚拟环境，实现依赖完全隔离。
**AutoDL 基础镜像：**
- **框架**：`Miniconda`
- **Python环境**：由各任务 `environment.yml`指定
- **系统**：`Ubuntu 22.04`

---

## 4. 环境部署

### 一、创建所有环境

```bash
#创建所有环境
cd deep-learning-classic-tasks
bash setup_envs.sh

#按需创建单个环境
bash setup_envs.sh --task 1
python start_train.py --task 1 --setup
```

### 二、环境管理命令

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
├── 04_sentiment_analysis/
├── 05_machine_translation/
├── 06_named_entity_recognition/
├── 07_text_summarization/
├── 08_speech_recognition/
├── 09_image_generation/
└── 10_time_series_forecasting/
```

> 每个任务子目录内部结构一致，均包含 `environment.yml`、`train.py`、`inference.py`、`app.py`。训练与展示都通过 `conda run` 在对应环境中执行。

---

## 6. 如何运行

### 第一步：训练模型

```bash
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

## 10. 推理效果展示规范（小样本优先）

当前项目更适合采用“小样本先展示推理样例”的方式，不强制要求每个任务都补训练曲线。
这样更省时间，也更符合 README 的展示目的：让读者快速看到模型实际输出。

### 统一目录规范

推荐在仓库根目录下统一使用：

```text
docs/
└── results/
    ├── 01_image_classification/
    ├── 02_object_detection/
    ├── 03_semantic_segmentation/
    ├── 04_sentiment_analysis/
    ├── 05_machine_translation/
    ├── 06_named_entity_recognition/
    ├── 07_text_summarization/
    ├── 08_speech_recognition/
    ├── 09_image_generation/
    └── 10_time_series_forecasting/
```

每个任务目录下只放“推理展示”相关内容，建议命名尽量固定：

```text
docs/results/01_image_classification/
├── sample_01.png
├── sample_02.png
└── README.md
```

说明：
- `sample_01.png`、`sample_02.png`：推理结果图，适合 CV / 生成 / 时序可视化任务
- `README.md`：该任务的简短说明，适合 NLP / Audio 这类更适合表格展示的任务

### 单任务建议内容

小样本阶段，每个任务保留下面一种即可：
- CV / 图像生成 / 时序任务：1 到 3 张推理结果图
- NLP / Audio 任务：1 个结果表格，或 3 到 5 条样例输入输出

### README 推荐写法

```markdown
## 10. 推理效果展示

### 01 图像分类
![01 sample 01](./docs/results/01_image_classification/sample_01.png)
![01 sample 02](./docs/results/01_image_classification/sample_02.png)

说明：展示输入图像、预测类别与置信度。

### 02 目标检测
![02 sample 01](./docs/results/02_object_detection/sample_01.png)

说明：图中应包含预测框、类别和置信度；如有余力，可补一张 GT 对照图。

### 04 情感分析
见：[`docs/results/04_sentiment_analysis/README.md`](./docs/results/04_sentiment_analysis/README.md)

建议表格：

| 输入文本 | 预测结果 | 置信度 |
| :--- | :---: | :---: |
| 这个产品真的很好用 | Positive | 0.98 |
| 体验很差，再也不会买了 | Negative | 0.99 |

### 05 机器翻译
见：[`docs/results/05_machine_translation/README.md`](./docs/results/05_machine_translation/README.md)

| Source | Prediction | Reference |
| :--- | :--- | :--- |
| I love deep learning. | J'aime l'apprentissage profond. | J'aime l'apprentissage profond. |

### 08 语音识别
见：[`docs/results/08_speech_recognition/README.md`](./docs/results/08_speech_recognition/README.md)

| 音频样例 | Prediction | Ground Truth |
| :--- | :--- | :--- |
| sample_01.wav | yes | yes |

### 10 时间序列预测
![10 sample 01](./docs/results/10_time_series_forecasting/sample_01.png)

说明：建议在同一张图中展示历史序列、预测曲线与真实值。
```

### 怎么体现“这是推理结果”

推荐统一遵循下面的展示原则：
- 图像任务：图上直接叠加预测类别、框、mask、分数
- 文本任务：展示 `Input / Prediction / Reference` 三列
- 语音任务：展示音频文件名、预测文本或标签、真实标签
- 时序任务：同图展示历史窗口、预测段、真实未来段
- 图注或说明中明确写明“由 `inference.py` 或 `app.py` 推理得到”

### 行业内更常见的轻量展示方式

如果是 GitHub README，而不是正式论文或报告，最常见的做法其实很克制：
- 首页只放每个任务 1 张代表图，或 1 个代表表
- 详细样例放到 `docs/results/<task_name>/README.md`
- 不追求把所有样本都堆到首页，重点是可读性和可信度

对你这个项目，当前最合适的方式就是：
- README 首页保留每个任务 1 个推理样例入口
- 具体样例统一收进 `docs/results/任务名/`
- 先不做训练曲线占位，后续真有需要再补
