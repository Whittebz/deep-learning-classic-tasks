# Task5-9 训练问题与修复记录

## 范围
本文记录任务 `5/6/7/8/9` 在实际训练过程中出现的关键问题、定位结论与已实施修复。

---

## Task 5 机器翻译（05_machine_translation）

### 问题 1：`sentencepiece` 缺失导致 MarianTokenizer 初始化失败
- 现象：`MarianTokenizer requires the SentencePiece library`
- 根因：环境中未安装 `sentencepiece`
- 修复：在 `dl_task05` 中安装 `sentencepiece`

### 问题 2：`Helsinki-NLP/opus-mt-en-fr` 在当前依赖/镜像下分词器加载不稳定
- 现象：`TypeError: expected str, bytes or os.PathLike object, not NoneType`
- 根因：镜像/模型文件不完整或依赖兼容问题导致 Marian tokenizer 资源异常
- 修复：
  1. 训练脚本加入模型回退：`opus-mt-en-fr` 失败时回退到 `t5-small`
  2. T5 路径使用 `translate English to French:` 前缀

### 问题 3：Transformers 参数兼容问题
- 现象：
  - `Seq2SeqTrainer.__init__() got an unexpected keyword argument 'tokenizer'`
  - `evaluation_strategy` 在部分版本不被接受
- 根因：`transformers` API 版本差异
- 修复：
  - `Seq2SeqTrainingArguments` 兼容 `eval_strategy/evaluation_strategy`
  - `Seq2SeqTrainer` 兼容 `processing_class/tokenizer`

### 结果
- 任务 5 训练成功，模型已保存：
  - `05_machine_translation/models/translation_en_fr`（实际可通过软链落盘到数据盘）

---

## Task 6 命名实体识别（06_named_entity_recognition）

### 问题 1：`conll2003` 脚本型数据集不再被当前 `datasets` 支持
- 现象：`RuntimeError: Dataset scripts are no longer supported, but found conll2003.py`
- 根因：`datasets` 新版本禁用旧脚本加载路径
- 修复：切换数据集源为 `tomaarsen/conll2003`（字段兼容原逻辑）

### 问题 2：Transformers 参数兼容问题
- 现象：与 task5 类似，`Trainer`/`TrainingArguments` 参数名不兼容
- 修复：
  - `TrainingArguments` 兼容 `eval_strategy/evaluation_strategy`
  - `Trainer` 兼容 `processing_class/tokenizer`

### 问题 3：训练中 checkpoint 写盘失败
- 现象：`PytorchStreamWriter failed writing file ...`
- 根因：系统盘空间不足
- 修复：
  - 将输出目录与模型目录迁移到 `/root/autodl-tmp`（通过软链保持项目路径不变）
  - 并统一缓存写盘到数据盘

### 结果
- 任务 6 训练成功，模型已保存：
  - `06_named_entity_recognition/models/distilbert_ner`（软链到数据盘）

---

## Task 7 文本摘要（07_text_summarization）

### 问题 1：`cnn_dailymail` 标识在当前 hub 解析失败
- 现象：`Invalid HF URI ... got 'cnn_dailymail'`
- 根因：无命名空间数据集名在当前 `datasets/hf_hub` 组合下解析不稳定
- 修复：训练脚本加入候选源回退，最终可用 `abisee/cnn_dailymail`

### 问题 2：Transformers 参数兼容问题
- 现象：与 task5/6 类似
- 修复：
  - 兼容 `eval_strategy/evaluation_strategy`
  - 兼容 `processing_class/tokenizer`
  - 标签编码使用 `text_target=` 写法

### 结果
- 任务 7 训练成功，模型已保存：
  - `07_text_summarization/models/t5_summarization`

---

## Task 8 语音识别（08_speech_recognition）

### 问题 1：下载目录缺失
- 现象：`FileNotFoundError ... ./data/speech_commands_v0.02.tar.gz...partial`
- 根因：`./data` 目录未提前创建
- 修复：创建 `08_speech_recognition/data`，并将其迁移软链到数据盘

### 问题 2：大文件下载 hash 校验失败
- 现象：`RuntimeError: invalid hash value (expected ..., got ...)`
- 根因：下载中断/损坏导致压缩包校验不一致
- 修复：
  - 清理损坏 `speech_commands_v0.02.tar.gz*` 与 `.partial`
  - 在 `train.py` 中加入一次自动重试：检测 hash 错误后自动删损坏包并重下

### 状态
- 问题已修复到“可重试下载并继续训练”的状态；
- 若网络波动，仍可能需要再次重试下载流程。

---

## Task 9 图像生成（09_image_generation）

### 问题：Gradio 启动失败（本地 startup-events 502）
- 现象：`Couldn't start the app ... localhost:6006/gradio_api/startup-events failed (code 502)`
- 根因：常见于代理环境/端口占用/本地回环不可达策略
- 建议修复方案：
  1. 启动前临时禁用代理（仅该命令生效）
  2. 只使用平台放行端口（如 6006/6008）
  3. 必要时在 app 启动逻辑中加入端口回退

### 训练状态
- task9 模型文件已存在（`dcgan_generator.pth`），问题主要在 UI 启动链路而非训练链路。

---

## 横向共性问题与统一修复

1. **代理影响 HF 访问**
- 现象：HF 数据/模型下载失败，或 localhost 回调异常
- 处理：对训练/UI 启动使用一次性无代理命令，或至少设置 `NO_PROXY=localhost,127.0.0.1`

2. **系统盘空间不足**
- 现象：checkpoint 写盘失败、下载中断
- 处理：
  - 统一把缓存目录重定向到 `/root/autodl-tmp`
  - 各任务 `models/`、大 `data/` 目录迁移并软链

3. **transformers/datasets 版本差异**
- 现象：参数名不兼容、数据集标识解析差异
- 处理：脚本中加入参数与数据源候选回退逻辑，提升跨环境稳定性。

---

## 结论
任务 5/6/7 已完成可复现训练；任务 8 的核心下载完整性问题已加自动修复；任务 9 的关键阻塞点集中在 Gradio 启动环境（代理/端口），训练产物本身可用。
