# 任务 4：中文情感分析完成记录

## 1. 任务概述
- 任务编号：04
- 任务名称：中文情感分析
- 使用模型：`bert-base-chinese` 微调后的二分类模型
- 数据集：ChnSentiCorp
- 界面文件：`04_sentiment_analysis/app.py`
- 推理入口：`04_sentiment_analysis/inference.py`

## 2. 训练产物与权重路径
- 任务 4 的训练产物实际保存到：`/root/autodl-tmp/04_sentiment_analysis/models/bert_chinese_sentiment`
- 这是一套 HuggingFace 模型目录，不是单个 `.pth` 文件。
- 因为训练产物不在仓库内的 `04_sentiment_analysis/models/`，所以 UI 启动时必须显式指向这个目录，否则会报“未找到模型权重”。

## 3. 已定位并修复的问题
### 问题 1：UI 默认找不到权重
- 现象：`app.py` 启动后提示未找到模型权重。
- 原因：默认推理路径和实际训练输出路径不一致。
- 处理：在 `04_sentiment_analysis/app.py` 中把默认权重路径调整为 `/root/autodl-tmp/04_sentiment_analysis/models/bert_chinese_sentiment`，同时保留 `SENTIMENT_MODEL_PATH` 环境变量覆盖能力。

### 问题 2：AutoDL 环境下 Gradio 启动自检返回 502
- 现象：Gradio 打印 `Running on local URL` 后，又因为访问 `http://localhost:6006/gradio_api/startup-events` 或 `http://localhost:6008/gradio_api/startup-events` 返回 `502` 而退出。
- 原因：实例环境中的代理设置干扰了 Gradio 对本机 `localhost` 的启动自检请求。
- 处理：在 `04_sentiment_analysis/app.py` 和 `start_ui.py` 中补充：
  - `NO_PROXY=127.0.0.1,localhost`
  - `no_proxy=127.0.0.1,localhost`
- 结果：任务 4 页面和统一控制台都可正常启动。

## 4. 统一展示控制台联动说明
- `6008`：统一控制台页面
- `6006`：当前被选中的具体任务页面
- 控制台现在会优先通过本地进程和 `6006` 页面元信息识别当前正在运行的任务。
- 即使任务不是通过控制台启动，只要它确实已经运行在 `6006`，控制台也应能检测并显示当前任务。

## 5. 当前结论
- 任务 4 的模型权重加载问题已解决。
- 任务 4 在 AutoDL 上的 Gradio 启动问题已解决。
- 任务 4 已纳入统一的单任务展示切换机制。
