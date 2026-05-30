# 利用AI ，写深度学习相关任务代码
利用codex写深度学习相关任务代码
任务目标： 以下十个目标需要全部完成。
1. 先做技术选型。
2. 简单模型且选定经典任务的具体一个应用场景选择一个模型。
3. 实际运用展示设计
4. 一个子文件夹一个项目
5. 给定具体项目的配置，后续方便我购置服务器跑代码执行。
服务器基本配置采用miniconda 来隔离不同任务需要的具体知识库。
注意不需要复杂的项目。能跑通能运行有可靠的结果是第一目标

以下是深度学习中十个最常见的经典任务，涵盖计算机视觉（CV）、自然语言处理（NLP）、语音处理及时间序列领域：。
1. 图像分类 (Image Classification)

  - 定义：识别输入图像所属的类别。
  - 典型应用：人脸识别、猫狗分类（常用模型：ResNet, ViT）。

2. 目标检测 (Object Detection)

  - 定义：定位图像中目标物体的位置（画出边界框）并分类。
  - 典型应用：自动驾驶行人检测、安防监控（常用模型：YOLO, Faster R-CNN）。

3. 语义分割 (Semantic Segmentation)

  - 定义：对图像中的每个像素进行分类，区分出不同的物体区域。
  - 典型应用：医学影像病灶提取、卫星地图路网识别（常用模型：U-Net, DeepLab）。

4. 文本分类/情感分析 (Text Classification / Sentiment Analysis)

  - 定义：对输入文本进行分类或判断其情感倾向（积极/消极）。
  - 典型应用：垃圾邮件过滤、舆情监控（常用模型：BERT, LSTM）。

5. 机器翻译 (Machine Translation)

  - 定义：将一种语言的文本自动翻译成另一种语言。
  - 典型应用：谷歌翻译、同声传译（常用模型：Transformer）。

6. 命名实体识别 (Named Entity Recognition - NER)

  - 定义：从非结构化文本中提取出人名、地名、组织机构名等特定实体。
  - 典型应用：信息抽取、简历解析（常用模型：BiLSTM-CRF, BERT-NER）。

7. 文本生成与摘要 (Text Generation & Summarization)

  - 定义：根据提示词或长文本，生成连贯的新文本或提炼出核心摘要。
  - 典型应用：ChatGPT、新闻简报自动生成（常用模型：GPT系列, T5）。

8. 语音识别 (Automatic Speech Recognition - ASR)

  - 定义：将人类的语音信号（音频）转换为对应的文字。
  - 典型应用：手机语音助手、会议实时转写（常用模型：Whisper, Conformer）。

9. 图像生成 (Image Generation)

  - 定义：根据随机噪声或文本描述生成全新的、高逼真度的图像。
  - 典型应用：AI 绘画（Midjourney）、数据增强（常用模型：Diffusion Models, GAN）。

10. 时间序列预测 (Time Series Forecasting)

  - 定义：基于历史时序数据预测未来的走势。
  - 典型应用：股票价格预测、气象预报、用电量预测（常用模型：Informer, DLinear, LSTM）。