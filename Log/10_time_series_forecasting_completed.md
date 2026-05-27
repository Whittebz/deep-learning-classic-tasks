# 任务 10：时间序列预测 (Time Series Forecasting) 完成记录

## 1. 任务概述
- **任务编号**：10
- **使用模型**：LSTM (长短期记忆网络)
- **数据集**：Airline Passengers (经典航空客运量预测)
- **代码结构**：
  - `train.py`：自动从公开数据源下载 `airline-passengers.csv` 文件。利用 `scikit-learn` 中的 `MinMaxScaler` 对序列数据进行归一化，使用滑动窗口（12个月）构造训练序列并训练自定义的 LSTM 预测模型，随后保存模型与缩放器。
  - `inference.py`：加载模型权重与缩放器，接收未来的预测步长，迭代进行序列生成预测。通过 `matplotlib` 将历史真实数据曲线与未来预测曲线合并绘制成红蓝走势图返回。
  - `app.py`：基于 Gradio 构建的交互界面，用户可以自由拖动滑块决定往未来预测几个月。

## 2. 数据集获取机制
在 `download_dataset()` 函数中，使用 `urllib.request` 自动拉取 github 上的公开开源 CSV 数据集至 `data/` 目录，全自动完成。

## 3. 完成状态
✅ 已解耦（训练、推理、UI分离）
✅ 时间序列滑窗处理与归一化反归一化代码完成
✅ 基于 Matplotlib 的预测曲线无缝拼装绘图完成
