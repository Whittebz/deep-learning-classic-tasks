# AutoDL 展示与公网映射说明

## 1. 端口约定
本项目当前采用固定双端口方案：

- `6008`：统一展示控制台
- `6006`：实际任务页面

控制台负责切换任务。任意时刻只允许一个任务占用 `6006`。

## 2. AutoDL 公网映射下的推荐使用方式
在 AutoDL 实例中，为 `6006` 和 `6008` 配置公网映射后：

1. 先访问 `6008` 对应的公网地址
2. 在控制台中选择要展示的任务
3. 控制台会先停止上一个任务，再启动新任务
4. 再访问 `6006` 对应的公网地址查看当前任务页面

这意味着：
- `6008` 是“切换器”
- `6006` 是“实际展示页”

## 3. 启动命令
### 启动控制台
```bash
conda run -n dl_task04 python start_ui.py
```

说明：
- `start_ui.py` 依赖 `gradio`
- 当前可直接复用 `dl_task04` 环境启动控制台

### 兼容旧方式：直接启动单个任务
```bash
python start_ui.py --task 4
```

这会直接把任务 4 启动在 `6006`，不经过控制台。

## 4. AutoDL 上遇到的实际问题
### 问题：Gradio 启动后立即因为 localhost 自检失败退出
典型报错形式：
- `Couldn't start the app because 'http://localhost:6006/gradio_api/startup-events' failed (code 502)`
- `Couldn't start the app because 'http://localhost:6008/gradio_api/startup-events' failed (code 502)`

### 原因
AutoDL 实例内常见的代理环境变量会影响本机回环地址访问，导致 Gradio 在启动阶段访问 `localhost` 的自检请求被代理拦截，最终返回 `502`。

### 处理方式
在启动控制台和任务 UI 前，确保本机回环地址不走代理：

```bash
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
```

当前仓库已在以下文件中内置这一处理：
- `start_ui.py`
- `04_sentiment_analysis/app.py`

## 5. 控制台当前行为
- 默认执行 `python start_ui.py` 时，启动 `6008` 控制台。
- 控制台会检测当前 `6006` 上已经在运行的任务。
- 如果切换任务，会先停掉旧任务，再启动新任务。
- 如果任务 4 已经手工启动在 `6006`，控制台刷新状态后应显示它为当前任务。

## 6. 当前已验证结果
- `http://127.0.0.1:6006` 返回 `200 OK`
- `http://127.0.0.1:6008` 返回 `200 OK`
- 任务 4 权重目录已确认为：`/root/autodl-tmp/04_sentiment_analysis/models/bert_chinese_sentiment`


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