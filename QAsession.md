autoDL 然后试用vscode ssh 无法连接
先通过windows cmd 连接建立指纹


wsl antigravity无法通信。
mirror 镜像模式也无法通信。
安装go 语言 然后设置流量转发。(电脑重启后又失败了)待修复
排查：
go version
https://github.com/ccpopy/antissh antigravity IDE 修改工具。
https://linux.do/t/topic/1187318/7


下载任务1 数据集失败。换国内源下载。


试用mamba 命令代替conda 加速 下载依赖库
miniconda 会自动寻找后面的依赖问题。



Collecting package metadata (repodata.json): | WARNING conda.models.version:get_matcher(544): Using .* with relational operator is superfluous and deprecated and will be removed in a future version of conda. Your spec was 1.7.1.*, but conda is ignoring the .* and treating it as 1.7.1
WARNING conda.models.version:get_matcher(544): Using .* with relational operator is superfluous and deprecated and will be removed in a future version of conda. Your spec was 1.8.0.*, but conda is ignoring the .* and treating it as 1.8.0
WARNING conda.models.version:get_matcher(544): Using .* with relational operator is superfluous and deprecated and will be removed in a future version of conda. Your spec was 1.6.0.*, but conda is ignoring the .* and treating it as 1.6.0
WARNING conda.models.version:get_matcher(544): Using .* with relational operator is superfluous and deprecated and will be removed in a future version of conda. Your spec was 1.9.0.*, but conda is ignoring the .* and treating it as 1.9.0 这个是什么问题？
这是一个完全无害的警告信息（WARNING），可以放心地忽略它，不会对你的环境安装或代码运行造成任何影响。

具体原因解释：
这个警告并不是因为我们写的 environment.yml 有错，而是因为 Conda 在服务器（比如 pytorch 或 conda-forge 的官方仓库）上拉取所有可用包的索引目录（即 repodata.json）时，发现上游某些包的依赖声明格式写得不规范。

例如，某个早期的库可能把依赖版本写成了：>=1.7.1.*。 新版本的 Conda 非常严格，它认为：既然用了 >=（大于等于），后面跟着的 .* 就毫无意义了。所以 Conda 弹出了这条警告告诉你：“这种写法过时了，我已经自动把 .* 删掉，当成 1.7.1 来处理了”。

总结：
这不是报错（Error），只是警告（Warning）。
问题出在第三方库仓库的元数据里，我们无法也不需要修改。
Conda 已经非常聪明地自动修正了这个问题（conda is ignoring the .*），所以请安心让它继续运行即可。安装会正常完成的。

# 1. 安装 libmamba 加速引擎（如果在 base 环境）
conda install -n base conda-libmamba-solver

# 2. 将 Conda 的默认求解器设置为 libmamba
conda config --set solver libmamba

# 清理损坏的包缓存
conda clean --packages --tarballs -y

错误	含义
No such file or directory: .../cuda-nvrtc-12.4.127-0.tar.bz2	CUDA 的包下载到一半就断了，文件根本没保存成功
EOFError: Compressed file ended before the end-of-stream marker was reached

# 然后重新创建环境
conda env create -f 10_time_series_forecasting/environment.yml



[2026-05-29] 训练与演示阶段问题记录（新增）

1) start_train/start_ui 在本机环境默认携带 `OMP_NUM_THREADS=0`，导致多任务启动时出现：
`libgomp: Invalid value for environment variable OMP_NUM_THREADS: 0`
处理：在统一入口脚本中将该值兜底为 `1`（仅在为空或为 0 时覆盖）。

2) HuggingFace 镜像端点 `https://hf-mirror.com` 在当前环境存在不稳定/不可达，表现为：
- 模型与数据集下载失败；
- `LocalEntryNotFoundError` / 无法连接镜像。
处理：运行时支持覆盖 `HF_ENDPOINT=https://huggingface.co`，并在入口脚本保留可配置默认值。

3) `datasets` 与 `huggingface_hub` 新版本对无命名空间数据集兼容变化：
- `cnn_dailymail` / `conll2003` 直接名称在部分版本与环境下触发加载异常。
处理：改为可用的仓库标识（如 `abisee/cnn_dailymail`、`tomaarsen/conll2003`），并增加候选回退逻辑。

4) `transformers` 新版本 API 兼容差异：
- `evaluation_strategy` 在新版本可能改为 `eval_strategy`；
- `Trainer/Seq2SeqTrainer` 可能不再接受 `tokenizer=`，改为 `processing_class=`。
处理：训练脚本中增加双分支兼容（先尝试新参数，失败回退旧参数）。

5) 根分区空间不足导致训练中断：
- 典型报错：checkpoint/optimizer 保存失败、写文件失败；
- `task8` 下载 SpeechCommands 默认写到任务目录 `./data`（位于根盘）导致再次挤占根分区。
处理：
- 迁移并软链接任务输出目录到 `/root/autodl-tmp`；
- 迁移并软链接 `08_speech_recognition/data` 到 `/root/autodl-tmp`；
- 在入口脚本统一设置 HF/Transformers/Datasets/Torch/TMP 缓存到 `/root/autodl-tmp`。

6) Conda 迁移经验：
- 直接跨盘复制 `miniconda3/envs` 可能因硬链接失效导致实际占用暴涨，出现 `No space left on device`。
处理：第一阶段采用“新写入重定向”方案：
- `envs_dirs` 首选 `/root/autodl-tmp/conda-envs`；
- `pkgs_dirs` 使用 `/root/autodl-tmp/conda-pkgs`；
- 保留旧环境路径以保证现有任务可继续运行。

7) start_ui 演示入口稳健性增强（本次已完成）：
- 新增与训练入口一致的缓存重定向策略；
- 增加 `OMP_NUM_THREADS` 兜底；
- 保持原有 `--task/--setup` 使用方式不变，避免对现有流程造成破坏性变更。



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


# Miniconda 迁移记录

日期：2026-05-30

## 迁移前检查

- 系统盘 `/`：`27G / 30G`，使用率 `90%`
- 数据盘 `/root/autodl-tmp`：已用约 `11G / 50G`

## 主要占用来源

- `/root/miniconda3`：`20G`
- `/root/deep-learning-classic-tasks`：`2.2G`
- `/root/.vscode-server`：`2.2G`
- `/root/.cache`：`1.5G`

## Conda 环境占用

- `dl_task10`：`7.8G`
- `dl_task09`：`2.2G`
- `dl_task04`：`2.1G`
- `dl_task08`：`1.3G`
- `dl_task05`：`1.2G`
- `dl_task06`：`1.2G`
- `dl_task07`：`1.2G`
- `dl_task01`：`958M`
- `dl_task03`：`859M`
- `dl_task02`：`859M`

## 原始方案

将整个 `/root/miniconda3` 迁移到 `/root/autodl-tmp/miniconda3`，并在原路径创建软链接：

- 目标路径：`/root/autodl-tmp/miniconda3`
- 保持兼容路径：`/root/miniconda3 -> /root/autodl-tmp/miniconda3`

原因：

- 当前 shell 正在使用 `/root/miniconda3/bin/python` 和 `/root/miniconda3/bin/conda`
- 整体迁移并保留原路径软链接，兼容性最好，风险低于重建环境
- `conda` 包缓存几乎不占空间，清缓存收益很小

## 首次执行结果

执行命令：

- `rsync -a /root/miniconda3/ /root/autodl-tmp/miniconda3/`

结果：

- 失败

失败原因：

- 数据盘写满，`/root/autodl-tmp` 达到 `100%`
- 复制后目标目录膨胀到 `40G`，而源目录实际为 `20G`
- 原因是首次复制未保留 Conda 环境中的硬链接，导致大量文件被展开复制

错误特征：

- `rsync` 报错 `No space left on device`

## 已执行清理

- 已删除失败副本：`/root/autodl-tmp/miniconda3`
- 清理后数据盘恢复为约 `11G / 50G`
- 当前数据盘可用空间约 `40G`

## 修正结论

不建议继续直接整套迁移 `/root/miniconda3` 到 `/root/autodl-tmp`，原因如下：

- 数据盘容量虽然名义上足够，但 Conda 环境包含大量硬链接
- 如果复制策略稍有不当，目标体积会显著膨胀
- 当前数据盘只有约 `40G` 可用，容错空间不够

## 后续推荐方案

### 方案 A：优先清理不用的 Conda 环境

先删除不再需要的环境，直接释放系统盘空间，风险最低。

优先关注：

- `dl_task10`：`7.8G`
- `dl_task09`：`2.2G`
- `dl_task04`：`2.1G`

### 方案 B：只迁移单个最大环境

如果必须保留所有环境，可考虑只迁移单个大环境到数据盘，并在原位置做软链接。

优先迁移：

- `/root/miniconda3/envs/dl_task10`

### 方案 C：整套迁移前先进一步腾挪数据盘

如果仍坚持整套迁移，需要先额外释放数据盘空间，并在复制时使用保留硬链接的方式，例如：

- `rsync -aH`

即使如此，也建议先做单环境验证，再做整套迁移。

