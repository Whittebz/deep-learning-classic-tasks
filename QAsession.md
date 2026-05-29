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

