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
