#!/usr/bin/env python3
"""
start_train.py - 深度学习经典任务统一训练入口
自动检测并激活对应任务的 Conda 环境后执行训练脚本。

用法:
    python start_train.py --task 1       # 训练任务 1（图像分类）
    python start_train.py --task all     # 顺序训练所有 10 个任务
    python start_train.py --task 1 --setup  # 先创建环境再训练

现代 AI 服务的环境隔离策略说明:
    在生产环境中，主流的做法是使用 Docker 容器来隔离不同模型服务。
    每个模型对应一个独立的 Docker 镜像，其中包含了所有依赖。
    Kubernetes 用来编排和调度这些容器。

    本项目出于教学目的，使用 Conda 虚拟环境来实现类似的隔离效果：
    - 每个任务有独立的 environment.yml 定义其精确依赖
    - start_train.py 使用 `conda run` 在对应环境中运行训练
    - 这在概念上等同于 Docker：每个任务 = 一个独立的运行环境
"""

import argparse
import subprocess
import os
import sys
import shutil

# ===== 国内镜像配置 =====
# HuggingFace 国内镜像（影响任务 04-07 的模型和数据集下载）
# 如果你的网络可以直连 huggingface.co，可以注释掉下面这行
HF_MIRROR = "https://hf-mirror.com"

TASKS = {
    "1":  "01_image_classification",
    "2":  "02_object_detection",
    "3":  "03_semantic_segmentation",
    "4":  "04_sentiment_analysis",
    "5":  "05_machine_translation",
    "6":  "06_named_entity_recognition",
    "7":  "07_text_summarization",
    "8":  "08_speech_recognition",
    "9":  "09_image_generation",
    "10": "10_time_series_forecasting"
}

def get_env_name(task_id):
    """根据任务编号生成 Conda 环境名称"""
    return f"dl_task{task_id.zfill(2)}"

def conda_env_exists(env_name):
    """检查 Conda 环境是否已存在"""
    try:
        result = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True, text=True, check=True
        )
        for line in result.stdout.splitlines():
            if line.strip().startswith(env_name + " ") or line.strip().startswith(env_name + "\t"):
                return True
        return False
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def create_conda_env(task_dir, env_name):
    """从 environment.yml 创建 Conda 环境"""
    env_file = os.path.join(task_dir, "environment.yml")
    if not os.path.exists(env_file):
        print(f"错误: 环境配置文件不存在: {env_file}")
        return False

    print(f"\n{'='*50}")
    print(f"正在为 {task_dir} 创建 Conda 环境: {env_name}")
    print(f"这可能需要几分钟，请耐心等待...")
    print(f"{'='*50}\n")

    try:
        subprocess.run(
            ["conda", "env", "create", "-f", env_file],
            check=True
        )
        print(f"\n✓ 环境 {env_name} 创建成功！\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 环境 {env_name} 创建失败 (错误码: {e.returncode})")
        print("请尝试手动运行: conda env create -f", env_file)
        return False

def check_conda_available():
    """检查系统中是否安装了 conda"""
    if shutil.which("conda") is None:
        print("="*50)
        print("错误: 未检测到 conda 命令")
        print("="*50)
        print()
        print("请先安装 Miniconda:")
        print("  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh")
        print("  bash Miniconda3-latest-Linux-x86_64.sh")
        print()
        print("安装完成后，重新打开终端再运行本脚本。")
        sys.exit(1)

def run_task(task_id, auto_setup=False):
    """在对应的 Conda 环境中运行训练脚本"""
    if task_id not in TASKS:
        print(f"错误: 任务 {task_id} 不存在。有效范围: 1-10")
        return False

    task_dir = TASKS[task_id]
    env_name = get_env_name(task_id)
    script_path = os.path.join(task_dir, "train.py")

    if not os.path.exists(script_path):
        print(f"错误: 训练脚本不存在: {script_path}")
        return False

    # 检查环境是否存在
    if not conda_env_exists(env_name):
        if auto_setup:
            print(f"环境 {env_name} 不存在，正在自动创建...")
            if not create_conda_env(task_dir, env_name):
                return False
        else:
            print(f"\n{'='*50}")
            print(f"错误: Conda 环境 {env_name} 不存在")
            print(f"{'='*50}")
            print(f"\n请先创建环境:")
            print(f"  方法一: bash setup_envs.sh --task {task_id}")
            print(f"  方法二: python start_train.py --task {task_id} --setup")
            print(f"  方法三: conda env create -f {task_dir}/environment.yml")
            return False

    print(f"\n{'='*50}")
    print(f"任务 {task_id}: {task_dir}")
    print(f"Conda 环境: {env_name}")
    print(f"{'='*50}\n")

    # 构建环境变量（传递 HF 镜像配置给子进程）
    env = os.environ.copy()
    env.setdefault("HF_ENDPOINT", HF_MIRROR)
    env.setdefault("HF_HOME", "/root/autodl-tmp/hf_home")
    env.setdefault("HUGGINGFACE_HUB_CACHE", "/root/autodl-tmp/hf_home/hub")
    env.setdefault("TRANSFORMERS_CACHE", "/root/autodl-tmp/hf_home/transformers")
    env.setdefault("HF_DATASETS_CACHE", "/root/autodl-tmp/hf_home/datasets")
    env.setdefault("TORCH_HOME", "/root/autodl-tmp/torch_home")
    env.setdefault("TMPDIR", "/root/autodl-tmp/tmp")
    if env.get("OMP_NUM_THREADS") in (None, "", "0"):
        env["OMP_NUM_THREADS"] = "1"

    os.makedirs(env["HF_HOME"], exist_ok=True)
    os.makedirs(env["HUGGINGFACE_HUB_CACHE"], exist_ok=True)
    os.makedirs(env["TRANSFORMERS_CACHE"], exist_ok=True)
    os.makedirs(env["HF_DATASETS_CACHE"], exist_ok=True)
    os.makedirs(env["TORCH_HOME"], exist_ok=True)
    os.makedirs(env["TMPDIR"], exist_ok=True)

    try:
        # 使用 conda run 在目标环境中执行训练脚本
        # --no-capture-output: 实时显示训练输出（不缓冲）
        subprocess.run(
            ["conda", "run", "--no-capture-output", "-n", env_name, "python", "train.py"],
            cwd=task_dir,
            check=True,
            env=env
        )
        print(f"\n✓ 任务 {task_id} 训练完成！\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 任务 {task_id} 训练失败 (错误码: {e.returncode})\n")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="深度学习经典任务统一训练入口（自动 Conda 环境切换）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python start_train.py --task 1          训练任务 1（图像分类）
  python start_train.py --task 8 --setup  自动创建环境并训练任务 8
  python start_train.py --task all        顺序训练所有任务
  python start_train.py --task all --setup 创建所有环境并顺序训练
        """
    )
    parser.add_argument(
        "--task", type=str, required=True,
        help="任务编号 (1-10) 或 'all' 顺序运行所有任务"
    )
    parser.add_argument(
        "--setup", action="store_true",
        help="如果环境不存在，自动创建（等同于先运行 setup_envs.sh）"
    )
    args = parser.parse_args()

    # 检查 conda 可用性
    check_conda_available()

    if args.task.lower() == "all":
        print("\n" + "="*50)
        print("  开始顺序训练所有 10 个任务")
        print("  每个任务使用独立的 Conda 环境")
        print("="*50)
        
        results = {}
        for i in range(1, 11):
            success = run_task(str(i), auto_setup=args.setup)
            results[i] = "✓ 成功" if success else "✗ 失败"
        
        # 打印汇总报告
        print("\n" + "="*50)
        print("  训练汇总报告")
        print("="*50)
        for task_id, status in results.items():
            task_dir = TASKS[str(task_id)]
            print(f"  任务 {task_id:2d} ({task_dir}): {status}")
        print("="*50 + "\n")
    else:
        run_task(args.task, auto_setup=args.setup)
