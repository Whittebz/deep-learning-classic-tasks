#!/usr/bin/env python3
"""
start_ui.py - 深度学习经典任务统一 UI 展示入口
自动检测并激活对应任务的 Conda 环境后启动 Gradio 展示界面。

用法:
    python start_ui.py --task 1         # 启动任务 1 的 UI 界面
    python start_ui.py --task 1 --setup # 先创建环境再启动 UI
"""

import argparse
import subprocess
import os
import sys
import shutil

# ===== 国内镜像配置 =====
HF_MIRROR = "https://hf-mirror.com"

TASK_MAP = {
    1:  "01_image_classification",
    2:  "02_object_detection",
    3:  "03_semantic_segmentation",
    4:  "04_sentiment_analysis",
    5:  "05_machine_translation",
    6:  "06_named_entity_recognition",
    7:  "07_text_summarization",
    8:  "08_speech_recognition",
    9:  "09_image_generation",
    10: "10_time_series_forecasting"
}


def get_env_name(task_id):
    """根据任务编号生成 Conda 环境名称"""
    return f"dl_task{str(task_id).zfill(2)}"


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

    print(f"\n正在为 {task_dir} 创建 Conda 环境: {env_name}")
    print("这可能需要几分钟，请耐心等待...\n")

    try:
        subprocess.run(["conda", "env", "create", "-f", env_file], check=True)
        print(f"\n✓ 环境 {env_name} 创建成功！\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 环境 {env_name} 创建失败 (错误码: {e.returncode})")
        return False


def check_conda_available():
    """检查系统中是否安装了 conda"""
    if shutil.which("conda") is None:
        print("错误: 未检测到 conda 命令。请先安装 Miniconda。")
        sys.exit(1)


def build_runtime_env():
    """构建统一运行环境，避免缓存写入根分区。"""
    env = os.environ.copy()

    # 如果外部已显式指定 HF_ENDPOINT，则保留；否则使用镜像。
    env.setdefault("HF_ENDPOINT", HF_MIRROR)
    env.setdefault("HF_HOME", "/root/autodl-tmp/hf_home")
    env.setdefault("HUGGINGFACE_HUB_CACHE", "/root/autodl-tmp/hf_home/hub")
    env.setdefault("TRANSFORMERS_CACHE", "/root/autodl-tmp/hf_home/transformers")
    env.setdefault("HF_DATASETS_CACHE", "/root/autodl-tmp/hf_home/datasets")
    env.setdefault("TORCH_HOME", "/root/autodl-tmp/torch_home")
    env.setdefault("TMPDIR", "/root/autodl-tmp/tmp")

    # 某些环境会默认注入 OMP_NUM_THREADS=0，触发 libgomp warning。
    if env.get("OMP_NUM_THREADS") in (None, "", "0"):
        env["OMP_NUM_THREADS"] = "1"

    for key in [
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
        "TORCH_HOME",
        "TMPDIR",
    ]:
        os.makedirs(env[key], exist_ok=True)

    return env


def main():
    parser = argparse.ArgumentParser(
        description="启动特定深度学习任务的展示 UI（自动 Conda 环境切换）"
    )
    parser.add_argument(
        "--task", type=int, choices=range(1, 11), required=True,
        help="任务编号 (1-10)"
    )
    parser.add_argument(
        "--setup", action="store_true",
        help="如果环境不存在，自动创建"
    )
    args = parser.parse_args()

    check_conda_available()

    task_dir = TASK_MAP[args.task]
    env_name = get_env_name(args.task)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    task_full_dir = os.path.join(current_dir, task_dir)
    app_full_path = os.path.join(task_full_dir, "app.py")

    if not os.path.exists(app_full_path):
        print(f"错误: 找不到文件 {app_full_path}。该任务代码可能尚未生成。")
        return

    if not conda_env_exists(env_name):
        if args.setup:
            print(f"环境 {env_name} 不存在，正在自动创建...")
            if not create_conda_env(task_dir, env_name):
                return
        else:
            print(f"\n错误: Conda 环境 {env_name} 不存在")
            print("\n请先创建环境:")
            print(f"  方法一: bash setup_envs.sh --task {args.task}")
            print(f"  方法二: python start_ui.py --task {args.task} --setup")
            return

    print(f"\n{'='*50}")
    print(f"正在启动任务 {args.task} ({task_dir}) 的 UI 界面")
    print(f"Conda 环境: {env_name}")
    print(f"{'='*50}\n")

    env = build_runtime_env()

    try:
        subprocess.run(
            ["conda", "run", "--no-capture-output", "-n", env_name, "python", "app.py"],
            cwd=task_full_dir,
            env=env,
        )
    except KeyboardInterrupt:
        print("\n\nUI 已关闭。")


if __name__ == "__main__":
    main()
