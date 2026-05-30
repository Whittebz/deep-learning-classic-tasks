#!/usr/bin/env python3
"""
start_ui.py - 深度学习经典任务统一 UI 展示入口

默认模式:
    python start_ui.py
    启动一个监听 6008 端口的控制台，用于切换当前运行中的任务。
    实际任务界面统一监听 6006 端口，且同一时间只允许一个任务运行。

兼容模式:
    python start_ui.py --task 1
    直接启动指定任务的 UI。
"""

import argparse
import importlib.util
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
import shutil

os.environ['NO_PROXY'] = '127.0.0.1,localhost'
os.environ['no_proxy'] = '127.0.0.1,localhost'

HF_MIRROR = "https://hf-mirror.com"
CONTROLLER_PORT = 6008
TASK_PORT = 6006
TASK_URL = f"http://127.0.0.1:{TASK_PORT}"
ROOT_DIR = Path(__file__).resolve().parent

TASK_MAP = {
    1: "01_image_classification",
    2: "02_object_detection",
    3: "03_semantic_segmentation",
    4: "04_sentiment_analysis",
    5: "05_machine_translation",
    6: "06_named_entity_recognition",
    7: "07_text_summarization",
    8: "08_speech_recognition",
    9: "09_image_generation",
    10: "10_time_series_forecasting",
}

TASK_TITLES = {
    1: "01 Image Classification - CIFAR-100",
    2: "02 Object Detection - Faster R-CNN",
    3: "03 Semantic Segmentation - FCN",
    4: "04 文本分类 - 中文情感分析",
    5: "05 Machine Translation - English to French",
    6: "06 中文命名实体识别",
    7: "07 Text Summarization",
    8: "08 Speech Recognition",
    9: "09 Image Generation -Diffusion",
    10: "10 Time Series Forecasting",
}

_CURRENT_PROCESS = None
_CURRENT_TASK_ID = None
_PROCESS_LOCK = threading.Lock()


def get_env_name(task_id):
    return f"dl_task{str(task_id).zfill(2)}"


def conda_env_exists(env_name):
    try:
        result = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

    prefixes = (env_name + " ", env_name + "\t")
    return any(line.strip().startswith(prefixes) for line in result.stdout.splitlines())


def _controller_bootstrap_env():
    candidates = [get_env_name(4)] + [get_env_name(task_id) for task_id in TASK_MAP if task_id != 4]
    for env_name in candidates:
        if conda_env_exists(env_name):
            return env_name
    return None


def ensure_controller_runtime():
    if importlib.util.find_spec("gradio") is not None:
        return

    if os.environ.get("START_UI_BOOTSTRAPPED") == "1":
        raise RuntimeError("当前 Conda 环境中仍未安装 gradio，无法启动 start_ui.py。")

    if shutil.which("conda") is None:
        raise RuntimeError("未检测到 conda，且当前环境没有 gradio，无法自动启动控制台。")

    env_name = _controller_bootstrap_env()
    if env_name is None:
        raise RuntimeError("未找到可用于启动控制台的 dl_taskXX Conda 环境。")

    env = os.environ.copy()
    env["START_UI_BOOTSTRAPPED"] = "1"
    cmd = ["conda", "run", "--no-capture-output", "-n", env_name, "python", str(Path(__file__).resolve()), *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd, env=env))


ensure_controller_runtime()

import gradio as gr


def create_conda_env(task_dir, env_name):
    env_file = ROOT_DIR / task_dir / "environment.yml"
    if not env_file.exists():
        return False, f"环境配置文件不存在: {env_file}"

    try:
        subprocess.run(["conda", "env", "create", "-f", str(env_file)], check=True)
        return True, f"环境 {env_name} 创建成功。"
    except subprocess.CalledProcessError as exc:
        return False, f"环境 {env_name} 创建失败，错误码: {exc.returncode}"


def check_conda_available():
    if shutil.which("conda") is None:
        print("错误: 未检测到 conda 命令。请先安装 Miniconda。")
        sys.exit(1)


def build_runtime_env(task_id=None):
    env = os.environ.copy()
    env.setdefault("HF_ENDPOINT", HF_MIRROR)
    env.setdefault("HF_HOME", "/root/autodl-tmp/hf_home")
    env.setdefault("HUGGINGFACE_HUB_CACHE", "/root/autodl-tmp/hf_home/hub")
    env.setdefault("TRANSFORMERS_CACHE", "/root/autodl-tmp/hf_home/transformers")
    env.setdefault("HF_DATASETS_CACHE", "/root/autodl-tmp/hf_home/datasets")
    env.setdefault("TORCH_HOME", "/root/autodl-tmp/torch_home")
    env.setdefault("TMPDIR", "/root/autodl-tmp/tmp")
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["no_proxy"] = "127.0.0.1,localhost"

    if task_id == 4:
        env.setdefault(
            "SENTIMENT_MODEL_PATH",
            "/root/autodl-tmp/04_sentiment_analysis/models/bert_chinese_sentiment",
        )

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


def task_paths(task_id):
    task_dir = TASK_MAP[task_id]
    task_full_dir = ROOT_DIR / task_dir
    app_path = task_full_dir / "app.py"
    return task_dir, task_full_dir, app_path


def _fetch_running_page_title():
    try:
        with urllib.request.urlopen(TASK_URL, timeout=2) as response:
            html = response.read().decode("utf-8", errors="ignore")
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None

    marker = '"title":"'
    if marker in html:
        start = html.index(marker) + len(marker)
        end = html.find('"', start)
        if end > start:
            return bytes(html[start:end], "utf-8").decode("unicode_escape")

    return None


def _detect_running_task_id_by_http():
    title = _fetch_running_page_title()
    if not title:
        return None

    for task_id, expected in TASK_TITLES.items():
        if title == expected:
            return task_id
    return None


def _detect_local_task_process():
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue

        pid = int(proc_dir.name)
        try:
            cwd = os.readlink(proc_dir / "cwd")
            cmdline_raw = (proc_dir / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue

        cmdline = cmdline_raw.replace(b"\x00", b" ").decode("utf-8", errors="ignore")
        if "python app.py" not in cmdline:
            continue

        for task_id, task_dir in TASK_MAP.items():
            if cwd == str(ROOT_DIR / task_dir):
                return task_id, pid

    return None, None


def _running_state():
    with _PROCESS_LOCK:
        proc = _CURRENT_PROCESS
        task_id = _CURRENT_TASK_ID

    if proc is not None and proc.poll() is None and task_id is not None:
        return {"task_id": task_id, "pid": proc.pid, "source": "controller", "managed": True}

    detected_task_id, detected_pid = _detect_local_task_process()
    if detected_task_id is not None:
        return {"task_id": detected_task_id, "pid": detected_pid, "source": "local_process", "managed": False}

    detected_task_id = _detect_running_task_id_by_http()
    if detected_task_id is not None:
        return {"task_id": detected_task_id, "pid": None, "source": "http_probe", "managed": False}

    return {"task_id": None, "pid": None, "source": None, "managed": False}


def _is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _wait_for_port_state(port, should_be_open, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _is_port_open(port) == should_be_open:
            return True
        time.sleep(0.2)
    return _is_port_open(port) == should_be_open


def _wait_for_task_ready(task_id, timeout=60):
    expected_title = TASK_TITLES[task_id]
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _fetch_running_page_title() == expected_title:
            return True
        time.sleep(0.5)
    return _fetch_running_page_title() == expected_title


def _terminate_process(proc, timeout=10):
    if proc is None or proc.poll() is not None:
        return

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.2)

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _terminate_pid(pid, timeout=10):
    if pid is None:
        return False

    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return False

    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return False

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.2)

    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    return True


def _stop_running_state(state):
    global _CURRENT_PROCESS, _CURRENT_TASK_ID

    if state["managed"]:
        with _PROCESS_LOCK:
            proc = _CURRENT_PROCESS
            tracked_task_id = _CURRENT_TASK_ID
            _CURRENT_PROCESS = None
            _CURRENT_TASK_ID = None
        if proc is not None and proc.poll() is None:
            _terminate_process(proc)
            if not _wait_for_port_state(TASK_PORT, should_be_open=False):
                return False, f"停止 {TASK_MAP.get(tracked_task_id, '当前任务')} 后，端口 {TASK_PORT} 仍未释放。"
            return True, None
        return True, None

    if state["pid"] is not None and state["task_id"] is not None:
        _terminate_pid(state["pid"])
        if not _wait_for_port_state(TASK_PORT, should_be_open=False):
            return False, f"停止 {TASK_MAP[state['task_id']]} 后，端口 {TASK_PORT} 仍未释放。"
        return True, None

    if state["task_id"] is not None:
        return False, f"检测到 {TASK_MAP[state['task_id']]} 正在运行，但未定位到可停止的本地进程。"

    return True, None


def stop_current_task():
    state = _running_state()
    if state["task_id"] is None:
        return "当前没有运行中的任务。"

    ok, error = _stop_running_state(state)
    if not ok:
        return error
    return f"已停止 {TASK_MAP[state['task_id']]}。"


def start_task(task_id, setup=False):
    global _CURRENT_PROCESS, _CURRENT_TASK_ID

    if task_id not in TASK_MAP:
        return _status_text(), f"无效任务编号: {task_id}"

    task_dir, task_full_dir, app_path = task_paths(task_id)
    env_name = get_env_name(task_id)

    if not app_path.exists():
        return _status_text(), f"找不到文件 {app_path}。"

    if not conda_env_exists(env_name):
        if not setup:
            return _status_text(), (
                f"Conda 环境 {env_name} 不存在。"
                f" 请勾选“自动创建缺失环境”后重试。"
            )
        ok, message = create_conda_env(task_dir, env_name)
        if not ok:
            return _status_text(), message

    state = _running_state()
    if state["task_id"] == task_id:
        return _status_text(), f"{task_dir} 已经在 {TASK_PORT} 端口运行。"

    previous_task_id = state["task_id"]
    if previous_task_id is not None:
        ok, error = _stop_running_state(state)
        if not ok:
            return _status_text(), error

    if _is_port_open(TASK_PORT):
        return _status_text(), f"端口 {TASK_PORT} 仍被占用，未启动 {task_dir}。"

    env = build_runtime_env(task_id)

    try:
        proc = subprocess.Popen(
            ["conda", "run", "--no-capture-output", "-n", env_name, "python", "app.py"],
            cwd=str(task_full_dir),
            env=env,
            preexec_fn=os.setsid,
        )
    except Exception as exc:
        return _status_text(), f"启动失败: {exc}"

    if not _wait_for_task_ready(task_id, timeout=60):
        if proc.poll() is None:
            _terminate_process(proc)
        return _status_text(), f"{task_dir} 启动失败，页面未在预期时间内就绪。"

    if proc.poll() is not None:
        return _status_text(), f"{task_dir} 启动失败，进程已退出。"

    with _PROCESS_LOCK:
        _CURRENT_PROCESS = proc
        _CURRENT_TASK_ID = task_id

    status = _status_text()
    if previous_task_id is None:
        message = f"已启动 {task_dir}，请访问 6006 端口对应的公网地址。"
    else:
        message = (
            f"已切换到 {task_dir}，上一个任务 {TASK_MAP[previous_task_id]} 已停止。"
            f" 请访问 6006 端口对应的公网地址。"
        )
    return status, message


def _status_text():
    state = _running_state()
    task_id = state["task_id"]

    if task_id is None:
        return f"当前运行任务：无\n任务服务端口：{TASK_PORT}\n控制台端口：{CONTROLLER_PORT}"

    if state["source"] == "controller":
        source_text = "控制台托管"
    elif state["source"] == "local_process":
        source_text = "检测到本地已运行服务"
    else:
        source_text = "通过 6006 页面探测到服务"

    return (
        f"当前运行任务：{task_id:02d} - {TASK_MAP[task_id]}\n"
        f"任务服务端口：{TASK_PORT}\n"
        f"控制台端口：{CONTROLLER_PORT}\n"
        f"来源：{source_text}"
    )


def stop_task_ui():
    message = stop_current_task()
    return _status_text(), message


def build_controller():
    task_choices = [f"{task_id:02d} - {TASK_MAP[task_id]}" for task_id in TASK_MAP]

    def handle_start(choice, setup):
        task_id = int(choice.split(" - ", 1)[0])
        return start_task(task_id, setup)

    with gr.Blocks(title="深度学习经典任务启动控制台") as demo:
        gr.Markdown("# 深度学习经典任务启动控制台")
        gr.Markdown(
            "此页面固定监听 6008 端口；实际任务页面固定监听 6006 端口。"
            " 每次只能运行一个任务，切换任务时会先停止上一个服务，并等待新任务页面真正就绪后再报告成功。"
        )

        status_box = gr.Textbox(label="当前状态", value=_status_text(), lines=5, interactive=False)
        task_radio = gr.Radio(task_choices, label="选择要运行的任务", value=task_choices[0])
        setup_checkbox = gr.Checkbox(label="自动创建缺失环境", value=False)

        with gr.Row():
            start_button = gr.Button("启动 / 切换任务", variant="primary")
            stop_button = gr.Button("停止当前任务")
            refresh_button = gr.Button("刷新状态")

        message_box = gr.Textbox(label="操作结果", interactive=False)

        start_button.click(handle_start, inputs=[task_radio, setup_checkbox], outputs=[status_box, message_box])
        stop_button.click(stop_task_ui, outputs=[status_box, message_box])
        refresh_button.click(lambda: (_status_text(), "状态已刷新。"), outputs=[status_box, message_box])
        demo.load(lambda: (_status_text(), "控制台已就绪。"), outputs=[status_box, message_box])

    return demo


def run_single_task_mode(task_id, setup=False):
    check_conda_available()

    task_dir, task_full_dir, app_path = task_paths(task_id)
    env_name = get_env_name(task_id)

    if not app_path.exists():
        print(f"错误: 找不到文件 {app_path}。")
        return

    if not conda_env_exists(env_name):
        if setup:
            print(f"环境 {env_name} 不存在，正在自动创建...")
            ok, message = create_conda_env(task_dir, env_name)
            print(message)
            if not ok:
                return
        else:
            print(f"错误: Conda 环境 {env_name} 不存在")
            print(f"可执行: python start_ui.py --task {task_id} --setup")
            return

    print(f"正在启动任务 {task_id} ({task_dir})，端口 {TASK_PORT}")
    env = build_runtime_env(task_id)

    try:
        subprocess.run(
            ["conda", "run", "--no-capture-output", "-n", env_name, "python", "app.py"],
            cwd=str(task_full_dir),
            env=env,
            check=False,
        )
    except KeyboardInterrupt:
        print("\nUI 已关闭。")


def main():
    parser = argparse.ArgumentParser(description="启动深度学习经典任务 UI")
    parser.add_argument("--task", type=int, choices=range(1, 11), help="直接启动指定任务")
    parser.add_argument("--setup", action="store_true", help="环境不存在时自动创建")
    args = parser.parse_args()

    check_conda_available()

    if args.task is not None:
        run_single_task_mode(args.task, args.setup)
        return

    demo = build_controller()
    try:
        demo.launch(server_name="0.0.0.0", server_port=CONTROLLER_PORT, share=False)
    finally:
        stop_current_task()


if __name__ == "__main__":
    main()
