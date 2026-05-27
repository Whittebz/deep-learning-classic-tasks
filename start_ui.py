import argparse
import subprocess
import os

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
    10: "10_time_series_forecasting"
}

def main():
    parser = argparse.ArgumentParser(description="启动特定深度学习任务的展示UI。")
    parser.add_argument("--task", type=int, choices=range(1, 11), required=True, help="任务编号 (1-10)")
    args = parser.parse_args()

    task_dir = TASK_MAP[args.task]
    app_path = os.path.join(task_dir, "app.py")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    task_full_dir = os.path.join(current_dir, task_dir)
    app_full_path = os.path.join(task_full_dir, "app.py")

    if not os.path.exists(app_full_path):
        print(f"错误: 找不到文件 {app_full_path}。该任务代码可能尚未生成。")
        return

    print(f"正在启动任务 {args.task} ({task_dir}) 的 UI 界面...")
    try:
        # 切换工作目录到子任务目录，确保内部相对路径加载权重文件时不出错
        subprocess.run(["python", "app.py"], cwd=task_full_dir)
    except KeyboardInterrupt:
        print("\nUI 已关闭。")

if __name__ == "__main__":
    main()
