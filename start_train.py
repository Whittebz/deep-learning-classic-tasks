import argparse
import subprocess
import os

TASKS = {
    "1": "01_image_classification",
    "2": "02_object_detection",
    "3": "03_semantic_segmentation",
    "4": "04_sentiment_analysis",
    "5": "05_machine_translation",
    "6": "06_named_entity_recognition",
    "7": "07_text_summarization",
    "8": "08_speech_recognition",
    "9": "09_image_generation",
    "10": "10_time_series_forecasting"
}

def run_task(task_id):
    if task_id not in TASKS:
        print(f"Error: Task {task_id} not found.")
        return
    
    task_dir = TASKS[task_id]
    print(f"\n{'='*50}")
    print(f"Starting Training for Task {task_id}: {task_dir}")
    print(f"{'='*50}\n")
    
    script_path = os.path.join(task_dir, "train.py")
    if not os.path.exists(script_path):
        print(f"Error: {script_path} does not exist.")
        return
        
    try:
        # Run the training script in its directory
        subprocess.run(["python", "train.py"], cwd=task_dir, check=True)
        print(f"\nTask {task_id} completed successfully!\n")
    except subprocess.CalledProcessError as e:
        print(f"\nTask {task_id} failed with error code {e.returncode}.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training for deep learning classic tasks.")
    parser.add_argument("--task", type=str, required=True, help="Task ID (1-10) or 'all' to run all tasks sequentially.")
    args = parser.parse_args()
    
    if args.task.lower() == "all":
        for i in range(1, 11):
            run_task(str(i))
    else:
        run_task(args.task)
