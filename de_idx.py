import json
import os

# 定义要处理的文件列表
files_to_process = [
    "prompts/train.json",
    "prompts/validation.json",
    "prompts/test/test_IDD.json",
    "prompts/test/test_category_unseen.json",
    "prompts/test/test_app_unseen.json",
    "prompts/test/test_task_unseen.json"
]

# 遍历每个文件
for file_path in files_to_process:
    print(f"Processing {file_path}...")

    # Step 1: Load the JSON file
    with open(file_path, "r") as f:
        episodes = json.load(f)

    print(f"Loaded {len(episodes)} episodes from {file_path}")

    # Step 2: Remove "global_step_idx" from each step
    total_steps = 0
    for episode in episodes:
        for step in episode["steps"]:
            if "global_step_idx" in step:
                del step["global_step_idx"]
            total_steps += 1

    print(f"Removed 'global_step_idx' from {total_steps} steps in {file_path}")

    # Step 3: Save the modified JSON file
    with open(file_path, "w") as f:
        json.dump(episodes, f, indent=4)

    print(f"Saved modified data to {file_path}\n")