import json
import os

# 确保输出目录存在
os.makedirs("output_prompts", exist_ok=True)

# Step 1: Load test.json
with open("output_prompts/test.json", "r") as f:
    test_episodes = json.load(f)

print(f"Loaded {len(test_episodes)} episodes from test.json")

# Step 2: Load test_subsplits.json
with open("/data/chensen/appvlm/dataset/android_control/test_subsplits.json", "r") as f:
    subsplits = json.load(f)

# Convert subsplits to sets for faster lookup
idd_ids = set(subsplits["IDD"])
category_unseen_ids = set(subsplits["category_unseen"])
app_unseen_ids = set(subsplits["app_unseen"])
task_unseen_ids = set(subsplits["task_unseen"])

print(f"IDD episode IDs: {len(idd_ids)}")
print(f"Category unseen episode IDs: {len(category_unseen_ids)}")
print(f"App unseen episode IDs: {len(app_unseen_ids)}")
print(f"Task unseen episode IDs: {len(task_unseen_ids)}")

# Step 3: Divide test episodes based on episode_id
idd_episodes = []
category_unseen_episodes = []
app_unseen_episodes = []
task_unseen_episodes = []
unspecified_episodes = []

for episode in test_episodes:
    episode_id = episode["episode_id"]
    if episode_id in idd_ids:
        idd_episodes.append(episode)
    elif episode_id in category_unseen_ids:
        category_unseen_episodes.append(episode)
    elif episode_id in app_unseen_ids:
        app_unseen_episodes.append(episode)
    elif episode_id in task_unseen_ids:
        task_unseen_episodes.append(episode)
    else:
        unspecified_episodes.append(episode)

# Step 4: Save to separate JSON files
# IDD
with open("output_prompts/test_IDD.json", "w") as f:
    json.dump(idd_episodes, f, indent=4)
print(f"Saved {len(idd_episodes)} episodes to output_prompts/test_IDD.json")

# Category unseen
with open("output_prompts/test_category_unseen.json", "w") as f:
    json.dump(category_unseen_episodes, f, indent=4)
print(f"Saved {len(category_unseen_episodes)} episodes to output_prompts/test_category_unseen.json")

# App unseen
with open("output_prompts/test_app_unseen.json", "w") as f:
    json.dump(app_unseen_episodes, f, indent=4)
print(f"Saved {len(app_unseen_episodes)} episodes to output_prompts/test_app_unseen.json")

# Task unseen
with open("output_prompts/test_task_unseen.json", "w") as f:
    json.dump(task_unseen_episodes, f, indent=4)
print(f"Saved {len(task_unseen_episodes)} episodes to output_prompts/test_task_unseen.json")

# Unspecified (optional)
if unspecified_episodes:
    with open("output_prompts/test_unspecified.json", "w") as f:
        json.dump(unspecified_episodes, f, indent=4)
    print(f"Saved {len(unspecified_episodes)} unspecified episodes to output_prompts/test_unspecified.json")
else:
    print("No unspecified episodes found.")