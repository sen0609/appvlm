import json
import os

# 确保输出目录存在
os.makedirs("output_prompts", exist_ok=True)

# Step 1: Load episodes_00019.json
with open("output_prompts/episodes_00019.json", "r") as f:
    all_episodes = json.load(f)

print(f"Loaded {len(all_episodes)} episodes from episodes_00019.json")

# Step 2: Load splits.json
with open("/data/chensen/appvlm/dataset/android_control/splits.json", "r") as f:
    splits = json.load(f)

# Convert splits to sets for faster lookup
train_ids = set(splits["train"])
val_ids = set(splits["validation"])
test_ids = set(splits["test"])

print(f"Train episode IDs: {len(train_ids)}")
print(f"Validation episode IDs: {len(val_ids)}")
print(f"Test episode IDs: {len(test_ids)}")

# Step 3: Divide episodes based on episode_id
train_episodes = []
val_episodes = []
test_episodes = []
unspecified_episodes = []

for episode in all_episodes:
    episode_id = episode["episode_id"]
    if episode_id in train_ids:
        train_episodes.append(episode)
    elif episode_id in val_ids:
        val_episodes.append(episode)
    elif episode_id in test_ids:
        test_episodes.append(episode)
    else:
        unspecified_episodes.append(episode)

# Step 4: Save to separate JSON files
# Train
with open("output_prompts/train.json", "w") as f:
    json.dump(train_episodes, f, indent=4)
print(f"Saved {len(train_episodes)} episodes to output_prompts/train.json")

# Validation
with open("output_prompts/validation.json", "w") as f:
    json.dump(val_episodes, f, indent=4)
print(f"Saved {len(val_episodes)} episodes to output_prompts/validation.json")

# Test
with open("output_prompts/test.json", "w") as f:
    json.dump(test_episodes, f, indent=4)
print(f"Saved {len(test_episodes)} episodes to output_prompts/test.json")

# Unspecified (optional)
if unspecified_episodes:
    with open("output_prompts/unspecified.json", "w") as f:
        json.dump(unspecified_episodes, f, indent=4)
    print(f"Saved {len(unspecified_episodes)} unspecified episodes to output_prompts/unspecified.json")
else:
    print("No unspecified episodes found.")