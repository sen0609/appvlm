import io
import json
import os
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from android_env.proto.a11y import android_accessibility_forest_pb2
from tqdm import tqdm

# 确保输出目录存在
os.makedirs("output_images", exist_ok=True)
os.makedirs("output_prompts", exist_ok=True)

# 字体加载（如果有的话）
try:
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font = ImageFont.truetype(font_path, 20)
except IOError:
    font = None

# 判断是否为可交互控件
def is_interactive(node):
    if hasattr(node, "is_clickable") and node.is_clickable:
        return True
    if any(a.id in [1, 16] for a in node.actions):  # click or long_click
        return True
    return False

# 查找包含点击点的 window
def find_target_window(windows, x, y):
    for window in windows:
        bounds = window.bounds_in_screen
        if bounds.left <= x < bounds.right and bounds.top <= y < bounds.bottom:
            return window
    return None

# 获取该 window 中的所有节点
def collect_nodes_from_window(window):
    return list(window.tree.nodes)

# 判断一个点是否在 bounds 中
def is_point_inside_bounds(x, y, bounds):
    return bounds.left <= x < bounds.right and bounds.top <= y < bounds.bottom

# 找出所有包含点击点的节点
def find_nodes_containing_point(nodes, x, y):
    return [node for node in nodes if is_point_inside_bounds(x, y, node.bounds_in_screen)]

# 判断是否具有语义信息
def has_semantic_info(node):
    return bool(
        getattr(node, "view_id_resource_name", "") or
        getattr(node, "content_description", "") or
        getattr(node, "text", "")
    )

# 找出最有语义的子节点
def find_descendant_with_text(node, all_nodes):
    best_node = None
    best_score = -1

    def dfs(n):
        nonlocal best_node, best_score
        score = 0
        if getattr(n, "text", ""):
            score += 3
        if getattr(n, "content_description", ""):
            score += 2
        if getattr(n, "view_id_resource_name", ""):
            score += 1
        if score > best_score:
            best_score = score
            best_node = n
        for cid in getattr(n, "child_ids", []):
            child = next((x for x in all_nodes if x.unique_id == cid), None)
            if child:
                dfs(child)

    dfs(node)
    return best_node if best_score > 0 else None

# 提取节点信息
def extract_node_info(node):
    return {
        "class": node.class_name,
        "text": getattr(node, "text", ""),
        "content description": getattr(node, "content_description", ""),
        "resource name": getattr(node, "view_id_resource_name", ""),
    }

# 从点击坐标构建 click action
def convert_click_to_element_action(forest_bytes, x, y):
    forest = android_accessibility_forest_pb2.AndroidAccessibilityForest()
    forest.ParseFromString(forest_bytes)

    target_window = find_target_window(forest.windows, x, y)
    if target_window is None:
        return None

    all_nodes = collect_nodes_from_window(target_window)
    matched_nodes = find_nodes_containing_point(all_nodes, x, y)

    if not matched_nodes:
        return None

    target_node = max(matched_nodes, key=lambda n: getattr(n, "depth", 0))
    refined_node = find_descendant_with_text(target_node, all_nodes)
    final_node = refined_node if refined_node else target_node

    return {
        "target element": extract_node_info(final_node)
    }

# Step 1: Define TFRecord parsing logic
def _parse_tfrecord_fn(example_proto):
    feature_description = {
        "episode_id": tf.io.FixedLenFeature([], tf.int64),  # 假设 episode_id 是整数
        "screenshots": tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
        "accessibility_trees": tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
        "goal": tf.io.FixedLenFeature([], tf.string),
        "actions": tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    return example

# Step 2: Process all TFRecord shards
tfrecord_files = [
    f"/data/chensen/appvlm/dataset/android_control/android_control-{str(i).zfill(5)}-of-00020"
    for i in range(20)
]

global_step_idx = 0  # 全局步骤索引，用于匹配 splits.json
episode_dict = {}  # 用字典存储 episode 数据，key 是 episode_id
total_steps = 0  # 统计总步骤数

for shard_idx, tfrecord_file in enumerate(tfrecord_files):
    print(f"Processing shard {shard_idx} of {len(tfrecord_files)}: {tfrecord_file}")

    # Load TFRecord shard
    dataset = tf.data.TFRecordDataset(tfrecord_file, compression_type="GZIP")
    dataset = dataset.map(_parse_tfrecord_fn)

    # Convert dataset to list to get total number of episodes for progress bar
    dataset_list = list(dataset)
    shard_episode_count = len(dataset_list)

    # Process each episode in the shard with progress bar
    for example in tqdm(dataset_list, desc=f"Processing shard {shard_idx}/{len(tfrecord_files)-1}", total=shard_episode_count):
        try:
            # Extract fields
            episode_id = example["episode_id"].numpy()
            screenshots = example["screenshots"].numpy()
            trees_raw = example["accessibility_trees"].numpy()
            trees = [android_accessibility_forest_pb2.AndroidAccessibilityForest().FromString(t) for t in trees_raw]
            goal = example["goal"].numpy().decode("utf-8")
            actions = [json.loads(a.decode("utf-8")) for a in example["actions"].numpy()]

            # Initialize episode if not exists
            if episode_id not in episode_dict:
                episode_dict[episode_id] = {
                    "episode_id": int(episode_id),
                    "steps": []
                }

            # Create episode directory
            episode_dir = f"output_images/episode_{episode_id}"
            os.makedirs(episode_dir, exist_ok=True)

            # Process screenshots
            screenshot_paths = []
            for step_idx, screenshot_bytes in enumerate(screenshots):
                image = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")
                draw = ImageDraw.Draw(image)

                # 选中当前激活窗口
                forest = trees[step_idx]
                active_window = next((w for w in forest.windows if w.is_active), forest.windows[-1])

                # 绘制可交互控件
                count = 0
                for node in active_window.tree.nodes:
                    b = node.bounds_in_screen
                    if b.left < b.right and b.top < b.bottom and is_interactive(node):
                        draw.rectangle([(b.left, b.top), (b.right, b.bottom)], outline="red", width=2)

                        label = str(count)
                        if font:
                            text_size = draw.textbbox((0, 0), label, font=font)
                        else:
                            text_size = draw.textbbox((0, 0), label)
                        text_width = text_size[2] - text_size[0]
                        text_height = text_size[3] - text_size[1]

                        bg_rect = [(b.left, b.top), (b.left + text_width + 6, b.top + text_height + 4)]
                        draw.rectangle(bg_rect, fill=(169, 169, 169, 128))

                        draw.text((b.left + 3, b.top + 2), label, fill="white", font=font)
                        count += 1

                # 保存截图到 episode 目录
                screenshot_filename = f"step_{step_idx}.png"
                screenshot_path = os.path.join(episode_dir, screenshot_filename)
                plt.figure(figsize=(6, 12))
                plt.imshow(image)
                plt.axis("off")
                plt.savefig(screenshot_path)
                plt.close()

                screenshot_paths.append(screenshot_path)

            # Process steps and build prompts
            steps = episode_dict[episode_id]["steps"]
            for step_idx in range(len(screenshots)):
                step = {
                    "global_step_idx": global_step_idx,
                    "screenshot": screenshot_paths[step_idx],
                    "Goal": goal,
                    "Previous Actions": [steps[j]["Label"] for j in range(max(0, len(steps) - 5), len(steps))],
                    "Label": {}
                }

                # 获取当前动作
                current_action = actions[step_idx] if step_idx < len(actions) else None

                if current_action:
                    label = {
                        "action_type": current_action["action_type"]
                    }

                    if current_action["action_type"] in ["click", "long_press"] and "x" in current_action and "y" in current_action:
                        x, y = current_action["x"], current_action["y"]
                        click_action = convert_click_to_element_action(trees_raw[step_idx], x, y)
                        if click_action:
                            label["target element"] = click_action["target element"]
                    else:
                        for key, value in current_action.items():
                            if key != "action_type":
                                label[key] = value

                    step["Label"] = label

                steps.append(step)
                global_step_idx += 1
                total_steps += 1

        except Exception as e:
            print(f"Error processing episode {episode_id}: {e}")
            continue

    # 保存当前分片的 episodes 为 JSON 文件
    shard_episodes = list(episode_dict.values())
    with open(f"output_prompts/episodes_{str(shard_idx).zfill(5)}.json", "w") as json_file:
        json.dump(shard_episodes, json_file, indent=4)

    print(f"Saved {len(shard_episodes)} episodes for shard {shard_idx} to 'output_prompts/episodes_{str(shard_idx).zfill(5)}.json'.")

# 统计 episode 数量
episode_count = len(episode_dict)
print(f"Processed all shards. Total episodes: {episode_count}, Total steps: {total_steps}")