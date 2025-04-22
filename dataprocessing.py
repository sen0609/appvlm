from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import io
import os
import tensorflow as tf
import json
from android_env.proto.a11y import android_accessibility_forest_pb2

# 读取数据集
raw_dataset = tf.data.TFRecordDataset("./dataset/android_control/android_control-00011-of-00020", compression_type="GZIP")
dataset_iterator = tf.compat.v1.data.make_one_shot_iterator(raw_dataset)

example = tf.train.Example.FromString(dataset_iterator.get_next().numpy())
features = example.features.feature

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

# 获取截图和树
screenshots = features["screenshots"].bytes_list.value
trees_raw = features["accessibility_trees"].bytes_list.value
trees = [android_accessibility_forest_pb2.AndroidAccessibilityForest().FromString(t) for t in trees_raw]

# 用来存储每个截图和prompt的对应数据
data = []

# 遍历每一帧 observation
for i, (screenshot_bytes, forest) in enumerate(zip(screenshots, trees)):
    image = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)

    # 选中当前激活窗口
    active_window = next((w for w in forest.windows if w.is_active), forest.windows[-1])

    count = 0
    for node in active_window.tree.nodes:
        b = node.bounds_in_screen
        if b.left < b.right and b.top < b.bottom and is_interactive(node):
            # 绘制红色边框
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

    # 保存图像到文件
    screenshot_path = f"output_images/screenshot_{i}.png"
    plt.figure(figsize=(6, 12))
    plt.imshow(image)
    plt.axis("off")
    plt.savefig(screenshot_path)

    # 获取文本输入
    goal = features["goal"].bytes_list.value[0].decode("utf-8")
    actions = [json.loads(a.decode("utf-8")) for a in features["actions"].bytes_list.value]

    # 动态获取最近的前5个 actions
    previous_actions = actions[max(0, i - 5):i]  # 只获取最近的前5个动作

    # 为每个截图添加label，label为当前截图的对应action
    label = actions[i] if i < len(actions) else None

    prompt = {
        "screenshot": screenshot_path,
        "Goal": goal,
        "Previous Actions": previous_actions,  # 仅参考最近5次的actions
        "Label": label  # 当前截图的label
    }

    # 将该数据存储到data中
    data.append(prompt)

# 保存为JSON文件
with open("output_images/prompts.json", "w") as json_file:
    json.dump(data, json_file, indent=4)

print("Data saved to 'output_images/prompts.json'")
