from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import io
import os
import tensorflow as tf
import json
from android_env.proto.a11y import android_accessibility_forest_pb2

print("当前工作目录:", os.getcwd())

raw_dataset = tf.data.TFRecordDataset("./dataset/android_control/android_control-00005-of-00020",compression_type="GZIP")
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

            # 计算文字尺寸并绘制灰色半透明背景
            label = str(count)
            if font:
                text_size = draw.textbbox((0, 0), label, font=font)
            else:
                text_size = draw.textbbox((0, 0), label)
            text_width = text_size[2] - text_size[0]
            text_height = text_size[3] - text_size[1]

            # 创建灰色半透明背景
            bg_rect = [(b.left, b.top), (b.left + text_width + 6, b.top + text_height + 4)]
            draw.rectangle(bg_rect, fill=(169, 169, 169, 128))  # 灰色半透明背景

            # 绘制白色文字
            draw.text((b.left + 3, b.top + 2), label, fill="white", font=font)
            count += 1

    # 显示
   # 用matplotlib保存图片
    plt.figure(figsize=(6, 12))
    plt.imshow(image)
    plt.axis("off")
    plt.savefig(f"output_images/screenshot_{i}.png")


