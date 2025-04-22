from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import io
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

# 判断是否为可交互控件(标注图片筛选)
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

def find_descendant_with_text(node, all_nodes):
    """
    找出该节点的所有后代中最“有语义”的那个（含 text / content_description / resource_id），
    优先 text > content_description > view_id_resource_name
    """
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


# 提取结构化信息
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
        #print(f"❌ 没有找到包含点击点 ({x}, {y}) 的窗口")
        return None

    all_nodes = collect_nodes_from_window(target_window)
    matched_nodes = find_nodes_containing_point(all_nodes, x, y)

    #print(f"\n🟡 找到 {len(matched_nodes)} 个匹配点 ({x}, {y}) 的节点")
    for node in matched_nodes:
        b = node.bounds_in_screen
        #print(f"  - ID={node.unique_id} Bounds=({b.left}, {b.top}, {b.right}, {b.bottom}) Class={node.class_name}")

    if not matched_nodes:
        return None

    # 优先选择最深的节点作为目标（比如 unique_id=43）
    target_node = max(matched_nodes, key=lambda n: getattr(n, "depth", 0))

    # 尝试向下找更具语义的子节点（比如 unique_id=71）
    refined_node = find_descendant_with_text(target_node, all_nodes)
    final_node = refined_node if refined_node else target_node

    #print(f"\n🎯 最终选中的节点 ID={final_node.unique_id}, Class={final_node.class_name}")
    return {
        "target element": extract_node_info(final_node)
    }


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

    prompt = {
        "screenshot": screenshot_path,
        "Goal": goal,
        "Previous Actions": [data[j]["Label"] for j in range(max(0, i - 5), i)],  # 使用前5个Label
        "Label": {}  # 初始化为空字典，稍后更新
    }

    # 获取当前动作
    current_action = actions[i] if i < len(actions) else None

    if current_action:
        label = {
            "action_type": current_action["action_type"]
        }

        # 对于 click 和 long_press 动作，将坐标转换为目标 UI 元素
        if current_action["action_type"] in ["click", "long_press"] and "x" in current_action and "y" in current_action:
            x, y = current_action["x"], current_action["y"]
            click_action = convert_click_to_element_action(trees_raw[i], x, y)
            if click_action:
                label["target element"] = click_action["target element"]
        
        # 对于其他操作（如 open_app 等），保存所有相关信息
        else:
            for key, value in current_action.items():
                if key != "action_type":  # action_type 已添加
                    label[key] = value

        prompt["Label"] = label  # 将动作的详细信息添加到 Label 中

    # 将该数据存储到 data 中
    data.append(prompt)

# 保存为 JSON 文件
with open("output_images/prompts.json", "w") as json_file:
    json.dump(data, json_file, indent=4)

print("Data saved to 'output_images/prompts.json'")