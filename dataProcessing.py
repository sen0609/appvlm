# save_intermediate_sample_split.py
import tensorflow as tf
import json
from PIL import Image
import io
import os

# 创建保存目录
os.makedirs("intermediate_data", exist_ok=True)

# 加载 TFRecord 文件
dataset = tf.data.TFRecordDataset(
    "./dataset/android_control/android_control-00000-of-00020",
    compression_type="GZIP"
)

# 只取第一条 episode
for raw_record in dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    features = example.features.feature

    goal = features["goal"].bytes_list.value[0].decode("utf-8")
    step_instructions = [s.decode("utf-8") for s in features["step_instructions"].bytes_list.value]
    actions = [json.loads(a.decode("utf-8")) for a in features["actions"].bytes_list.value]
    screenshots = features["screenshots"].bytes_list.value

    for i in range(len(actions)):  # 拆成多步
        image = Image.open(io.BytesIO(screenshots[i])).convert("RGB")
        image_path = f"intermediate_data/sample_{i}.png"
        image.save(image_path)

        json_path = f"intermediate_data/sample_{i}.json"
        with open(json_path, "w") as f:
            json.dump({
                "id": f"sample_{i}",
                "goal": goal,
                "step_instruction": step_instructions[i],
                "action": actions[i],
                "image_path": image_path
            }, f, indent=2)

        print(f"✅ 保存第 {i} 步到 {json_path}")
