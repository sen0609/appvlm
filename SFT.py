from transformers import AutoModelForVision2Seq, AutoProcessor

# Load the model
model = AutoModelForVision2Seq.from_pretrained("models/paligemma", local_files_only=True)
processor = AutoProcessor.from_pretrained("models/paligemma", local_files_only=True)

print(processor)

