from transformers import AutoTokenizer, AutoModel
import torch

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

model.eval()

# dummy input
inputs = tokenizer("hello world", return_tensors="pt")

torch.onnx.export(
    model,
    (inputs["input_ids"], inputs["attention_mask"]),
    "model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["output"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "output": {0: "batch", 1: "sequence"},
    },
    opset_version=13,
)

print("✅ Exported model.onnx")