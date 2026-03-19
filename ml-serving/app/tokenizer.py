from transformers import AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def encode(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=16,  # keep small for now
    )

    return (
        inputs["input_ids"][0].tolist(),
        inputs["attention_mask"][0].tolist(),
    )


