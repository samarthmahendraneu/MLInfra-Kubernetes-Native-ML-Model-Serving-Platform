import requests

TRITON_URL = "http://localhost:8000"


def infer(input_ids, attention_mask):
    url = f"{TRITON_URL}/v2/models/distilbert/infer"

    payload = {
        "inputs": [
            {
                "name": "input_ids",
                "shape": [1, len(input_ids)],
                "datatype": "INT64",
                "data": input_ids,
            },
            {
                "name": "attention_mask",
                "shape": [1, len(attention_mask)],
                "datatype": "INT64",
                "data": attention_mask,
            },
        ]
    }

    response = requests.post(url, json=payload)
    return response.json()