from fastapi import FastAPI
from pydantic import BaseModel
from app.triton_client import infer
from app.text_encoder import encode

app = FastAPI()


class Request(BaseModel):
    input_ids: list[int]
    attention_mask: list[int]


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/infer")
def run_inference(req: Request):
    input_ids, attention_mask = encode(req.text)
    result = infer(input_ids, attention_mask)
    return result