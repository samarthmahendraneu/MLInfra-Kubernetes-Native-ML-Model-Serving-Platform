from fastapi import FastAPI
from pydantic import BaseModel
from app.triton_client import infer

app = FastAPI()


class Request(BaseModel):
    input_ids: list[int]
    attention_mask: list[int]


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/infer")
def run_inference(req: Request):
    result = infer(req.input_ids, req.attention_mask)
    return result