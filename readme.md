# ML Model Serving Infrastructure 

This project builds a local, production-style ML serving pipeline — from model export to a running inference server — simulating how real-world ML systems are deployed and served.

---

## Phase 1 — Model Export + Serving (ONNX + Triton)

### Overview

A HuggingFace transformer model (`distilbert-base-uncased`) is:

1. Exported from PyTorch → ONNX (portable format)
2. Loaded into Triton Inference Server
3. Served via HTTP API

This separates:
- **training stack (PyTorch)** from
- **serving stack (Triton + ONNX Runtime)**

---

## Why ONNX?

- Removes dependency on Python/PyTorch at inference time
- Enables optimized execution (C++ runtime)
- Standard format used in production ML systems

---

## ⚙️ Working Version Matrix (Important)

These versions are **intentionally chosen for compatibility**:

| Component      | Version       | Reason                              |
|----------------|---------------|-------------------------------------|
| PyTorch        | `2.0.1`       | Stable ONNX export (pre-dynamo)     |
| Transformers   | `4.36.0`      | Compatible with PyTorch 2.0         |
| ONNX           | `1.14.0`      | Produces IR version ≤ 9             |
| Triton Server  | `24.03-py3`   | Includes ONNX Runtime backend       |
| ONNX Opset     | `13`          | Compatible with Triton runtime      |

> ⚠️ Newer ONNX / PyTorch versions produce **IR version 10**, which Triton (current runtime) does not support.

---

## Project Structure

```
ml-serving/
├── export.py                  # Converts model → ONNX
├── requirements.txt
├── README.md
├── .gitignore
│
├── model_repository/          # Triton-compatible layout
│   └── distilbert/
│       ├── config.pbtxt       # Model config for Triton
│       └── 1/
│           └── model.onnx     # Generated (ignored in git)
```

---

## Setup

### 1. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

> On Ubuntu: `sudo apt install python3-venv`

### 2. Install dependencies

```bash
pip install torch==2.0.1
pip install transformers==4.36.0
pip install onnx==1.14.0
```

> **Note:** Do NOT install `onnxscript` — it forces the new exporter path and breaks compatibility.

---

## Export Model to ONNX

```bash
python export.py
```

Output:

```
model.onnx
```

Export characteristics:

- Dynamic batching support (batch + sequence)
- Uses legacy ONNX exporter (stable)
- Generates IR version ≤ 9 (Triton compatible)

---

## Serve Model with Triton

```bash
sudo docker run --rm -it \
  -p 8000:8000 \
  -p 8001:8001 \
  -v $(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.03-py3 \
  tritonserver --model-repository=/models
```

---

<img width="3024" height="1748" alt="image" src="https://github.com/user-attachments/assets/28a41493-b077-4235-bf83-89757712ffd5" />

## Verify Server

```bash
curl localhost:8000/v2/health/ready
```

Expected response:

```
OK
```

---

## Run Inference

Prepare `request.json`:

```json
{
  "inputs": [
    {
      "name": "input_ids",
      "shape": [1, 5],
      "datatype": "INT64",
      "data": [101, 7592, 2088, 102, 0]
    },
    {
      "name": "attention_mask",
      "shape": [1, 5],
      "datatype": "INT64",
      "data": [1, 1, 1, 1, 0]
    }
  ]
}
```

Call the API:

```bash
curl -X POST localhost:8000/v2/models/distilbert/infer \
  -H "Content-Type: application/json" \
  -d @request.json
```

---

## Notes & Gotchas

- ONNX IR version mismatch is a common failure mode — `opset_version` ≠ `IR version`
- New PyTorch exporters (dynamo) may break Triton compatibility
- Triton images may or may not include all backends depending on the tag
- CPU-only environments will show CUDA warnings — safe to ignore

---

## Current State (Phase 1)

- ✅ Model exported to ONNX
- ✅ Triton server running
- ✅ Model loaded successfully (`READY`)
- ✅ Inference endpoint working

---

## Phase 2 — API Gateway (FastAPI)

### Overview

A FastAPI service sits in front of Triton, acting as the public-facing service layer. Clients talk to FastAPI — FastAPI talks to Triton.

```
Client
  ↓
FastAPI (your service layer)
  ↓
Triton (inference engine)
  ↓
Model
```

This decouples the inference engine from the client API, enabling validation, auth, routing, and observability to be added without touching the model layer.

---

### Project Structure (Updated)

```
ml-serving/
├── export.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── app/
│   └── main.py                # FastAPI app — proxies requests to Triton
│
├── model_repository/
│   └── distilbert/
│       ├── config.pbtxt
│       └── 1/
│           └── model.onnx
```

---

### Setup

#### 5. Run FastAPI

```bash
uvicorn app.main:app --reload --port 9000
```

<img width="3024" height="364" alt="image" src="https://github.com/user-attachments/assets/ddf04f89-919a-47bf-ae39-497845a6c57e" />


---

### Test the Gateway

#### Health check

```bash
curl localhost:9000/
```

#### Inference

```bash
curl -X POST localhost:9000/infer \
  -H "Content-Type: application/json" \
  -d '{
    "input_ids": [101, 7592, 2088, 102, 0],
    "attention_mask": [1, 1, 1, 1, 0]
  }'
```

---

## Current State (Phase 2)

- ✅ Model exported to ONNX
- ✅ Triton server running
- ✅ Model loaded successfully (`READY`)
- ✅ FastAPI gateway running on port `9000`
- ✅ Health endpoint working
- ✅ Inference proxied through FastAPI → Triton

---

## Phase 3 — Containerized Deployment

### Overview

The full system is containerized and deployed using Docker. Both services run as isolated containers communicating over a shared Docker network.

```
Client → FastAPI (container) → Triton (container) → Model
```

| Service  | Description        |
|----------|--------------------|
| FastAPI  | API Gateway        |
| Triton   | Inference Server   |
| ONNX     | Model format       |

---

### Docker Setup

#### 1. Build API image

```bash
docker build -t ml-api .
```

#### 2. Create network

```bash
docker network create ml-network
```

#### 3. Run Triton

```bash
docker run -d \
  --name triton \
  --network ml-network \
  -p 8000:8000 \
  -p 8001:8001 \
  -v $(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.03-py3 \
  tritonserver --model-repository=/models
```

#### 4. Run API

```bash
docker run -d \
  --name ml-api \
  --network ml-network \
  -p 9000:9000 \
  ml-api
```

---

### Networking

Services communicate via the Docker network. FastAPI reaches Triton at:

```
http://triton:8000
```

No host IP needed — Docker resolves container names as hostnames within the network.

---

### Test

```bash
curl -X POST localhost:9000/infer \
  -H "Content-Type: application/json" \
  -d '{"text": "hello world"}'
```

---

## Key Learnings

- ONNX IR version compatibility is a common failure mode when upgrading dependencies
- `opset_version` ≠ `IR version` — these are different and both matter
- New PyTorch exporters (dynamo) may break Triton compatibility
- Container networking: services communicate by name, not by IP
- API abstraction decouples the client contract from the inference engine

---

## Current State (Phase 3)

- ✅ Model exported to ONNX
- ✅ Triton serving working
- ✅ FastAPI gateway implemented
- ✅ Text → embedding pipeline working
- ✅ Dockerized multi-service setup

---

## Next Steps

- Deploy on k3s cluster (multi-node setup)
- Add CI/CD (model build → deploy → rollout)
- Add monitoring (Prometheus + Grafana)