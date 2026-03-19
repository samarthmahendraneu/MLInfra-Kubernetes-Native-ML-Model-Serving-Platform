# ML Model Serving Infrastructure (Local)

This project builds a local, production-style ML serving pipeline from scratch — starting with model export and moving toward fully automated deployment, monitoring, and recovery.

---

## Phase 1 — Model Export (ONNX)

### Overview

In this phase, a HuggingFace transformer model (`distilbert-base-uncased`) is converted from PyTorch into ONNX format to make it portable and ready for high-performance inference.

This step separates **model training frameworks** from **serving infrastructure**, which is critical in real-world systems.

---

## Why ONNX?

- Removes dependency on PyTorch at inference time
- Enables execution in optimized runtimes (C++ / Triton)
- Standard format for production ML systems

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

> On Ubuntu, you may need: `sudo apt install python3-venv`

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Export Model to ONNX

Run:

```bash
python export.py
```

This will generate:

```
model.onnx
```

The export:

- Uses dynamic shapes (batch + sequence)
- Targets ONNX opset 18
- Produces a model compatible with Triton Inference Server

---

## Notes

- Warnings during export (e.g., opset conversion or unused weights) are expected and safe to ignore
- The ONNX file is intentionally not committed to Git
- Model export is designed to be reproducible from code

---

## Next Step

Serve the ONNX model using **Triton Inference Server** and expose it via HTTP.