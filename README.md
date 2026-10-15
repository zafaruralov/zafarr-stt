# zafarr-stt

Uzbek speech-to-text service for **stt.zafarr.info**, running on Kubernetes (k3d).

```
browser → stt-api (FastAPI) → Redis queue → stt-worker pods (ONNX model on CPU) → result in Redis
```

- `POST /transcribe` (form field `file`, max 10 MB) → `{id}`; `GET /jobs/{id}` → status and text
- One image, two commands: `uvicorn api:app` (API) and `python worker.py` (worker)
- Workers scale with a HorizontalPodAutoscaler; Kubernetes manifests live in
  [zafarr-infra](https://github.com/zafaruralov/zafarr-infra) under `apps/stt/`

## Model

[nvidia/stt_uz_fastconformer_hybrid_large_pc](https://huggingface.co/nvidia/stt_uz_fastconformer_hybrid_large_pc)
(CC-BY-4.0, by NVIDIA), CTC head exported to ONNX and quantized to int8 with `export.py`
(10.9% WER on FLEURS uz in my Uzbek STT benchmark). `model.int8.onnx` is
attached to the `model-v1` release; CI downloads it into `model/` before building the image.

## Run locally

```bash
gh release download model-v1 -D model
docker build -t zafarr-stt .
docker network create stt
docker run -d --name redis --network stt redis:8-alpine
docker run -d --network stt -p 8000:8000 zafarr-stt
docker run -d --network stt zafarr-stt python worker.py
open http://localhost:8000
```
