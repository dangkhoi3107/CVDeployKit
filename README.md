# CVDeployKit: Production-Ready CV Microservice (FastAPI + Docker) 🐳🤖

CVDeployKit is a minimal, practical **deployment kit** for serving Computer Vision (CV) models as a **REST API**.
This repository demonstrates how to package a trained **HelmetGuard** model (YOLOv8) into a reproducible **FastAPI** service and run it consistently via **Docker**.

> Why it matters: moving from a notebook to a containerized API is a strong signal of **AI Engineering / MLOps readiness**.

---

## ✅ Key Features

- **FastAPI + Uvicorn** async service with clean endpoints:
  - `GET /health` – service readiness
  - `POST /predict` – image inference
- **YOLOv8 inference** via [Ultralytics](https://github.com/ultralytics/ultralytics)
- **Business rule demo (HelmetGuard logic)**:
  - For each detected person, evaluate **helmet compliance** using a **top-1/3 head-region overlap** heuristic.
- **Dockerized runtime** to eliminate “works on my machine” issues
- **Swagger UI** for interactive testing: `/docs`

---

## 🧰 Tech Stack

- **Python**: 3.11
- **API**: FastAPI, Uvicorn
- **Model**: YOLOv8 (Ultralytics)
- **CV**: OpenCV (headless)
- **Containerization**: Docker

---

## 📁 Project Structure

```text
CVDeployKit/
├── app/
│   ├── main.py          # FastAPI app & endpoints
│   ├── engine.py        # YOLO inference + HelmetGuard business logic
│   └── __init__.py
├── models/
│   └── best.pt          # trained YOLO weights (HelmetGuard)
├── requirements.txt     # Python dependencies
└── Dockerfile           # Docker build instructions
```

> If your Dockerfile is located in `docker/Dockerfile`, see the Docker section below for the correct build command.

---

## ⚙️ Setup (Local)

### 1) Create environment & install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Ensure model weights exist

Place your YOLO weights at:

```text
models/best.pt
```

### 3) Run the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open:
- Health check: `http://localhost:8000/health`
- Swagger UI: `http://localhost:8000/docs`

---

## 🐳 Run with Docker

### Build

**If `Dockerfile` is in the project root:**

```bash
docker build -t cvdeploykit:latest .
```

**If your Dockerfile is in `docker/Dockerfile`:**

```bash
docker build -f docker/Dockerfile -t cvdeploykit:latest .
```

### Run

```bash
docker run --rm -p 8000:8000 cvdeploykit:latest
```

Then test:
- `http://localhost:8000/health`
- `http://localhost:8000/docs`

---

## 🔌 API Endpoints

### `GET /health`

Returns a basic readiness signal (server is up, model import is OK).

Example response:

```json
{
  "status": "online",
  "model": "YOLOv8-HelmetGuard"
}
```

### `POST /predict`

Upload an image and receive detections + helmet compliance status per person.

**Request (curl):**

```bash
curl -F "file=@/path/to/image.jpg" http://localhost:8000/predict
```

**Response (example):**

```json
{
  "filename": "image.jpg",
  "detections": [
    {
      "label": "HELMET_OK",
      "confidence": 0.93,
      "coordinates": { "x1": 120, "y1": 44, "x2": 290, "y2": 410 }
    }
  ],
  "total_persons": 1
}
```

---

## 🧠 HelmetGuard Compliance Logic (Top-1/3 Overlap)

For each detected person bounding box:

1. Compute the **head region** as the **top 1/3** of the person box.
2. If any detected helmet box overlaps that head region (non-zero overlap),
   the person is labeled **`HELMET_OK`**, otherwise **`NO_HELMET`**.

This is a simple, explainable rule that demonstrates product logic on top of model predictions.

---

## 📝 Notes / Assumptions

- Current implementation assumes the model outputs **person** and **helmet** classes.
- If your training class order differs (e.g., helmet is not class `0` and person is not class `1`), update the logic in `app/engine.py`
  to map by **class name** instead of hard-coded IDs.

---

## 🧪 Quick Troubleshooting

- **Model not found**
  - Ensure `models/best.pt` exists inside the container (Docker build must copy `models/`).
- **OpenCV errors in Docker**
  - Use `opencv-python-headless` (already included) to avoid GUI dependencies.
- **Wrong labels**
  - Verify your dataset class names and ordering, then update class mapping logic.

---

## 🚀 Future Improvements (Roadmap)

- GPU runtime (`--gpus all`) with NVIDIA Container Toolkit
- TensorRT export & serving for faster inference on NVIDIA GPUs
- Batch inference endpoint (`/predict_batch`)
- Add CI pipeline (lint, unit tests, container build)
- Structured logging + request timing metrics (prometheus-ready)

---

## 📌 Related Project: HelmetGuard (Training Pipeline)

HelmetGuard is the upstream project that covers the **end-to-end CV training workflow**:
public data sourcing → labeling → YOLOv8 fine-tuning → inference demo + NO_HELMET rule.

CVDeployKit focuses on the **deployment** side: turning trained weights into a **production-style microservice**.

---

