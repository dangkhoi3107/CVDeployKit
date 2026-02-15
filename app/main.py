from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

app = FastAPI(title="CVDeployKit")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    _img = Image.open(io.BytesIO(content)).convert("RGB")
    return {
        "filename": file.filename,
        "status": "ok",
        "note": "WIP: model inference will be added later"
    }
