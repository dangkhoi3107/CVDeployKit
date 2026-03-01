from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
from app.engine import process_detection

# Gateway
# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="CVDeployKit - Helmet Guard API",
    description="API for detecting safety helmets using YOLOv8",
    version="1.0.0"
)

@app.get("/health")
async def health_check():
    """Check if the server and model are ready."""
    return {
        "status": "online",
        "model": "YOLOv8-HelmetGuard",
        "device": "NVIDIA RTX 4060"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Receive an image and return helmet detection results in JSON format."""
    
    # 1. Kiểm tra định dạng file gửi lên
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # 2. Đọc dữ liệu binary từ file upload
        image_bytes = await file.read()
        
        # 3. Gọi 'bộ não' engine.py để xử lý
        results = process_detection(image_bytes)
        
        # 4. Trả về kết quả JSON
        return {
            "filename": file.filename,
            "detections": results,
            "total_persons": len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Chạy server tại cổng 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)