# app/engine.py
import cv2
import numpy as np
from ultralytics import YOLO

# Đảm bảo đường dẫn này đúng với cấu trúc thư mục của bạn
model = YOLO("models/best.pt")

def process_detection(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    results = model(img, verbose=False)
    result = results[0]
    
    persons = []
    helmets = []
    detections = []

    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        if cls_id == 1:    # Person
            persons.append({"box": [x1, y1, x2, y2], "conf": conf})
        elif cls_id == 0:  # Helmet
            helmets.append({"box": [x1, y1, x2, y2], "conf": conf})

    for p in persons:
        px1, py1, px2, py2 = p["box"]
        head_y_limit = py1 + (py2 - py1) // 3
        status = "NO_HELMET"
        for h in helmets:
            hx1, hy1, hx2, hy2 = h["box"]
            if (px1 < hx2 and px2 > hx1 and py1 < hy2 and head_y_limit > hy1):
                status = "HELMET_OK"
                break
        
        detections.append({
            "label": status,
            "confidence": round(p["conf"], 2),
            "coordinates": {"x1": px1, "y1": py1, "x2": px2, "y2": py2}
        })
    return detections