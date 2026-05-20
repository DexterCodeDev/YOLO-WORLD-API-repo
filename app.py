import os
from fastapi import FastAPI, UploadFile, File
import uvicorn
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI()

# 1. Load the YOLO-World model globally
model = YOLO('yolov8m-world.pt')

# 2. Define your exact fashion vocabulary here
# You can add or remove items from this list anytime!
custom_classes = ["dress", "jeans", "earrings", "bangles", "shoes", "handbag"]
model.set_classes(custom_classes)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and decode the uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Run AI inference
    results = model(img)
    
    # Extract results
    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    confidences = results[0].boxes.conf.tolist()
    
    # Map the AI's class ID numbers back to your custom text names
    class_names = [custom_classes[int(c)] for c in classes]
    
    return {
        "boxes": boxes, 
        "classes": class_names, 
        "confidences": confidences
    }

if __name__ == "__main__":
    # Cloud Run requires binding to 0.0.0.0 and the injected PORT
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
