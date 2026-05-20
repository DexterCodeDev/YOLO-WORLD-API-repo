FROM python:3.10-slim

# Prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system graphics dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Automatically download YOLO-World Medium weights during the Docker build
RUN python -c "from ultralytics import YOLO; YOLO('yolov8m-world.pt')"

COPY . .

CMD ["python", "app.py"]
