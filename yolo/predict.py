import os
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

image_path = os.path.join(BASE_DIR, "..", "images", "1.jpg")

model = YOLO("yolov8n.pt")

results = model(image_path, save=True)

print("Prediction completed")