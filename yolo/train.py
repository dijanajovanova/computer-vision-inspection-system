import os
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(BASE_DIR, "..", "dataset", "data.yaml")

model = YOLO("yolov8n.pt")

model.train(
    data=data_path,
    epochs=50,
    imgsz=640,
    batch=4,
    name="solar_defect_model"
)

print("Training completed")