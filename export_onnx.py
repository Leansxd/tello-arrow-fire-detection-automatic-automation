import os
from ultralytics import YOLO

def export_model(path):
    if os.path.exists(path):
        print(f"Exporting {path}...")
        model = YOLO(path)
        # imgsz should match what we use in fly.py or just use 480/320 as planned
        onnx_path = model.export(format="onnx", imgsz=480, simplify=True)
        print(f"Exported to {onnx_path}")
    else:
        print(f"File {path} not found.")

if __name__ == "__main__":
    export_model("best.pt")
    export_model("fire.pt")
