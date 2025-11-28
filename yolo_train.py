import ultralytics
import torch

from ultralytics import YOLO

def manin():
# Load a model
    model = YOLO("yolo11n-seg.pt")  

    # Train the model
    results = model.train(data="data.yaml", 
        epochs=10, 
        imgsz=1024, 
        device=0, 
        batch=-1,
        plots=True)

if __name__ == "__main__":
    manin()