from ultralytics import YOLO
import os


def main():
    model = YOLO(r"yolov11_seg3\weights\best.pt") 

    for root, dirs, files in os.walk("dataset2/test/images"):
        for f in files:
            image_path = os.path.join(root, f)
            model.predict(image_path, save=True)

if __name__ == '__main__':
    main()