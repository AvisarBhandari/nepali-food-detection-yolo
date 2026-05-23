from os import name

from ultralytics import YOLO
import cv2
from PIL import Image


def train(epochs=1):
    model = YOLO("yolov8n.pt")

    model.train(
        data="Dataset/data.yml",
        epochs=epochs,
        imgsz=640,
        save=True,
        patience=10,
        plots=True,
    )


def load_model(model_path, source=None, save=True, show=False):
    model = YOLO(model_path)
    results = model.predict(
        source=source,
        save=save,
        show=show,
    )

    for result in results:
        xywh = result.boxes.xywh
        xyxy = result.boxes.xyxy

        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]

        confs = result.boxes.conf

        print("Classes:", names)
        print("Confidence:", confs)


# load_model(model_path="runs/detect/train-2/weights/best.pt",source="Nepali_Food_Image/chatpate/images/chatpate8_png.rf.vxPvQQSR1hvEV6pJFL6o.png", show=True)
# if __name__ == "__main__":
#     train(epochs=1)
