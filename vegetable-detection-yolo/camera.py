from ultralytics import YOLO

if __name__ == "__main__":

    # Load our trained YOLO model
    model = YOLO(
        "E:/object_detection/runs/detect/train-3/weights/best.pt"
    )

    # Predict on our own image
    model.predict(
        source=0, # use webcam
        conf=0.5,
        imgsz=640,
        device=0,
        show=True # means show labels on the screen
    )