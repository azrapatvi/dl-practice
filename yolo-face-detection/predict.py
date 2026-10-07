from ultralytics import YOLO

if __name__ == "__main__":

    model=YOLO(r"E:\object_detection\runs\detect\train-4\weights\best.pt")

    result=model.predict(
        source=r"E:\object_detection\testing1.jpg",
        conf=0.5,
        imgsz=640,
        device=0,
        save=True
    )