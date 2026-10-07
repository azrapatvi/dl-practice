from ultralytics import YOLO

if __name__ == "__main__":
        model = YOLO("yolov8n.pt")

        results =model.train(
            data=r"E:\object_detection\dataset\data.yaml",
            epochs=50,
            patience=5,
            imgsz=640,
            batch=4,
            device=0,
            workers=0

        )
