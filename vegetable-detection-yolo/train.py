from ultralytics import YOLO

if __name__ == "__main__":

    # Load pretrained YOLOv8 Nano model
    model = YOLO("yolov8n.pt")

    # Train the model
    results = model.train(
       
        data="E:/object_detection/dataset/data.yaml",
        epochs=50, #Maximum 50 training rounds
        imgsz=640, #Images are processed at 640×640 ; it means before giving any imge to the model make it of 640*640 pixels ; and this is vbecuase neural networks works best when they get fixed size of images everytime
        batch=4,
        device=0, # use my own gpu rtx
        patience=5,
        workers=0        # Safer for Windows; avoids multiprocessing issues
    )