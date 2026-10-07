from ultralytics import YOLO

if __name__ == "__main__":

        #load pretrained mode
        model=YOLO(r"E:\object_detection\runs\detect\train-4\weights\best.pt")

        results = model.val(
            data=r"E:\object_detection\dataset\data.yaml", #my dataset configuration file. Use it to find my images, labels, and class names.
            split="test", #Don't evaluate on train or validation. Evaluate on the TEST dataset
            imgsz=640,
            batch=8,
            device=0,
            workers=0
        )