# Vegetable Detection using YOLOv8

This project is an **object detection system** that detects three types of vegetables:

* Onion
* Potato
* Tomato

The model is trained using **YOLOv8 Nano** and can detect vegetables from images, webcam video, and a real-time camera through a Streamlit web application.

---

## Project Overview

The main goal of this project is to train a custom YOLO model and use it for real-time vegetable detection.

The complete workflow is:

```text
Dataset
   ↓
YOLOv8n Pretrained Model
   ↓
Training
   ↓
Best Trained Model
   ↓
Testing
   ↓
Image / Webcam Prediction
   ↓
Streamlit Web Application
   ↓
Real-Time Vegetable Detection
```

---

## Classes

The model detects 3 classes:

| Class ID | Vegetable |
| -------- | --------- |
| 0        | Onion     |
| 1        | Potato    |
| 2        | Tomato    |

---

## Dataset

The dataset contains:

* **350 training images**
* **99 validation images**
* **51 test images**
* **500 images in total**

The dataset contains images and YOLO-format labels.

The dataset structure is:

```text
dataset/
│
├── train/
│   ├── images/
│   └── labels/
│
├── valid/
│   ├── images/
│   └── labels/
│
├── test/
│   ├── images/
│   └── labels/
│
└── data.yaml
```

### `data.yaml`

```yaml
train: train/images
val: valid/images
test: test/images

nc: 3
names: ['onion', 'potato', 'tomato']
```

---

# Model

I used **YOLOv8 Nano (`yolov8n.pt`)** as the pretrained model.

YOLOv8 Nano was selected because it is a lightweight model and is suitable for real-time object detection.

The model was then trained on the custom vegetable dataset.

---

# Training

The model was trained using:

```python
from ultralytics import YOLO

if __name__ == "__main__":

    # Load pretrained YOLOv8 Nano model
    model = YOLO("yolov8n.pt")

    # Train the model
    results = model.train(

        data="E:/object_detection/dataset/data.yaml",

        epochs=50,

        imgsz=640,

        batch=4,

        device=0,

        patience=5,

        workers=0
    )
```

### Training parameters

| Parameter  |   Value | Meaning                               |
| ---------- | ------: | ------------------------------------- |
| Model      | YOLOv8n | Lightweight YOLO model                |
| Epochs     |      50 | Maximum number of training rounds     |
| Image Size |     640 | Model input size                      |
| Batch Size |       4 | Images processed together             |
| Device     |       0 | NVIDIA GPU                            |
| Patience   |       5 | Early stopping                        |
| Workers    |       0 | Avoids Windows multiprocessing issues |

The model stopped early after **27 epochs** because validation performance stopped improving for the specified patience period.

---

# Model Performance

## Validation Results

The model achieved:

| Metric    | Result |
| --------- | -----: |
| Precision |  99.4% |
| Recall    |  99.1% |
| mAP50     |  98.9% |
| mAP50-95  |  82.7% |

### Test Results

The model was also evaluated on the separate test dataset.

| Metric    | Result |
| --------- | -----: |
| Precision |  99.8% |
| Recall    |   100% |
| mAP50     |  99.5% |
| mAP50-95  |  80.7% |

The test dataset contained **51 images and 120 annotated objects**.

> Note: The test set is relatively small, so these results should not be assumed to represent performance on every real-world image.

---

# What are the Evaluation Metrics?

### Precision

Precision tells us:

> When the model predicts an object, how often is that prediction correct?

### Recall

Recall tells us:

> How many of the actual objects did the model find?

### mAP50

mAP50 measures object detection performance when the predicted bounding box has an IoU of at least 0.50.

### mAP50-95

mAP50-95 is stricter. It calculates mAP across IoU thresholds from 0.50 to 0.95.

---

# Prediction on an Image

The trained model can be used to detect vegetables in a custom image.

Example:

```python
from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO(
        "E:/object_detection/runs/detect/train-3/weights/best.pt"
    )

    results = model.predict(
        source="E:/object_detection/many.jpg",
        conf=0.5,
        imgsz=640,
        device=0,
        save=True
    )
```

The model detects the vegetables and draws bounding boxes around them.

The output image is saved by YOLO.

---

# Real-Time Webcam Detection

YOLO can also directly use a webcam.

```python
from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO(
        "E:/object_detection/runs/detect/train-3/weights/best.pt"
    )

    model.predict(
        source=0,
        conf=0.5,
        imgsz=640,
        device=0,
        show=True
    )
```

Here:

```text
source=0
```

means the first available webcam.

The camera continuously sends frames to YOLO:

```text
Camera
   ↓
Frame
   ↓
YOLO
   ↓
Detection
   ↓
Next Frame
   ↓
YOLO
   ↓
Detection
   ↓
...
```

---

# Streamlit Real-Time Camera Application

The project also includes a Streamlit web application.

The application uses:

* Streamlit
* Streamlit WebRTC
* YOLO
* PyAV

The camera provides live video frames and YOLO processes those frames.

```text
Camera
   ↓
WebRTC
   ↓
Video Frame
   ↓
YOLO
   ↓
Bounding Boxes
   ↓
Browser
```

## Streamlit Code

```python
import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av


# Load YOLO model
model = YOLO("best.pt")


# Website title
st.title("Vegetable Detection")

st.write("Start the camera and show vegetables to the camera.")


# Process every camera frame
def detect(frame):

    # Convert video frame to image
    img = frame.to_ndarray(format="bgr24")

    # Run YOLO prediction
    results = model.predict(
        source=img,
        conf=0.5,
        imgsz=640,
        device=0,
        verbose=False
    )

    # Draw bounding boxes
    result = results[0].plot()

    # Convert back to video frame
    return av.VideoFrame.from_ndarray(
        result,
        format="bgr24"
    )


# Start camera
webrtc_streamer(
    key="vegetable-detection",
    video_frame_callback=detect
)
```

---

# How to Run the Streamlit Application

First install the required packages:

```bash
pip install -r requirements.txt
```

Then run:

```bash
streamlit run app.py
```

The application will open in the browser.

You can start the camera and show vegetables to the camera.

---

# Using the Application on a Phone

The Streamlit application can also be tested from a phone while the application is running on the laptop.

Both devices should be connected to the **same Wi-Fi network**.

Run:

```bash
streamlit run app.py --server.address 0.0.0.0
```

Then open the laptop's network address on the phone:

```text
http://YOUR-LAPTOP-IP:8501
```

For example:

```text
http://192.168.1.105:8501
```

The phone camera can then be used for real-time vegetable detection while the YOLO model runs on the laptop.

---

# Technologies Used

* Python
* YOLOv8
* Ultralytics
* PyTorch
* Streamlit
* Streamlit WebRTC
* OpenCV
* PyAV
* Pillow
* CUDA
* NVIDIA GPU

---

# Hardware

The model was trained and tested using an NVIDIA RTX 3050 6GB Laptop GPU.

GPU acceleration was used during training and inference.

---

# Project Structure

```text
vegetable-detection/
│
├── main.py
├── real_time_detection.py
├── train.py
├── test.py
├── predict.py
├── camera.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── dataset/
│   ├── train/
│   ├── valid/
│   └── test/
│
└── runs/
```

The `dataset/`, `runs/`, and model files are excluded from GitHub using `.gitignore`.

---

# Future Improvements

Some possible improvements for this project are:

* Deploy the application online
* Run the model directly on a mobile device
* Convert the YOLO model to a mobile-friendly format
* Improve real-time FPS
* Add more vegetable classes
* Add detection history
* Add vegetable counting
* Add a better mobile UI


This project was created as a practical project to learn custom object detection, YOLO, model evaluation, real-time inference, and deployment.
