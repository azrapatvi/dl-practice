import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av


# Load YOLO model
model = YOLO(r"E:\object_detection\runs\detect\train-3\weights\best.pt")


# Website title
st.title("Vegetable Detection")

st.write("Start the camera and show vegetables to the camera.")


# Function that processes every camera frame
def detect(frame):

    # Convert WebRTC frame to image
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

    # Convert image back to video frame
    return av.VideoFrame.from_ndarray(
        result,
        format="bgr24"
    )


# Start camera
webrtc_streamer(
    key="vegetable-detection",
    video_frame_callback=detect
)