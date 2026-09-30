import streamlit as st
from ultralytics import YOLO
from PIL import Image

model = YOLO(r"E:\object_detection\runs\detect\train-3\weights\best.pt")

st.title("Vegetable Detection")
st.write("Take a photo and let YOLO detect the vegetables.")

picture = st.camera_input("Take a picture")

if picture is not None:

    image = Image.open(picture)

    st.subheader("Original Image")
    st.image(image)

    results = model.predict(
        source=image,
        conf=0.5,
        imgsz=640,
        device=0
    )

    result_image = results[0].plot()

    st.subheader("Detection Result")
    st.image(result_image, channels="BGR")