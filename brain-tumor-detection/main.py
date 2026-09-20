from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = FastAPI()

# allow index.html (opened as a local file) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model('brain_tumor_model.keras')
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("L")
    img = img.resize((256, 256))

    img_array = np.array(img) / 255.
    img_array = img_array.astype(np.float32)
    img_array = img_array.reshape(1, 256, 256, 1)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction) * 100)

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2)
    }


@app.get("/")
def home():
    return FileResponse("templates/index.html")