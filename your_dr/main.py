"""Your Dr - FastAPI backend.
Run:  uvicorn main:app --reload
"""
import io
import json
import os
from datetime import datetime, timezone
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from bson import ObjectId
from bson.errors import InvalidId
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from pymongo import MongoClient
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_pre
from tensorflow.keras.models import load_model

# ------------------------------------------------------------------ setup
app = FastAPI()
templates = Jinja2Templates(directory="templates")
load_dotenv()  # reads MONGO_URI from the .env file next to main.py
MONGO_URI = "mongodb+srv://azrapatvi:azra1234@sample.2yxytjc.mongodb.net/"

db = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)["your_dr"]

MODEL_DIR = "models"
MAX_IMAGE_BYTES = 10 * 1024 * 1024

# Load everything ONCE at startup (not inside routes)
MODELS = {
    "brain": load_model(f"{MODEL_DIR}/brain_tumor_model_final1.h5"),
    "ovarian": load_model(f"{MODEL_DIR}/ovarian_vgg16_model.h5", compile=False),
    "kidney": load_model(f"{MODEL_DIR}/kidney_disease_detection.h5", compile=False),
}
# Alphabetical order = same as flow_from_directory class_indices in the notebooks
CLASSES = {
    "brain": ["glioma", "meningioma", "notumor", "pituitary"],
    "ovarian": ["complex_cyst", "dominant_follicle", "healthy", "poly_cyst", "simple_cyst"],
    "kidney": ["Cyst", "Normal", "Stone", "Tumor"],
}

# PCOS: sklearn preprocessor (pcos_preprocessor_fitted.pkl) + ANN (pcos_predictor_ann.h5)
PCOS_PRE = joblib.load(f"{MODEL_DIR}/pcos_preprocessor_fitted.pkl")
PCOS_MODEL = load_model(f"{MODEL_DIR}/pcos_predictor_ann.h5", compile=False)
if not hasattr(PCOS_PRE, "feature_names_in_"):
    raise RuntimeError(
        "pcos_preprocessor_fitted.pkl is NOT fitted. In the PCOS notebook run "
        "preprocessor.fit(X_train) and joblib.dump(preprocessor, 'pcos_preprocessor_fitted.pkl')."
    )
PCOS_COLS = list(PCOS_PRE.feature_names_in_)


def defaults_from_preprocessor(pre) -> dict:
    """Fallback defaults (training means) read from the fitted scalers."""
    d = {}
    for name, trans, cols in pre.transformers_:
        if name == "skewed":   # log1p -> scaler, so undo log1p
            means = np.expm1(trans.named_steps["scaler"].mean_)
        elif name == "normal":
            means = trans.named_steps["scaler"].mean_
        else:
            continue
        d.update(dict(zip(cols, map(float, means))))
    return d


# Better defaults (medians) if you saved them from the notebook:
#   json.dump(X_train.median().to_dict(), open("pcos_medians.json", "w"))
_med = f"{MODEL_DIR}/pcos_medians.json"
if os.path.exists(_med):
    with open(_med) as f:
        PCOS_MEDIANS = json.load(f)
else:
    PCOS_MEDIANS = defaults_from_preprocessor(PCOS_PRE)

# form field -> substring of the real dataset column name (lower-cased)
PCOS_FIELD_KEY = {
    "age": "age (yrs)", "weight": "weight (kg)", "height": "height(cm)",
    "cycle": "cycle(r/i)", "cycle_length": "cycle length", "waist": "waist(inch)",
    "hip": "hip(inch)", "weight_gain": "weight gain", "hair_growth": "hair growth",
    "skin_darkening": "skin darkening", "hair_loss": "hair loss", "pimples": "pimples",
    "fast_food": "fast food", "exercise": "reg.exercise", "follicle_l": "follicle no. (l)",
    "follicle_r": "follicle no. (r)", "amh": "amh", "fsh": "fsh(miu", "lh": "lh(miu",
}


def col(key: str) -> Optional[str]:
    return next((c for c in PCOS_COLS if key in c), None)


# ------------------------------------------------------------------ report text
# (title, is_abnormal, explanation)
INFO = {
    "brain": {
        "glioma": ("Glioma pattern", True, "The scan shows features the model associates with a glioma, a tumor of the brain's supporting cells."),
        "meningioma": ("Meningioma pattern", True, "The scan shows features the model associates with a meningioma, a tumor of the membranes around the brain."),
        "notumor": ("No tumor detected", False, "The model did not find features of the tumor types it was trained on."),
        "pituitary": ("Pituitary tumor pattern", True, "The scan shows features the model associates with a tumor of the pituitary gland."),
    },
    "ovarian": {
        "complex_cyst": ("Complex cyst pattern", True, "The image resembles a complex ovarian cyst, which has solid or mixed parts and needs clinical review."),
        "dominant_follicle": ("Dominant follicle", False, "This looks like a normal dominant follicle seen around ovulation."),
        "healthy": ("Healthy ovary", False, "No cyst or polycystic pattern was detected."),
        "poly_cyst": ("Polycystic pattern", True, "The image shows many small follicles, a pattern often linked with PCOS."),
        "simple_cyst": ("Simple cyst pattern", True, "The image resembles a simple fluid-filled ovarian cyst, which is often harmless but should be checked."),
    },
    "kidney": {
        "Cyst": ("Kidney cyst pattern", True, "The CT resembles a fluid-filled kidney cyst. Most are harmless, but a doctor should confirm."),
        "Normal": ("Normal kidney", False, "No cyst, stone or tumor pattern was detected."),
        "Stone": ("Kidney stone pattern", True, "The CT shows features of a kidney stone."),
        "Tumor": ("Kidney tumor pattern", True, "The CT shows features the model associates with a kidney tumor."),
    },
    "pcos": {
        "yes": ("PCOS likely", True, "Your answers resemble patterns seen in women diagnosed with PCOS."),
        "no": ("PCOS unlikely", False, "Your answers do not strongly resemble patterns seen in PCOS."),
    },
}
SPECIALIST = {
    "brain": ("neurologist", "a radiologist review of the original MRI and a contrast MRI if advised"),
    "ovarian": ("gynaecologist", "a repeat ultrasound and a hormone panel"),
    "kidney": ("urologist or nephrologist", "a radiologist review of the CT and kidney function tests"),
    "pcos": ("gynaecologist or endocrinologist", "hormone tests (LH, FSH, AMH, TSH, prolactin) and a pelvic ultrasound"),
}


def build_result(kind: str, probs: dict) -> dict:
    """probs: {class_key: probability}"""
    top = max(probs, key=probs.get)
    conf = float(probs[top])
    title, abnormal, info = INFO[kind][top]
    who, tests = SPECIALIST[kind]
    if abnormal:
        steps = [f"See a {who} soon and share this report along with your original scan or lab reports.",
                 f"Ask about confirmatory tests, such as {tests}.",
                 "Do not start or change any treatment based on this result alone."]
    else:
        steps = ["The model found nothing concerning, but this does not rule out disease.",
                 "If you have symptoms, see a doctor regardless of this result.",
                 "Keep up regular check-ups."]
    if conf < 0.70:
        steps.insert(0, "Confidence is low. The image may not match the selected scan type, or the result is unclear. Treat it as uncertain.")
    return {
        "label": title, "is_abnormal": abnormal, "confidence": conf,
        "probabilities": {INFO[kind][k][0]: float(v) for k, v in probs.items()},
        "info": info, "steps": steps,
    }


# ------------------------------------------------------------------ preprocessing / prediction
def predict_image(kind: str, img: Image.Image) -> dict:
    if kind == "brain":
        arr = np.array(img.convert("L").resize((256, 256)), dtype="float32") / 255.0
        arr = arr.reshape(1, 256, 256, 1)
    else:  # ovarian + kidney use VGG16 preprocessing
        arr = np.array(img.convert("RGB").resize((224, 224)), dtype="float32")
        arr = vgg_pre(np.expand_dims(arr, 0))
    out = MODELS[kind].predict(arr, verbose=0)[0]
    return {c: float(p) for c, p in zip(CLASSES[kind], out)}


def predict_pcos(values: dict) -> dict:
    row = dict(PCOS_MEDIANS)  # start from training medians
    for field, key in PCOS_FIELD_KEY.items():
        c, v = col(key), values.get(field)
        if c and v is not None:
            row[c] = v
    # derived columns
    w, h = values.get("weight"), values.get("height")
    if w and h and col("bmi"):
        row[col("bmi")] = round(w / ((h / 100) ** 2), 2)
    if values.get("waist") and values.get("hip") and col("waist:hip"):
        row[col("waist:hip")] = round(values["waist"] / values["hip"], 3)
    if values.get("fsh") and values.get("lh") and col("fsh/lh"):
        row[col("fsh/lh")] = round(values["fsh"] / values["lh"], 3)
    df = pd.DataFrame([row])[PCOS_COLS]
    X = np.asarray(PCOS_PRE.transform(df), dtype="float32")
    p = float(PCOS_MODEL.predict(X, verbose=0)[0][0])
    return {"no": 1 - p, "yes": p}


def get_patient_id(pid: str) -> ObjectId:
    try:
        oid = ObjectId(pid)
    except (InvalidId, TypeError):
        raise HTTPException(400, "Invalid patient id")
    if not db.patients.find_one({"_id": oid}):
        raise HTTPException(404, "Patient not found")
    return oid


def save_report(oid: ObjectId, kind: str, result: dict):
    db.reports.insert_one({
        "patient_id": oid, "type": kind, "label": result["label"],
        "is_abnormal": result["is_abnormal"], "confidence": result["confidence"],
        "probabilities": result["probabilities"], "created_at": datetime.now(timezone.utc),
    })


# ------------------------------------------------------------------ pages
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/diagnose")
def diagnose(request: Request):
    return templates.TemplateResponse(request=request, name="diagnose.html")


# ------------------------------------------------------------------ API
class Patient(BaseModel):
    name: str
    age: int
    gender: str
    phone: str
    email: Optional[str] = ""
    city: Optional[str] = ""
    consent: bool


@app.post("/api/patient")
def save_patient(p: Patient):
    if not p.consent:
        raise HTTPException(400, "Consent is required")
    if not (1 <= p.age <= 120) or not p.name.strip():
        raise HTTPException(400, "Invalid name or age")
    doc = p.model_dump()
    doc["created_at"] = datetime.now(timezone.utc)
    return {"patient_id": str(db.patients.insert_one(doc).inserted_id)}


@app.post("/api/predict-image")
async def api_predict_image(scan_type: str = Form(...), patient_id: str = Form(...),
                            file: UploadFile = File(...)):
    if scan_type not in MODELS:
        raise HTTPException(400, "Unknown scan type")
    oid = get_patient_id(patient_id)
    data = await file.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image too large (max 10 MB)")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Could not read this file as an image")
    probs = await run_in_threadpool(predict_image, scan_type, img)
    result = build_result(scan_type, probs)
    save_report(oid, scan_type, result)
    return result


class PcosForm(BaseModel):
    patient_id: str
    age: Optional[float] = None
    weight: Optional[float] = None
    height: Optional[float] = None
    cycle: Optional[float] = None
    cycle_length: Optional[float] = None
    waist: Optional[float] = None
    hip: Optional[float] = None
    weight_gain: Optional[float] = None
    hair_growth: Optional[float] = None
    skin_darkening: Optional[float] = None
    hair_loss: Optional[float] = None
    pimples: Optional[float] = None
    fast_food: Optional[float] = None
    exercise: Optional[float] = None
    follicle_l: Optional[float] = None
    follicle_r: Optional[float] = None
    amh: Optional[float] = None
    fsh: Optional[float] = None
    lh: Optional[float] = None


@app.post("/api/predict-form")
async def api_predict_form(form: PcosForm):
    oid = get_patient_id(form.patient_id)
    probs = await run_in_threadpool(predict_pcos, form.model_dump())
    result = build_result("pcos", probs)
    save_report(oid, "pcos", result)
    return result
