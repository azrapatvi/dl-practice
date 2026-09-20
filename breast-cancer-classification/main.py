from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
import joblib
import pandas as pd

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Load both models once at startup
ml_model = joblib.load('logistic_regression_model.pkl')
ann_model = load_model('ann_model.keras')  # <-- point this at your actual ANN file

# Load scaler (shared by both models — change if each model needs its own)
scaler = joblib.load("scaler.pkl")


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {}
    )


@app.post("/")
def predict(
    request: Request,

    model_type: str = Form(...),  # "ml" or "ann"

    radius_mean: float = Form(...),
    texture_mean: float = Form(...),
    perimeter_mean: float = Form(...),
    area_mean: float = Form(...),
    smoothness_mean: float = Form(...),
    compactness_mean: float = Form(...),
    concavity_mean: float = Form(...),
    concave_points_mean: float = Form(...),
    symmetry_mean: float = Form(...),
    fractal_dimension_mean: float = Form(...),

    radius_se: float = Form(...),
    texture_se: float = Form(...),
    perimeter_se: float = Form(...),
    area_se: float = Form(...),
    smoothness_se: float = Form(...),
    compactness_se: float = Form(...),
    concavity_se: float = Form(...),
    concave_points_se: float = Form(...),
    symmetry_se: float = Form(...),
    fractal_dimension_se: float = Form(...),

    radius_worst: float = Form(...),
    texture_worst: float = Form(...),
    perimeter_worst: float = Form(...),
    area_worst: float = Form(...),
    smoothness_worst: float = Form(...),
    compactness_worst: float = Form(...),
    concavity_worst: float = Form(...),
    concave_points_worst: float = Form(...),
    symmetry_worst: float = Form(...),
    fractal_dimension_worst: float = Form(...)
):

    # Build the feature DataFrame
    new_data = pd.DataFrame([{
        "radius_mean": radius_mean,
        "texture_mean": texture_mean,
        "perimeter_mean": perimeter_mean,
        "area_mean": area_mean,
        "smoothness_mean": smoothness_mean,
        "compactness_mean": compactness_mean,
        "concavity_mean": concavity_mean,
        "concave points_mean": concave_points_mean,
        "symmetry_mean": symmetry_mean,
        "fractal_dimension_mean": fractal_dimension_mean,

        "radius_se": radius_se,
        "texture_se": texture_se,
        "perimeter_se": perimeter_se,
        "area_se": area_se,
        "smoothness_se": smoothness_se,
        "compactness_se": compactness_se,
        "concavity_se": concavity_se,
        "concave points_se": concave_points_se,
        "symmetry_se": symmetry_se,
        "fractal_dimension_se": fractal_dimension_se,

        "radius_worst": radius_worst,
        "texture_worst": texture_worst,
        "perimeter_worst": perimeter_worst,
        "area_worst": area_worst,
        "smoothness_worst": smoothness_worst,
        "compactness_worst": compactness_worst,
        "concavity_worst": concavity_worst,
        "concave points_worst": concave_points_worst,
        "symmetry_worst": symmetry_worst,
        "fractal_dimension_worst": fractal_dimension_worst
    }])

    # Scale input (same scaler for both models — adjust if they were trained differently)
    new_data_scaled = scaler.transform(new_data)

    # Route to the model the user picked
    if model_type == "ann":
        # Keras model returns an array like [[0.87]]
        probability = float(ann_model.predict(new_data_scaled)[0][0])
        model_label = "ANN (Neural Network)"
    else:
        # sklearn LogisticRegression: use predict_proba for a real probability,
        # not predict() (which returns 0/1 class labels, not a probability)
        probability = float(ml_model.predict_proba(new_data_scaled)[0][1])
        model_label = "ML (Logistic Regression)"

    result = "Malignant" if probability >= 0.5 else "Benign"

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "result": result,
            "probability": probability,
            "model_label": model_label,
            "selected_model": model_type
        }
    )