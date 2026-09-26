from fastapi import FastAPI,Request,Form
from fastapi.templating import Jinja2Templates
from utils import preprocess_text
from tensorflow.keras.models import load_model

model=load_model('sentiment_analysis.h5')

app=FastAPI()

templates=Jinja2Templates(directory='templates')

@app.get("/")
def home(request:Request):
    return templates.TemplateResponse(request=request,name="index.html")

@app.post("/predict")
def predict(request:Request,
            text:str=Form(...)):

    padded_text=preprocess_text(text)

    prediction=model.predict(padded_text)

    sentiment="Positive" if prediction[0][0]>0.5 else "negative"
    score = float(prediction[0][0])

    return templates.TemplateResponse(request=request,name='index.html',context={
        "prediction":score,
        "sentiment":sentiment
    })
