from fastapi import FastAPI, Request
import requests
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()

TF_URL = "http://tf-serving:8501/v1/models/sentiment:predict"

templates = Jinja2Templates(directory="app/templates")


# GUI PAGE
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# PREDICTION API
@app.post("/predict")
def predict(data: dict):

    text = data["text"]

    payload = {
        "instances": [text]
    }

    response = requests.post(TF_URL, json=payload).json()

    prediction = response["predictions"][0]

    labels = ["Negative", "Neutral", "Positive", "Sarcasm"]

    result = labels[prediction.index(max(prediction))]

    return {"prediction": result}