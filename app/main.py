from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import re
import math
import emoji

app = FastAPI(title="SentimentIQ", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

TF_URL = "http://tf-serving:8501/v1/models/sentiment:predict"

LABELS = ["Negative", "Neutral", "Positive", "Sarcastic"]
LABEL_EMOJI = {
    "Positive":  "😊",
    "Negative":  "😡",
    "Neutral":   "😐",
    "Sarcastic": "🤨",
}

SLANG = {
    "lol": "laughing", "omg": "oh my god", "brb": "be right back",
    "idk": "i do not know", "btw": "by the way", "ngl": "not going to lie",
    "tbh": "to be honest", "imo": "in my opinion", "smh": "shaking my head",
    "ikr": "i know right", "rn": "right now", "fr": "for real",
    "lowkey": "somewhat", "highkey": "very much", "lit": "amazing",
    "goat": "greatest of all time", "slay": "doing great", "vibe": "feeling",
    "fire": "excellent", "sus": "suspicious", "cap": "lie",
    "bet": "okay agreed", "bussin": "really good",
}


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = emoji.demojize(text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = [SLANG.get(w, w) for w in text.split()]
    return " ".join(words).strip()


def explain_slang(text: str) -> str:
    words = text.lower().split()
    return " ".join(
        f"{w} -> {SLANG[w]}" if w in SLANG else w
        for w in words
    )


def softmax(logits: list) -> list:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [round(e / s, 4) for e in exps]


class PredictRequest(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
   return templates.TemplateResponse(request=request, name="index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model": "bert-base-uncased", "classes": LABELS}


@app.post("/predict")
def predict(data: PredictRequest):
    try:
        raw_text = data.text
        processed_text = clean_text(raw_text)

        payload = {"instances": [processed_text]}
        resp = requests.post(TF_URL, json=payload, timeout=10)
        resp.raise_for_status()
        raw_scores = resp.json()["predictions"][0]

        probs = softmax(raw_scores)
        pred_index = probs.index(max(probs))
        pred_label = LABELS[pred_index]
        confidence = round(max(probs), 4)
        scores_dict = {label: probs[i] for i, label in enumerate(LABELS)}

        return {
            "input":                raw_text,
            "processed_text":       processed_text,
            "prediction":           pred_label,
            "emoji":                LABEL_EMOJI[pred_label],
            "confidence":           confidence,
            "slang_interpretation": explain_slang(raw_text),
            "scores":               scores_dict,
        }

    except requests.exceptions.ConnectionError:
        return {"error": "Model server unreachable. Is TF Serving running?"}
    except Exception as e:
        return {"error": str(e)}