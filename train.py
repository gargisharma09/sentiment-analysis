import pandas as pd
import tensorflow as tf
import numpy as np
import re
import os
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers

# create model directory
os.makedirs("model/1", exist_ok=True)

# load dataset
df = pd.read_csv("social_media_sentiment_train.csv")

# ---------- TEXT CLEANING ----------
slang_dict = {
    "lol": "laughing",
    "omg": "oh my god",
    "brb": "be right back",
    "idk": "i do not know",
    "btw": "by the way"
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    words = text.split()
    words = [slang_dict.get(w, w) for w in words]
    return " ".join(words)

df["text"] = df["text"].astype(str).apply(clean_text)

# ---------- LABEL ENCODING ----------
df["label"] = df["label"].astype("category")
labels = df["label"].cat.codes.values

texts = df["text"].values

num_classes = len(np.unique(labels))

# ---------- TOKENIZER ----------
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)

padded = pad_sequences(
    sequences,
    maxlen=100,
    padding="post"
)

# save tokenizer
with open("tokenizer.pkl","wb") as f:
    pickle.dump(tokenizer,f)

# ---------- MODEL ----------
if num_classes == 2:
    output_units = 1
    activation = "sigmoid"
    loss_fn = "binary_crossentropy"
else:
    output_units = num_classes
    activation = "softmax"
    loss_fn = "sparse_categorical_crossentropy"

model = tf.keras.Sequential([
    layers.Embedding(10000,128,input_length=100),
    layers.LSTM(64),
    layers.Dense(64,activation="relu"),
    layers.Dense(output_units,activation=activation)
])

model.compile(
    optimizer="adam",
    loss=loss_fn,
    metrics=["accuracy"]
)

model.fit(
    padded,
    labels,
    epochs=5,
    batch_size=32
)

# ---------- EXPORT MODEL ----------
model.export("model/1")

print("Model exported successfully")