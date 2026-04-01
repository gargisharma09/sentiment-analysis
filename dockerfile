FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app /app/app
COPY tokenizer.pkl /app
COPY model /app/model
COPY social_media_sentiment_train.csv /app
COPY social_media_sentiment_test.csv /app

# Create static directory if it doesn't exist
RUN mkdir -p /app/app/static

CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]