FROM python:3.9

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY app /app/app
COPY tokenizer.pkl /app
COPY model /app/model

CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]