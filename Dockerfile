FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python rent_prediction.py --retrain

EXPOSE 8080
ENV HOST=0.0.0.0 PORT=8080

CMD ["python", "-m", "waitress", "--host=0.0.0.0", "--port=8080", "app:app"]
