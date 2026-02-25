# 1. Base image
FROM python:3.11-slim

# 2. Çalışma dizini
WORKDIR /app

# 3. Gereksinimleri kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Kod ve model dosyasını kopyala
COPY main.py .
COPY car_prediction_model.pkl .

# 5. Port ayarı
EXPOSE 8000

# 6. Uvicorn ile başlat
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]