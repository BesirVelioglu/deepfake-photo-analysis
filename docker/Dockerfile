# ==========================
# CPU için Hafif Python 3.8 + Ubuntu 22.04
# ==========================
FROM python:3.8-slim

# Çalışma dizinini belirle
WORKDIR /app

# ==========================
# Gerekli Sistem Kütüphaneleri
# ==========================
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ==========================
# Python Bağımlılıklarını Yükle
# ==========================

COPY docker/requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# ==========================
# Proje Dosyalarını Kopyala
# ==========================
COPY . /app/
WORKDIR /app/src/

# ==========================
# Varsayılan Komut
# ==========================
CMD ["python", "main.py"]
