# Gunakan image Python sebagai dasar
FROM python:3.10.11-slim

# Buat direktori kerja
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Salin file requirements.txt dan instal dependensi
COPY requirements.txt /usr/src/app/
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode aplikasi ke direktori kerja
COPY . /usr/src/app

# Jalankan Gunicorn untuk menjalankan aplikasi Flask
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
