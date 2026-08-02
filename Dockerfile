FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema operativo
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar paquetes Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copiar el código fuente completo de la aplicación
COPY . .

ENV PYTHONPATH=/app

EXPOSE 5050

CMD ["python3", "app/app.py"]
