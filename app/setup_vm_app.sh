#!/bin/bash
set -e

echo "=== Iniciando la configuración de la aplicación en la VM ==="

# 1. Crear entorno virtual de Python
echo "Creando entorno virtual de Python..."
python3 -m venv /home/cplaza/app/venv

# 2. Instalar dependencias en el entorno virtual
echo "Instalando dependencias de Python..."
/home/cplaza/app/venv/bin/pip install --upgrade pip
/home/cplaza/app/venv/bin/pip install Flask requests psycopg2-binary scikit-learn joblib numpy

# 3. Asegurar que PostgreSQL está listo y acepta la conexión
echo "Verificando conexión con PostgreSQL..."
# Intentar conectarse a PostgreSQL usando Python y las credenciales del usuario
/home/cplaza/app/venv/bin/python -c "
import psycopg2
try:
    conn = psycopg2.connect(host='localhost', database='btc_trading', user='cplaza', password='9421')
    print('Conexión con PostgreSQL exitosa.')
    conn.close()
except Exception as e:
    print('Fallo al conectar, intentando configurar pg_hba.conf:', e)
    import sys
    sys.exit(1)
" || {
    echo "Fallo de conexión. Configurando pg_hba.conf para permitir conexiones locales de contraseña..."
    # Configurar pg_hba.conf para usar scram-sha-256/md5 en local
    PG_VERSION=$(psql --version | awk '{print $3}' | cut -d'.' -f1)
    PG_HBA="/etc/postgresql/$PG_VERSION/main/pg_hba.conf"
    
    # Hacer backup de pg_hba.conf y configurarlo para permitir conexiones md5
    sudo cp "$PG_HBA" "${PG_HBA}.bak"
    # Cambiar peer por md5 o scram-sha-256 en las líneas correspondientes
    sudo sed -i 's/local   all             all                                     peer/local   all             all                                     md5/g' "$PG_HBA"
    sudo sed -i 's/host    all             all             127.0.0.1\/32            scram-sha-256/host    all             all             127.0.0.1\/32            md5/g' "$PG_HBA"
    
    # Recargar postgresql
    sudo systemctl restart postgresql
    echo "PostgreSQL reiniciado con nueva configuración."
}

# 4. Configurar el servicio systemd para la aplicación
echo "Configurando el servicio systemd para BTC-MACHINE..."
sudo bash -c 'cat > /etc/systemd/system/btc-app.service <<EOF
[Unit]
Description=Aplicación Web de Trading de BTC
After=network.target postgresql.service

[Service]
User=cplaza
WorkingDirectory=/home/cplaza/app
ExecStart=/home/cplaza/app/venv/bin/python /home/cplaza/app/app.py
Restart=always
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF'

# 5. Arrancar y habilitar el servicio
echo "Iniciando servicio btc-app..."
sudo systemctl daemon-reload
sudo systemctl enable btc-app.service
sudo systemctl restart btc-app.service

echo "=== Configuración finalizada con éxito. La app está corriendo en el puerto 3333 ==="
