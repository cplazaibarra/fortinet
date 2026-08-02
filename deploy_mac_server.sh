#!/bin/bash
set -e

echo "🚀 Iniciando despliegue de la arquitectura multi-app en MAC-SERVER..."

# 1. Crear red compartida de Docker 'apps-net' si no existe
if ! docker network ls | grep -q "apps-net"; then
    echo "🌐 Creando red de Docker 'apps-net'..."
    docker network create apps-net
fi

# 2. Desplegar App Fortinet en /home/cplaza/apps/fortinet
echo "📦 Desplegando Contenedores de Fortinet Trading Bot..."
if command -v docker-compose &> /dev/null; then
    docker-compose down --remove-orphans 2>/dev/null || true
    docker-compose up -d --build
else
    docker compose down --remove-orphans 2>/dev/null || true
    docker compose up -d --build
fi

# 3. Desplegar Proxy Reverso Nginx en /home/cplaza/apps/nginx-proxy si existe la carpeta
if [ -d "../nginx-proxy" ]; then
    echo "🌐 Actualizando Proxy Reverso Nginx en /home/cplaza/apps/nginx-proxy..."
    cd ../nginx-proxy
    if command -v docker-compose &> /dev/null; then
        docker-compose down --remove-orphans 2>/dev/null || true
        docker-compose up -d
    else
        docker compose down --remove-orphans 2>/dev/null || true
        docker compose up -d
    fi
    cd ../fortinet
fi

echo ""
echo "=========================================================================="
echo "  ✅ ARQUITECTURA EN MAC-SERVER DESPLEGADA EXITOSAMENTE"
echo "=========================================================================="
echo "  🌐 Acceso por URL Red Local:"
echo "     • Fortinet App: http://192.168.1.222/fortinet/"
echo "     • BTC App:      http://192.168.1.222/"
echo "=========================================================================="
