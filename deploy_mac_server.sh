#!/bin/bash
set -e

echo "🚀 Desplegando FORTINET TRADING BOT en MAC-SERVER..."

# 1. Obtener la última versión del repositorio en GitHub
git pull origin main

# 2. Levantar los contenedores en Docker Compose (PostgreSQL, App Python y Nginx Proxy)
if command -v docker-compose &> /dev/null; then
    docker-compose down --remove-orphans 2>/dev/null || true
    docker-compose up -d --build
else
    docker compose down --remove-orphans 2>/dev/null || true
    docker compose up -d --build
fi

echo ""
echo "=========================================================================="
echo "  ✅ DESPLIEGUE EN MAC-SERVER COMPLETADO EXITOSAMENTE"
echo "=========================================================================="
echo "  🌐 Acceso por URL Red Local:"
echo "     • http://192.168.1.222/fortinet/"
echo "     • http://192.168.1.222/"
echo "=========================================================================="
