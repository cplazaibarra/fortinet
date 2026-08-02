# 🤖 Fortinet Trading Bot (FTNT)

Bot de trading algorítmico para acciones de **Fortinet (FTNT)** con datos históricos de 3 años extraídos desde **Alpaca Markets**, análisis técnico avanzado y filtro inteligente de **Machine Learning**.

---

## 📊 Resultados del Backtest (Estrategia Óptima - Opción 1)

| Métrica | Sin ML | Con Filtro ML |
|---|---|---|
| **Capital Final** | $17,836.37 USDT | **$35,223.44 USDT** |
| **Ganancia Neta** | +78.36% | **+252.23%** |
| **Win Rate** | 46.8% | **80.6%** |
| **Trades Ejecutados** | 220 | 103 |
| **Pérdidas Bloqueadas** | — | 97 trades (-$23,715.55 evitados) |

> Período: 3 años · Datos históricos: 19,438 velas de 15 minutos

---

## 🧠 Estrategia Activa (Perfil 1)

**Entrada (AND):**
- ✅ MACD Line cruza por encima de MACD Signal
- ✅ RSI (14) > 45.0

**Salida (OR):**
- 💰 Take Profit >= **15.0%** 
- 🛡️ Trailing Stop dinámico >= **2.0%** desde el pico

**Filtro ML:**
- 🤖 RandomForest Meta-Model con 27 features (20 indicadores técnicos + 7 variables temporales)
- Umbral de confianza: **47.0%**

---

## 🏗️ Stack Técnico

| Componente | Tecnología |
|---|---|
| **Backend** | Python 3.11 + Flask |
| **Base de Datos** | PostgreSQL 15 (Docker) |
| **Machine Learning** | Scikit-learn (RandomForestClassifier) |
| **Datos** | Alpaca Markets API (15m OHLCV) |
| **Indicadores** | MACD, RSI, EMA 9/21/35/50/100/200, ATR, Bollinger Bands |
| **Frontend** | HTML5 + Vanilla JS + CSS Dark Pro Theme |

---

## 🚀 Instalación y Ejecución

### 1. Requisitos previos
```bash
# Docker (para PostgreSQL)
docker run --name fortinet-postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=fortinet_db -p 5432:5432 -d postgres:15

# Python 3.11+
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Variables de entorno
Crea un archivo `.env` (o configura en la UI web):
```env
ALPACA_API_KEY=tu_api_key
ALPACA_SECRET_KEY=tu_secret_key
DB_HOST=localhost
DB_PORT=5432
DB_NAME=fortinet_db
DB_USER=postgres
DB_PASS=postgres
```

### 3. Arrancar el servidor
```bash
PYTHONPATH=app .venv/bin/python3 app/app.py
```

Abrir en el navegador: **http://localhost:5050**

---

## 📁 Estructura del Proyecto

```
accion-fortinet/
├── app/
│   ├── app.py              # Flask server + lógica de trading
│   ├── database.py         # PostgreSQL: candles, trades, settings
│   ├── ml_engine.py        # RandomForest Meta-Model (27 features)
│   ├── alpaca_client.py    # Conexión a Alpaca Markets API
│   ├── static/
│   │   └── css/style.css   # Dark Pro Theme (Apple/Salesforce)
│   └── templates/          # Jinja2 HTML templates
│       ├── index.html      # Dashboard principal
│       ├── simulation.html # Simulador histórico
│       ├── chart.html      # Gráfica TradingView-style
│       ├── database_view.html
│       └── ...
├── data/
│   └── models/             # Modelos ML entrenados (excluidos de git)
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

## 📈 Features Principales

- **Dashboard en tiempo real**: Monitoreo vela por vela con checklist de condiciones en vivo
- **Filtro ML integrado**: La IA aparece como 5ª condición en la tabla de entrada
- **Simulador histórico**: Backtest paginado sobre 19,438 candles con comparación Sin ML vs Con ML
- **Trailing Stop dinámico**: Actualiza el pico de precio desde que se entra al trade
- **7 Variables Temporales ML**: Año, Mes, Semana ISO, Día, Día de semana, Hora, Minuto
- **Trading Manual**: Panel de compra/venta directa desde el Dashboard
- **Sync de datos**: Sincronización automática con Alpaca para candles de 15m en horario bursátil

---

## 📜 Licencia

MIT License — Uso libre para aprendizaje y proyectos personales.

---

*Desarrollado con ❤️ para trading algorítmico de Fortinet (FTNT) en el mercado NASDAQ.*
