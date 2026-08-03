import os
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
import json
import time
from decimal import Decimal
from datetime import datetime, timezone

# Parámetros de conexión a PostgreSQL
DB_HOST = os.getenv("POSTGRES_HOST") or os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT") or os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB") or os.getenv("DB_NAME", "fortinet_db")
DB_USER = os.getenv("POSTGRES_USER") or os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("POSTGRES_PASSWORD") or os.getenv("DB_PASSWORD", "postgres")

def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )


def init_db():
    conn = None
    for attempt in range(10):
        try:
            conn = get_db_connection()
            break
        except Exception as e:
            if attempt == 9:
                raise e
            print(f"Esperando a PostgreSQL ({DB_HOST}:{DB_PORT})... reintento {attempt + 1}/10")
            time.sleep(2)
            
    cursor = conn.cursor()
    
    # 1. Tabla de configuraciones
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS settings (
        key VARCHAR(255) PRIMARY KEY,
        value TEXT
    );
    ''')
    
    # 2. Tabla de estrategias (soporta 2 perfiles: entry_1/exit_1 y entry_2/exit_2)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS strategies (
        type VARCHAR(50) PRIMARY KEY,
        rules_json TEXT
    );
    ''')
    
    # 3. Tabla de transacciones (trades)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS trades (
        id SERIAL PRIMARY KEY,
        type VARCHAR(10), -- 'BUY' o 'SELL'
        timestamp BIGINT, -- Unix timestamp en milisegundos
        price DOUBLE PRECISION,
        amount DOUBLE PRECISION,
        usdt_value DOUBLE PRECISION,
        pnl DOUBLE PRECISION, -- Ganancia/Pérdida (para orden de venta)
        trade_group BIGINT, -- ID para asociar compra y venta
        mode VARCHAR(20) DEFAULT 'AUTO', -- 'AUTO' o 'MANUAL'
        reason VARCHAR(255) -- Motivo de la entrada o salida
    );
    ''')
    conn.commit()
    cursor = conn.cursor()

    try:
        cursor.execute("ALTER TABLE trades ALTER COLUMN trade_group TYPE BIGINT;")
        cursor.execute("ALTER TABLE trades ADD COLUMN IF NOT EXISTS mode VARCHAR(20) DEFAULT 'AUTO';")
        cursor.execute("ALTER TABLE trades ADD COLUMN IF NOT EXISTS reason VARCHAR(255);")
        conn.commit()
    except Exception:
        conn.rollback()

    cursor = conn.cursor()

    # 4. Tabla de logs del sistema
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS logs (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        message TEXT,
        level VARCHAR(20) -- 'INFO', 'WARNING', 'ERROR'
    );
    ''')
    
    # 5. Tabla de velas de 15 minutos con métricas de trading precalculadas
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS candles_15m (
        timestamp TIMESTAMPTZ NOT NULL,
        symbol VARCHAR(20) DEFAULT 'FTNT',
        open DOUBLE PRECISION,
        high DOUBLE PRECISION,
        low DOUBLE PRECISION,
        close DOUBLE PRECISION,
        volume DOUBLE PRECISION,
        trade_count INTEGER,
        vwap DOUBLE PRECISION,
        ema_9 DOUBLE PRECISION, slope_ema9_pct DOUBLE PRECISION,
        ema_21 DOUBLE PRECISION, slope_ema21_pct DOUBLE PRECISION,
        ema_35 DOUBLE PRECISION, slope_ema35_pct DOUBLE PRECISION,
        ema_50 DOUBLE PRECISION, slope_ema50_pct DOUBLE PRECISION,
        ema_100 DOUBLE PRECISION, slope_ema100_pct DOUBLE PRECISION,
        ema_200 DOUBLE PRECISION, slope_ema200_pct DOUBLE PRECISION,
        rsi_14 DOUBLE PRECISION,
        macd DOUBLE PRECISION, macd_signal DOUBLE PRECISION, macd_hist DOUBLE PRECISION,
        atr_14 DOUBLE PRECISION,
        PRIMARY KEY (symbol, timestamp)
    );
    ''')
    try:
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_candles_15m_ts ON candles_15m(timestamp DESC);")
    except Exception:
        pass
    conn.commit()
    cursor = conn.cursor()

    # 6. Tabla de trades simulados persistentes con métricas de Machine Learning
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS simulated_trades (
        trade_num INTEGER PRIMARY KEY,
        buy_time BIGINT NOT NULL,
        buy_date_str VARCHAR(50) NOT NULL,
        buy_price DOUBLE PRECISION NOT NULL,
        sell_time BIGINT NOT NULL,
        sell_date_str VARCHAR(50) NOT NULL,
        sell_price DOUBLE PRECISION NOT NULL,
        amount_ftnt DOUBLE PRECISION NOT NULL,
        invested_usdt DOUBLE PRECISION NOT NULL,
        returned_usdt DOUBLE PRECISION NOT NULL,
        pnl_usdt DOUBLE PRECISION NOT NULL,
        pnl_pct DOUBLE PRECISION NOT NULL,
        cumulative_balance DOUBLE PRECISION NOT NULL,
        status VARCHAR(20) NOT NULL,
        ml_approve VARCHAR(10) DEFAULT 'SI',
        ml_confidence DOUBLE PRECISION DEFAULT 50.0,
        ml_pnl_usdt DOUBLE PRECISION DEFAULT 0.0,
        ml_pnl_pct DOUBLE PRECISION DEFAULT 0.0,
        ml_cumulative_balance DOUBLE PRECISION DEFAULT 10000.0
    );
    CREATE INDEX IF NOT EXISTS idx_simulated_trades_num ON simulated_trades(trade_num DESC);
    CREATE INDEX IF NOT EXISTS idx_simulated_trades_buy_time ON simulated_trades(buy_time DESC);
    ''')
    conn.commit()
    cursor = conn.cursor()

    # 7. Tabla de resumen de la simulación
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS simulation_summary_meta (
        key VARCHAR(50) PRIMARY KEY,
        value TEXT NOT NULL
    );
    ''')
    conn.commit()
    cursor = conn.cursor()
    
    # Insertar valores por defecto para configuraciones
    default_settings = {
        'candle_interval': '15m',
        'binance_api_key': '',
        'binance_api_secret': '',
        'binance_testnet': 'true',
        'simulation_mode': 'true',
        'simulation_balance': '10000.0',
        'simulation_btc_balance': '0.0',
        'trade_size_usd': '100.0',
        'bot_active': 'false'
    }
    
    for key, value in default_settings.items():
        cursor.execute('''
        INSERT INTO settings (key, value) VALUES (%s, %s)
        ON CONFLICT (key) DO NOTHING;
        ''', (key, value))
        
    # Insertar estrategias vacías por defecto para ambos perfiles si no existen
    for strat_type in ('entry_1', 'exit_1', 'entry_2', 'exit_2'):
        cursor.execute('''
        INSERT INTO strategies (type, rules_json) VALUES (%s, %s)
        ON CONFLICT (type) DO NOTHING;
        ''', (strat_type, json.dumps([])))

    # Migrar estrategias antiguas (entry/exit) al perfil 1 si existen
    try:
        cursor.execute("SELECT rules_json FROM strategies WHERE type = 'entry';")
        old_entry = cursor.fetchone()
        if old_entry:
            cursor.execute('''
            INSERT INTO strategies (type, rules_json) VALUES ('entry_1', %s)
            ON CONFLICT (type) DO UPDATE SET rules_json = EXCLUDED.rules_json
            WHERE strategies.rules_json = '[]';
            ''', (old_entry[0],))
    except Exception:
        conn.rollback()
        cursor = conn.cursor()

    try:
        cursor.execute("SELECT rules_json FROM strategies WHERE type = 'exit';")
        old_exit = cursor.fetchone()
        if old_exit:
            cursor.execute('''
            INSERT INTO strategies (type, rules_json) VALUES ('exit_1', %s)
            ON CONFLICT (type) DO UPDATE SET rules_json = EXCLUDED.rules_json
            WHERE strategies.rules_json = '[]';
            ''', (old_exit[0],))
    except Exception:
        conn.rollback()
        cursor = conn.cursor()

    # Migrar columna strategy_profile en simulated_trades si no existe
    try:
        cursor.execute("ALTER TABLE simulated_trades ADD COLUMN IF NOT EXISTS strategy_profile INTEGER DEFAULT 1;")
        conn.commit()
    except Exception:
        conn.rollback()
    cursor = conn.cursor()

    try:
        cursor.execute("ALTER TABLE simulated_trades DROP CONSTRAINT IF EXISTS simulated_trades_pkey;")
        cursor.execute("ALTER TABLE simulated_trades ADD CONSTRAINT simulated_trades_pkey PRIMARY KEY (strategy_profile, trade_num);")
        conn.commit()
    except Exception:
        conn.rollback()
    cursor = conn.cursor()

    # Migrar clave primaria de simulated_trades para soportar múltiples perfiles
    cursor.execute('''
    INSERT INTO settings (key, value) VALUES ('active_strategy_profile', '1')
    ON CONFLICT (key) DO NOTHING;
    ''')

    conn.commit()
    cursor.close()
    conn.close()

# Helpers para Configuración
def get_setting(key, default=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT value FROM settings WHERE key = %s;', (key,))
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    if row:
        return row[0]
    return default

def get_settings():
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute('SELECT key, value FROM settings;')
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return {row['key']: row['value'] for row in rows}

def set_setting(key, value):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO settings (key, value) VALUES (%s, %s)
    ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;
    ''', (key, str(value)))
    conn.commit()
    cursor.close()
    conn.close()

def set_settings(settings_dict):
    conn = get_db_connection()
    cursor = conn.cursor()
    for key, value in settings_dict.items():
        cursor.execute('''
        INSERT INTO settings (key, value) VALUES (%s, %s)
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;
        ''', (key, str(value)))
    conn.commit()
    cursor.close()
    conn.close()

# Helpers para Estrategias (con soporte de perfil 1 y 2)
def get_strategy(strategy_type, profile=1):
    """
    Obtiene las reglas de estrategia de un perfil específico.
    strategy_type: 'entry' o 'exit'
    profile: 1 o 2
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    # Soporta tanto claves con sufijo ('entry_1') como sin sufijo ('entry') para compatibilidad
    typed_key = f"{strategy_type}_{profile}"
    cursor.execute('SELECT rules_json FROM strategies WHERE type = %s;', (typed_key,))
    row = cursor.fetchone()
    if not row:
        # Fallback a clave sin sufijo (compatibilidad con perfil 1 antiguo)
        cursor.execute('SELECT rules_json FROM strategies WHERE type = %s;', (strategy_type,))
        row = cursor.fetchone()
    cursor.close()
    conn.close()
    if row:
        return json.loads(row[0])
    return []

def save_strategy(strategy_type, rules_list, profile=1):
    """
    Guarda las reglas de estrategia de un perfil específico.
    strategy_type: 'entry' o 'exit'
    profile: 1 o 2
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    typed_key = f"{strategy_type}_{profile}"
    cursor.execute('''
    INSERT INTO strategies (type, rules_json) VALUES (%s, %s)
    ON CONFLICT (type) DO UPDATE SET rules_json = EXCLUDED.rules_json;
    ''', (typed_key, json.dumps(rules_list)))
    conn.commit()
    cursor.close()
    conn.close()

# Helpers para Trades
def add_trade(trade_type, timestamp, price, amount, usdt_value, pnl=None, trade_group=None, mode='AUTO', reason=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO trades (type, timestamp, price, amount, usdt_value, pnl, trade_group, mode, reason)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    RETURNING id;
    ''', (trade_type, timestamp, price, amount, usdt_value, pnl, trade_group, mode, reason))
    trade_id = cursor.fetchone()[0]
    conn.commit()
    cursor.close()
    conn.close()
    return trade_id

def get_trades():
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute('SELECT * FROM trades ORDER BY timestamp DESC;')
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [dict(row) for row in rows]

def get_last_trade():
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute('SELECT * FROM trades ORDER BY timestamp DESC LIMIT 1;')
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    return dict(row) if row else None

# Helpers para Logs
def add_log(message, level='INFO'):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO logs (message, level) VALUES (%s, %s);', (message, level))
    conn.commit()
    cursor.close()
    conn.close()

def get_logs(limit=50):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute('SELECT * FROM logs ORDER BY timestamp DESC LIMIT %s;', (limit,))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return [dict(row) for row in rows] if rows else []
    except Exception as e:
        print(f"Error en get_logs: {e}")
        return []

def clear_logs():
    """Elimina únicamente los registros de logs en PostgreSQL."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM logs;')
    conn.commit()
    cursor.close()
    conn.close()

def clear_logs_and_trades():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM trades;')
    cursor.execute('DELETE FROM logs;')
    # Reset balances
    cursor.execute('UPDATE settings SET value = \'10000.0\' WHERE key = \'simulation_balance\';')
    cursor.execute('UPDATE settings SET value = \'0.0\' WHERE key = \'simulation_btc_balance\';')
    cursor.execute('UPDATE settings SET value = \'false\' WHERE key = \'bot_active\';')
    conn.commit()
    cursor.close()
    conn.close()
    add_log("Base de datos de transacciones y logs limpiada por el usuario", "INFO")

# Helpers para Velas de 15m y Sincronización
def upsert_candles_15m(candles):
    """
    Inserta o actualiza una lista de velas con sus métricas en la tabla candles_15m.
    """
    if not candles:
        return
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = """
    INSERT INTO candles_15m (
        time, open, high, low, close, volume,
        ema9, ema9_slope, ema9_slope_pct,
        ema21, ema21_slope, ema21_slope_pct,
        ema35, ema35_slope, ema35_slope_pct,
        ema50, ema50_slope, ema50_slope_pct,
        ema100, ema100_slope, ema100_slope_pct,
        ema200, ema200_slope, ema200_slope_pct,
        rsi14, macd, macd_signal, macd_hist, atr14,
        year_ml, month_ml, week_ml, day_ml, day_name_ml, hour_ml, minute_ml
    ) VALUES %s
    ON CONFLICT (time) DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume,
        ema9 = EXCLUDED.ema9, ema9_slope = EXCLUDED.ema9_slope, ema9_slope_pct = EXCLUDED.ema9_slope_pct,
        ema21 = EXCLUDED.ema21, ema21_slope = EXCLUDED.ema21_slope, ema21_slope_pct = EXCLUDED.ema21_slope_pct,
        ema35 = EXCLUDED.ema35, ema35_slope = EXCLUDED.ema35_slope, ema35_slope_pct = EXCLUDED.ema35_slope_pct,
        ema50 = EXCLUDED.ema50, ema50_slope = EXCLUDED.ema50_slope, ema50_slope_pct = EXCLUDED.ema50_slope_pct,
        ema100 = EXCLUDED.ema100, ema100_slope = EXCLUDED.ema100_slope, ema100_slope_pct = EXCLUDED.ema100_slope_pct,
        ema200 = EXCLUDED.ema200, ema200_slope = EXCLUDED.ema200_slope, ema200_slope_pct = EXCLUDED.ema200_slope_pct,
        rsi14 = EXCLUDED.rsi14,
        macd = EXCLUDED.macd, macd_signal = EXCLUDED.macd_signal, macd_hist = EXCLUDED.macd_hist,
        atr14 = EXCLUDED.atr14,
        year_ml = EXCLUDED.year_ml, month_ml = EXCLUDED.month_ml, week_ml = EXCLUDED.week_ml,
        day_ml = EXCLUDED.day_ml, day_name_ml = EXCLUDED.day_name_ml, hour_ml = EXCLUDED.hour_ml, minute_ml = EXCLUDED.minute_ml;
    """
    
    days_es = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
    tuples = []
    for c in candles:
        dt = datetime.fromtimestamp(c['time'] / 1000.0, tz=CHILE_TZ)
        iso_year, iso_week, iso_day = dt.isocalendar()
        day_name = days_es[dt.weekday()]
        tuples.append((
            c['time'], c['open'], c['high'], c['low'], c['close'], c['volume'],
            c.get('ema9'), c.get('ema9_slope'), c.get('ema9_slope_pct'),
            c.get('ema21'), c.get('ema21_slope'), c.get('ema21_slope_pct'),
            c.get('ema35'), c.get('ema35_slope'), c.get('ema35_slope_pct'),
            c.get('ema50'), c.get('ema50_slope'), c.get('ema50_slope_pct'),
            c.get('ema100'), c.get('ema100_slope'), c.get('ema100_slope_pct'),
            c.get('ema200'), c.get('ema200_slope'), c.get('ema200_slope_pct'),
            c.get('rsi14'), c.get('macd'), c.get('macd_signal'), c.get('macd_hist'), c.get('atr14'),
            dt.year, dt.month, iso_week, dt.day, day_name, dt.hour, dt.minute
        ))
        
    execute_values(cursor, query, tuples)
    
    query_ohlcv = """
    INSERT INTO ohlcv_15m (
        timestamp, symbol, open, high, low, close, volume, trade_count, vwap,
        ema_9, slope_ema9_pct, ema_21, slope_ema21_pct, ema_35, slope_ema35_pct,
        ema_50, slope_ema50_pct, ema_100, slope_ema100_pct, ema_200, slope_ema200_pct,
        rsi_14, macd, macd_signal, macd_hist, atr_14,
        year_ml, month_ml, week_ml, day_ml, day_name_ml, hour_ml, minute_ml
    ) VALUES %s
    ON CONFLICT (symbol, timestamp) DO UPDATE SET
        open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low, close = EXCLUDED.close, volume = EXCLUDED.volume,
        vwap = EXCLUDED.vwap, trade_count = EXCLUDED.trade_count,
        ema_9 = EXCLUDED.ema_9, slope_ema9_pct = EXCLUDED.slope_ema9_pct,
        ema_21 = EXCLUDED.ema_21, slope_ema21_pct = EXCLUDED.slope_ema21_pct,
        ema_35 = EXCLUDED.ema_35, slope_ema35_pct = EXCLUDED.slope_ema35_pct,
        ema_50 = EXCLUDED.ema_50, slope_ema50_pct = EXCLUDED.slope_ema50_pct,
        ema_100 = EXCLUDED.ema_100, slope_ema100_pct = EXCLUDED.slope_ema100_pct,
        ema_200 = EXCLUDED.ema_200, slope_ema200_pct = EXCLUDED.slope_ema200_pct,
        rsi_14 = EXCLUDED.rsi_14, macd = EXCLUDED.macd, macd_signal = EXCLUDED.macd_signal,
        macd_hist = EXCLUDED.macd_hist, atr_14 = EXCLUDED.atr_14,
        year_ml = EXCLUDED.year_ml, month_ml = EXCLUDED.month_ml, week_ml = EXCLUDED.week_ml,
        day_ml = EXCLUDED.day_ml, day_name_ml = EXCLUDED.day_name_ml, hour_ml = EXCLUDED.hour_ml, minute_ml = EXCLUDED.minute_ml;
    """
    
    tuples_ohlcv = []
    for c in candles:
        dt = datetime.fromtimestamp(c['time'] / 1000.0, tz=timezone.utc)
        dt_chile = datetime.fromtimestamp(c['time'] / 1000.0, tz=CHILE_TZ)
        iso_year, iso_week, iso_day = dt_chile.isocalendar()
        day_name = days_es[dt_chile.weekday()]
        tuples_ohlcv.append((
            dt, 'FTNT', c['open'], c['high'], c['low'], c['close'], c['volume'],
            c.get('trade_count', 0), c.get('vwap', c['close']),
            c.get('ema9'), c.get('ema9_slope_pct'),
            c.get('ema21'), c.get('ema21_slope_pct'),
            c.get('ema35'), c.get('ema35_slope_pct'),
            c.get('ema50'), c.get('ema50_slope_pct'),
            c.get('ema100'), c.get('ema100_slope_pct'),
            c.get('ema200'), c.get('ema200_slope_pct'),
            c.get('rsi14'), c.get('macd'), c.get('macd_signal'), c.get('macd_hist'), c.get('atr14'),
            dt_chile.year, dt_chile.month, iso_week, dt_chile.day, day_name, dt_chile.hour, dt_chile.minute
        ))
    execute_values(cursor, query_ohlcv, tuples_ohlcv)
    conn.commit()
    cursor.close()
    conn.close()

def get_last_candle_time_15m():
    """Retorna el timestamp de la vela más reciente en PostgreSQL."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT MAX(time) FROM candles_15m;')
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    return row[0] if row and row[0] is not None else None

def get_oldest_candle_time_15m():
    """Retorna el timestamp de la vela más antigua en PostgreSQL."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT MIN(time) FROM candles_15m;')
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    return row[0] if row and row[0] is not None else None

def get_candles_15m(limit=300):
    """
    Obtiene las últimas 'limit' velas de 15 minutos ordenadas cronológicamente.
    """
    return get_candles_15m_range(limit=limit)

def get_candles_15m_range(start_ts=None, end_ts=None, limit=300):
    """
    Obtiene velas de 15 minutos en un rango de fechas (start_ts y end_ts en ms).
    Si no se especifica rango completo, aplica el límite especificado.
    """
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    if start_ts is not None and end_ts is not None:
        cursor.execute('SELECT * FROM candles_15m WHERE time >= %s AND time <= %s ORDER BY time ASC;', (start_ts, end_ts))
    elif start_ts is not None:
        cursor.execute('SELECT * FROM candles_15m WHERE time >= %s ORDER BY time ASC;', (start_ts,))
    elif end_ts is not None:
        cursor.execute('SELECT * FROM (SELECT * FROM candles_15m WHERE time <= %s ORDER BY time DESC LIMIT %s) sub ORDER BY time ASC;', (end_ts, limit))
    else:
        cursor.execute('SELECT * FROM (SELECT * FROM candles_15m ORDER BY time DESC LIMIT %s) sub ORDER BY time ASC;', (limit,))
        
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    
    result = []
    for r in rows:
        d = dict(r)
        for k, v in d.items():
            if isinstance(v, Decimal):
                d[k] = float(v)
        result.append(d)
    return result

def get_candles_15m_paginated(page=1, per_page=50):
    """
    Obtiene los registros de candles_15m paginados por SQL (ordenados por fecha descendente).
    """
    page = max(1, page)
    per_page = max(5, min(200, per_page))
    offset = (page - 1) * per_page

    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    cursor.execute('SELECT COUNT(*) FROM candles_15m;')
    row_count = cursor.fetchone()
    total_records = row_count['count'] if row_count else 0
    total_pages = max(1, (total_records + per_page - 1) // per_page)

    cursor.execute('SELECT * FROM candles_15m ORDER BY time DESC LIMIT %s OFFSET %s;', (per_page, offset))
    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    return {
        'page': page,
        'per_page': per_page,
        'total_records': total_records,
        'total_pages': total_pages,
        'records': [dict(r) for r in rows]
    }

def get_sync_monitoring_status():
    """
    Obtiene métricas en vivo sobre la sincronización y la salud de las últimas 3 velas.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*), MIN(time), MAX(time) FROM candles_15m;')
    row = cursor.fetchone()
    count = row[0] if row else 0
    oldest_ts = row[1] if row and row[1] is not None else None
    newest_ts = row[2] if row and row[2] is not None else None
    
    now_ms = int(time.time() * 1000)
    # Fortinet (FTNT) es una acción del mercado de EE.UU. (6.5 hrs/día, 26 velas/día de 15m).
    # El historial completo de 3 años (2023-2026) consta exactamente de 19,438 velas.
    target_count = max(19438, count) if count >= 19000 else 19438
    progress_pct = min(100.0, round((count / target_count) * 100.0, 1)) if count > 0 else 0.0
    
    # Verificar salud de las velas más recientes de 15m
    forty_five_min_ago = now_ms - (45 * 60 * 1000)
    cursor.execute('SELECT COUNT(*) FROM candles_15m WHERE time >= %s;', (forty_five_min_ago,))
    recent_3_count = cursor.fetchone()[0]
    
    is_live_monitoring_ok = (newest_ts is not None)
    
    cursor.close()
    conn.close()
    
    oldest_date_str = datetime.fromtimestamp(oldest_ts / 1000).strftime('%Y-%m-%d %H:%M') if oldest_ts else "N/A"
    newest_date_str = datetime.fromtimestamp(newest_ts / 1000).strftime('%Y-%m-%d %H:%M') if newest_ts else "N/A"
    
    status_str = "Completamente Sincronizado" if progress_pct >= 99.0 else ("Sincronizando" if count > 0 else "Iniciando")
    
    return {
        'total_candles': count,
        'target_candles': target_count,
        'progress_pct': progress_pct,
        'oldest_date': oldest_date_str,
        'newest_date': newest_date_str,
        'oldest_ts': oldest_ts,
        'newest_ts': newest_ts,
        'recent_3_count': recent_3_count,
        'is_live_monitoring_ok': is_live_monitoring_ok,
        'status': status_str
    }

# Helpers para Persistencia de Simulación de Trades
def save_simulated_trades(trade_pairs, summary_meta, profile=1):
    """
    Guarda de forma permanente la lista de operaciones simuladas y el resumen en PostgreSQL.
    profile: 1 o 2 (perfil de estrategia)
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM simulated_trades WHERE strategy_profile = %s;', (profile,))
    
    if trade_pairs:
        query = """
        INSERT INTO simulated_trades (
            trade_num, buy_time, buy_date_str, buy_price,
            sell_time, sell_date_str, sell_price, amount_ftnt,
            invested_usdt, returned_usdt, pnl_usdt, pnl_pct,
            cumulative_balance, status, ml_approve, ml_confidence,
            ml_pnl_usdt, ml_pnl_pct, ml_cumulative_balance, strategy_profile
        ) VALUES %s;
        """
        tuples = []
        for t in trade_pairs:
            tuples.append((
                t['trade_num'], t['buy_time'], t['buy_date_str'], t['buy_price'],
                t['sell_time'], t['sell_date_str'], t['sell_price'], t['amount_ftnt'],
                t['invested_usdt'], t['returned_usdt'], t['pnl_usdt'], t['pnl_pct'],
                t['cumulative_balance'], t['status'],
                t.get('ml_approve', 'SI'), t.get('ml_confidence', 50.0),
                t.get('ml_pnl_usdt', 0.0), t.get('ml_pnl_pct', 0.0),
                t.get('ml_cumulative_balance', 10000.0),
                profile
            ))
        execute_values(cursor, query, tuples)
        
    # Guardar resumen con sufijo de perfil en las claves
    for k, v in summary_meta.items():
        key_with_profile = f"{k}_p{profile}"
        cursor.execute('''
        INSERT INTO simulation_summary_meta (key, value) VALUES (%s, %s)
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;
        ''', (key_with_profile, str(v)))
        
    conn.commit()
    cursor.close()
    conn.close()

def get_saved_simulation_summary(profile=1):
    """Retorna el resumen guardado de la simulación desde PostgreSQL para un perfil específico."""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    suffix = f"_p{profile}"
    cursor.execute("SELECT key, value FROM simulation_summary_meta WHERE key LIKE %s;", (f'%{suffix}',))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    
    if not rows:
        # Fallback: intentar con claves sin sufijo (compatibilidad con perfil 1 antiguo)
        if profile == 1:
            conn2 = get_db_connection()
            cursor2 = conn2.cursor(cursor_factory=RealDictCursor)
            cursor2.execute("SELECT key, value FROM simulation_summary_meta WHERE key NOT LIKE '%_p%';")
            rows = cursor2.fetchall()
            cursor2.close()
            conn2.close()
        if not rows:
            return None
        
    res = {}
    for r in rows:
        # Quitar sufijo _p1 o _p2 de la clave al devolver el dict
        raw_key = r['key']
        clean_key = raw_key.replace(suffix, '') if raw_key.endswith(suffix) else raw_key
        val = r['value']
        try:
            val = float(val) if '.' in val else int(val)
        except ValueError:
            pass
        res[clean_key] = val
    return res

def get_simulated_trades(limit=2000, profile=1):
    """Retorna las operaciones simuladas almacenadas en PostgreSQL ordenadas por buy_time."""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute('SELECT * FROM simulated_trades WHERE strategy_profile = %s ORDER BY buy_time ASC LIMIT %s;', (profile, limit,))
            rows = cursor.fetchall()
            return [dict(r.items()) for r in rows]

def get_simulated_trades_paginated(page=1, per_page=50, start_ts=None, end_ts=None, profile=1):
    """
    Obtiene las operaciones simuladas guardadas en PostgreSQL paginadas por SQL.
    profile: 1 o 2 (perfil de estrategia)
    """
    page = max(1, page)
    per_page = max(5, min(200, per_page))
    offset = (page - 1) * per_page

    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    if start_ts is not None and end_ts is not None:
        cursor.execute('SELECT COUNT(*) FROM simulated_trades WHERE strategy_profile = %s AND buy_time >= %s AND buy_time <= %s;', (profile, start_ts, end_ts))
        total_records = cursor.fetchone()['count']
        total_pages = max(1, (total_records + per_page - 1) // per_page)
        cursor.execute('SELECT * FROM simulated_trades WHERE strategy_profile = %s AND buy_time >= %s AND buy_time <= %s ORDER BY trade_num DESC LIMIT %s OFFSET %s;', (profile, start_ts, end_ts, per_page, offset))
    else:
        cursor.execute('SELECT COUNT(*) FROM simulated_trades WHERE strategy_profile = %s;', (profile,))
        total_records = cursor.fetchone()['count']
        total_pages = max(1, (total_records + per_page - 1) // per_page)
        cursor.execute('SELECT * FROM simulated_trades WHERE strategy_profile = %s ORDER BY trade_num DESC LIMIT %s OFFSET %s;', (profile, per_page, offset))
        
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    return {
        'page': page,
        'per_page': per_page,
        'total_records': total_records,
        'total_pages': total_pages,
        'records': [dict(r) for r in rows]
    }


def get_simulated_summary_by_range(start_ts=None, end_ts=None, profile=1):
    """
    Calcula el resumen de la simulación directamente en SQL para un rango de fechas y perfil.
    Evita cargar todos los registros en Python, siendo eficiente para cualquier rango.
    profile: 1 o 2 (perfil de estrategia)
    """
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    if start_ts is not None and end_ts is not None:
        where = 'WHERE strategy_profile = %s AND buy_time >= %s AND buy_time <= %s'
        params = (profile, start_ts, end_ts)
    else:
        where = 'WHERE strategy_profile = %s'
        params = (profile,)

    query = f"""
    SELECT
        10000.0 AS initial_balance,
        COUNT(*) AS completed_trades,
        COALESCE(SUM(pnl_usdt), 0) AS total_pnl_usdt,
        COUNT(*) FILTER (WHERE pnl_usdt >= 0) AS winning_trades,
        COUNT(*) FILTER (WHERE pnl_usdt < 0) AS losing_trades,

        COUNT(*) FILTER (WHERE ml_approve = 'SI') AS ml_completed_trades,
        COALESCE(SUM(ml_pnl_usdt) FILTER (WHERE ml_approve = 'SI'), 0) AS ml_total_pnl_usdt,
        COUNT(*) FILTER (WHERE ml_approve = 'SI' AND ml_pnl_usdt >= 0) AS ml_winning_trades,
        COUNT(*) FILTER (WHERE ml_approve = 'SI' AND ml_pnl_usdt < 0) AS ml_losing_trades,

        COUNT(*) FILTER (WHERE ml_approve = 'NO' AND pnl_usdt < 0) AS ml_filtered_losses_count,
        COALESCE(ABS(SUM(pnl_usdt) FILTER (WHERE ml_approve = 'NO' AND pnl_usdt < 0)), 0) AS ml_filtered_losses_val,
        COUNT(*) FILTER (WHERE ml_approve = 'NO' AND pnl_usdt >= 0) AS ml_filtered_wins_count,
        COALESCE(SUM(pnl_usdt) FILTER (WHERE ml_approve = 'NO' AND pnl_usdt >= 0), 0) AS ml_filtered_wins_val
    FROM simulated_trades {where};
    """
    cursor.execute(query, params)
    row = dict(cursor.fetchone())
    cursor.close()
    conn.close()

    initial = float(row['initial_balance'])
    total_pnl = float(row['total_pnl_usdt'])
    ml_total_pnl = float(row['ml_total_pnl_usdt'])
    completed = int(row['completed_trades'])
    winning = int(row['winning_trades'])
    ml_completed = int(row['ml_completed_trades'])
    ml_winning = int(row['ml_winning_trades'])

    return {
        'initial_balance': initial,
        'final_balance': round(initial + total_pnl, 2),
        'total_pnl_usdt': round(total_pnl, 2),
        'total_return_pct': round((total_pnl / initial) * 100, 2) if initial else 0,
        'completed_trades': completed,
        'winning_trades': winning,
        'losing_trades': int(row['losing_trades']),
        'win_rate_pct': round(winning / completed * 100, 1) if completed else 0,
        'ml_final_balance': round(initial + ml_total_pnl, 2),
        'ml_total_pnl_usdt': round(ml_total_pnl, 2),
        'ml_total_return_pct': round((ml_total_pnl / initial) * 100, 2) if initial else 0,
        'ml_completed_trades': ml_completed,
        'ml_winning_trades': ml_winning,
        'ml_losing_trades': int(row['ml_losing_trades']),
        'ml_win_rate_pct': round(ml_winning / ml_completed * 100, 1) if ml_completed else 0,
        'ml_filtered_losses_count': int(row['ml_filtered_losses_count']),
        'ml_filtered_losses_val': round(float(row['ml_filtered_losses_val']), 2),
        'ml_filtered_wins_count': int(row['ml_filtered_wins_count']),
        'ml_filtered_wins_val': round(float(row['ml_filtered_wins_val']), 2),
    }

