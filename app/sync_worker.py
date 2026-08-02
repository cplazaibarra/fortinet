import os
import time
from datetime import datetime
import database
from alpaca_client import AlpacaClient
import indicators

SYMBOL = "FTNT"

def sync_recent_and_gaps(client):
    """
    Verifica la última vela guardada de Fortinet en PostgreSQL.
    Si la app estuvo apagada o requiere actualización en vivo, descarga el bloque de velas
    de FTNT desde Alpaca e inserta en PostgreSQL.
    """
    try:
        now_ms = int(time.time() * 1000)
        last_ts = database.get_last_candle_time_15m()
        
        if last_ts is None:
            return

        # Consultar las velas más recientes de FTNT desde Alpaca
        start_fetch = max(0, last_ts - (30 * 60 * 1000))
        klines_raw = client.get_klines(symbol=SYMBOL, interval="15m", limit=1000, startTime=start_fetch)
        if not klines_raw:
            return

        # Cargar hasta 300 velas históricas previas de PostgreSQL para contexto de indicadores
        prior_candles = database.get_candles_15m(limit=300)

        combined_dict = {}
        for c in prior_candles:
            combined_dict[c['time']] = c
        for c in klines_raw:
            combined_dict[c['time']] = c

        combined_list = sorted(combined_dict.values(), key=lambda x: x['time'])

        # Recalcular indicadores (EMAs, RSI, MACD, ATR)
        klines_with_indicators = indicators.add_indicators_to_klines(combined_list)

        # Guardar todas las velas procesadas en PostgreSQL (candles_15m y ohlcv_15m)
        database.upsert_candles_15m(klines_with_indicators)
        database.add_log(f"Sincronizadas {len(klines_raw)} velas recientes de Fortinet (FTNT) desde Alpaca", "INFO")

    except Exception as e:
        database.add_log(f"Error en sync_recent_and_gaps para FTNT: {e}", "ERROR")

def sync_historical_2years_chunk(client):
    """
    Descarga retroactivamente velas de 15m de Fortinet desde Alpaca hasta completar el rango.
    """
    try:
        now_ms = int(time.time() * 1000)
        three_years_ms = 3 * 365 * 24 * 60 * 60 * 1000
        target_start_ts = now_ms - three_years_ms

        oldest_ts = database.get_oldest_candle_time_15m()

        if oldest_ts is None:
            klines_raw = client.get_klines(symbol=SYMBOL, interval="15m", limit=1000)
        elif oldest_ts > target_start_ts:
            klines_raw = client.get_klines(symbol=SYMBOL, interval="15m", limit=1000, endTime=oldest_ts - 1)
        else:
            return True

        if not klines_raw:
            return True

        klines_with_indicators = indicators.add_indicators_to_klines(klines_raw)
        database.upsert_candles_15m(klines_with_indicators)
        return False

    except Exception as e:
        database.add_log(f"Error en descarga histórica de FTNT desde Alpaca: {e}", "ERROR")
        return False

def run_sync_worker():
    """
    Hilo de fondo autónomo que ejecuta la sincronización constante de Fortinet (FTNT) desde Alpaca.
    """
    database.add_log("Servicio autónomo de sincronización de Fortinet (FTNT) desde Alpaca iniciado en PostgreSQL", "INFO")
    
    settings = database.get_settings()
    api_key = settings.get('alpaca_api_key') or os.getenv("ALPACA_API_KEY", "")
    api_secret = settings.get('alpaca_secret_key') or os.getenv("ALPACA_SECRET_KEY", "")
    
    client = AlpacaClient(api_key=api_key, api_secret=api_secret, is_paper=True)
    
    sync_recent_and_gaps(client)
    
    historical_complete = False
    
    while True:
        try:
            if not historical_complete:
                historical_complete = sync_historical_2years_chunk(client)
            
            sync_recent_and_gaps(client)
            time.sleep(60) # Sincronizar cada minuto
        except Exception as e:
            database.add_log(f"Error en bucle del worker FTNT: {e}", "ERROR")
            time.sleep(10)
