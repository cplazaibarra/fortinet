"""
Script para descargar 3 años de velas de 15 minutos de Fortinet (FTNT) desde Alpaca,
calcular los 17 indicadores técnicos (EMAs, Slopes %, RSI, MACD, ATR) e insertarlos
en la base de datos PostgreSQL.
"""

import os
import sys
import time
import requests
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

console = Console()

# 1. Cargar variables de entorno
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

DB_HOST = os.getenv("POSTGRES_HOST") or os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("POSTGRES_PORT") or os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("POSTGRES_DB") or os.getenv("DB_NAME", "fortinet_db")
DB_USER = os.getenv("POSTGRES_USER") or os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("POSTGRES_PASSWORD") or os.getenv("DB_PASSWORD", "postgres")

SYMBOL = "FTNT"
TIMEFRAME = "15Min"
YEARS_BACK = 3

if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    console.print("[bold red]ERROR: No se encontraron las claves ALPACA_API_KEY / ALPACA_SECRET_KEY en el archivo .env[/bold red]")
    sys.exit(1)


def get_alpaca_bars(symbol: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """Descarga velas históricas desde Alpaca Market Data API v2 con paginación."""
    url = f"https://data.alpaca.markets/v2/stocks/bars"
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        "accept": "application/json"
    }
    
    params = {
        "symbols": symbol,
        "timeframe": TIMEFRAME,
        "start": start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": 10000,
        "adjustment": "split",
        "feed": "iex"  # feed accesible para cuentas Paper / gratis
    }
    
    all_bars = []
    page_token = None
    
    console.print(f"[cyan]Solicitando velas de {symbol} desde {params['start']} hasta {params['end']}...[/cyan]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task(f"Descargando velas de Alpaca...", total=None)
        
        while True:
            if page_token:
                params["page_token"] = page_token
            
            for attempt in range(5):
                try:
                    resp = requests.get(url, headers=headers, params=params, timeout=30)
                    if resp.status_code == 429:
                        time.sleep(2)
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except Exception as e:
                    if attempt == 4:
                        raise e
                    time.sleep(2 ** attempt)
            
            bars = data.get("bars", {}).get(symbol, [])
            if not bars:
                break
            
            all_bars.extend(bars)
            progress.update(task, description=f"Descargadas {len(all_bars):,} velas...")
            
            page_token = data.get("next_page_token")
            if not page_token:
                break
    
    if not all_bars:
        console.print("[yellow]No se obtuvieron velas. Probando con feed 'sip'...[/yellow]")
        params["feed"] = "sip"
        params.pop("page_token", None)
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            all_bars = data.get("bars", {}).get(symbol, [])

    if not all_bars:
        raise RuntimeError("No se pudieron obtener velas desde Alpaca.")

    df = pd.DataFrame(all_bars)
    # Renombrar columnas devueltas por Alpaca (t: timestamp, o: open, h: high, l: low, c: close, v: volume, n: trade_count, vw: vwap)
    df.rename(columns={
        't': 'timestamp',
        'o': 'open',
        'h': 'high',
        'l': 'low',
        'c': 'close',
        'v': 'volume',
        'n': 'trade_count',
        'vw': 'vwap'
    }, inplace=True)

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['symbol'] = SYMBOL
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula las 17 columnas de indicadores técnicos sobre el DataFrame."""
    console.print("[cyan]Calculando indicadores técnicos (EMAs, Slopes %, RSI, MACD, ATR)...[/cyan]")
    
    close = df['close']
    
    # 1. Medias Móviles Exponenciales (EMAs) y Pendientes (%)
    ema_periods = [9, 21, 35, 50, 100, 200]
    for p in ema_periods:
        ema_col = f"ema_{p}"
        slope_col = f"slope_ema{p}_pct"
        
        df[ema_col] = close.ewm(span=p, adjust=False).mean()
        prev_ema = df[ema_col].shift(1)
        df[slope_col] = np.where(prev_ema.notna() & (prev_ema != 0), ((df[ema_col] - prev_ema) / prev_ema) * 100.0, 0.0)

    # 2. RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    df['rsi_14'] = np.where(avg_loss == 0, 100.0, 100.0 - (100.0 / (1.0 + rs)))

    # 3. MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # 4. ATR (14)
    high_low = df['high'] - df['low']
    high_cp = (df['high'] - close.shift(1)).abs()
    low_cp = (df['low'] - close.shift(1)).abs()
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    df['atr_14'] = tr.ewm(span=14, adjust=False).mean()

    # Reemplazar valores NaN / Inf por None (para NULL en PostgreSQL)
    df = df.replace({np.nan: None, np.inf: None, -np.inf: None})
    return df


def insert_to_postgres(df: pd.DataFrame):
    """Inserta las filas procesadas en PostgreSQL mediante lotes optimizados."""
    console.print(f"[cyan]Conectando a PostgreSQL en {DB_HOST}:{DB_PORT}/{DB_NAME}...[/cyan]")
    
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )
    cursor = conn.cursor()

    columns = [
        "timestamp", "symbol", "open", "high", "low", "close", "volume", "trade_count", "vwap",
        "ema_9", "slope_ema9_pct", "ema_21", "slope_ema21_pct", "ema_35", "slope_ema35_pct",
        "ema_50", "slope_ema50_pct", "ema_100", "slope_ema100_pct", "ema_200", "slope_ema200_pct",
        "rsi_14", "macd", "macd_signal", "macd_hist", "atr_14"
    ]

    tuples = [tuple(x) for x in df[columns].to_numpy()]

    insert_query = f"""
    INSERT INTO candles_15m ({', '.join(columns)})
    VALUES %s
    ON CONFLICT (symbol, timestamp) DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume,
        trade_count = EXCLUDED.trade_count,
        vwap = EXCLUDED.vwap,
        ema_9 = EXCLUDED.ema_9,
        slope_ema9_pct = EXCLUDED.slope_ema9_pct,
        ema_21 = EXCLUDED.ema_21,
        slope_ema21_pct = EXCLUDED.slope_ema21_pct,
        ema_35 = EXCLUDED.ema_35,
        slope_ema35_pct = EXCLUDED.slope_ema35_pct,
        ema_50 = EXCLUDED.ema_50,
        slope_ema50_pct = EXCLUDED.slope_ema50_pct,
        ema_100 = EXCLUDED.ema_100,
        slope_ema100_pct = EXCLUDED.slope_ema100_pct,
        ema_200 = EXCLUDED.ema_200,
        slope_ema200_pct = EXCLUDED.slope_ema200_pct,
        rsi_14 = EXCLUDED.rsi_14,
        macd = EXCLUDED.macd,
        macd_signal = EXCLUDED.macd_signal,
        macd_hist = EXCLUDED.macd_hist,
        atr_14 = EXCLUDED.atr_14;
    """

    console.print(f"[green]Insertando {len(tuples):,} registros en PostgreSQL...[/green]")
    execute_values(cursor, insert_query, tuples, page_size=2000)
    conn.commit()
    cursor.close()
    conn.close()
    console.print("[bold green]¡Inserción completada exitosamente en PostgreSQL![/bold green]")


def main():
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=YEARS_BACK * 365)
    
    console.print(Panel(
        f"[bold white]Carga de Velas 15m - Fortinet (FTNT)[/bold white]\n"
        f"Desde: [yellow]{start_dt.strftime('%Y-%m-%d')}[/yellow] Hasta: [yellow]{end_dt.strftime('%Y-%m-%d')}[/yellow]\n"
        f"Base de Datos: [green]{DB_NAME}[/green] ({DB_HOST}:{DB_PORT})",
        title="Robot FTNT - Alpaca Data Sync",
        border_style="blue"
    ))

    # 1. Obtener velas
    df = get_alpaca_bars(SYMBOL, start_dt, end_dt)
    console.print(f"[bold green]Se obtuvieron {len(df):,} velas históricas.[/bold green]")

    # 2. Calcular indicadores
    df = calculate_technical_indicators(df)

    # 3. Guardar en PostgreSQL
    insert_to_postgres(df)


if __name__ == "__main__":
    main()
