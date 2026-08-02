"""
FASE 2 — Ingeniería de Features para Fortinet (FTNT)
Carga el OHLCV raw de 15m y calcula ~100 indicadores técnicos de alta calidad.
Salida: data/master/master_dataset.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

console = Console()

RAW_PATH    = Path("data/raw/spot/btcusdt_15m_raw.parquet")
MASTER_DIR  = Path("data/master")
MASTER_DIR.mkdir(parents=True, exist_ok=True)
MASTER_PATH = MASTER_DIR / "master_dataset.parquet"


def calc_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula SMAs, EMAs, pendientes, aceleraciones y cruces."""
    close = df["close"]
    
    for p in [5, 9, 20, 50, 100, 200]:
        col = f"sma_{p}"
        df[col]               = close.rolling(window=p, min_periods=1).mean()
        df[f"{col}_slope"]    = df[col].diff(3)
        df[f"{col}_dist_pct"] = (close - df[col]) / df[col] * 100

    for p in [9, 12, 20, 21, 26, 50, 100, 200]:
        col = f"ema_{p}"
        df[col]               = close.ewm(span=p, adjust=False).mean()
        df[f"{col}_slope"]    = df[col].diff(3)
        df[f"{col}_accel"]    = df[col].diff(3).diff(3)
        df[f"{col}_dist_pct"] = (close - df[col]) / df[col] * 100

    df["ema9aboveema20"]   = df["ema_9"] > df["ema_20"]
    df["ema20aboveema50"]  = df["ema_20"] > df["ema_50"]
    df["ema50aboveema200"] = df["ema_50"] > df["ema_200"]
    
    df["all_emas_aligned"] = (
        (df["ema_9"] > df["ema_20"]) &
        (df["ema_20"] > df["ema_50"]) &
        (df["ema_50"] > df["ema_200"])
    )
    return df


def calc_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula RSI, MACD, Stochastic y Momentum."""
    close = df["close"]
    
    # RSI (7, 14, 21)
    for p in [7, 14, 21]:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0)
        avg_gain = gain.ewm(com=p-1, adjust=False).mean()
        avg_loss = loss.ewm(com=p-1, adjust=False).mean()
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        df[f"rsi_{p}"] = np.where(avg_loss == 0, 100.0, 100.0 - (100.0 / (1.0 + rs)))
        df[f"rsi_{p}_slope"] = df[f"rsi_{p}"].diff(3)

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd_line"]       = ema12 - ema26
    df["macd_signal"]     = df["macd_line"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]       = df["macd_line"] - df["macd_signal"]
    df["macd_hist_slope"] = df["macd_hist"].diff(3)

    # ROC
    for p in [10, 20]:
        df[f"roc_{p}"] = (close - close.shift(p)) / close.shift(p) * 100

    return df


def calc_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula ATR, Bandas de Bollinger, Donchian y Volatilidad Histórica."""
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # ATR (7, 14, 21)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    for p in [7, 14, 21]:
        df[f"atr_{p}"]     = tr.ewm(span=p, adjust=False).mean()
        df[f"atr_{p}_pct"] = df[f"atr_{p}"] / close * 100

    df["atr_14_z_50"] = (
        (df["atr_14"] - df["atr_14"].rolling(50, min_periods=1).mean()) /
        (df["atr_14"].rolling(50, min_periods=1).std() + 1e-10)
    )

    # Bollinger Bands (20, 2.0)
    bb_mid = close.rolling(20, min_periods=1).mean()
    bb_std = close.rolling(20, min_periods=1).std()
    df["bb_mid"]   = bb_mid
    df["bb_upper"] = bb_mid + (bb_std * 2.0)
    df["bb_lower"] = bb_mid - (bb_std * 2.0)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (bb_mid + 1e-10) * 100
    df["bb_pct"]   = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)
    df["bb_squeeze"] = df["bb_width"] < df["bb_width"].rolling(50, min_periods=1).quantile(0.20)

    # Donchian Channels (20, 55)
    for p in [20, 55]:
        df[f"dc_{p}_lower"] = low.rolling(p, min_periods=1).min()
        df[f"dc_{p}_upper"] = high.rolling(p, min_periods=1).max()
        df[f"dc_{p}_width"] = (df[f"dc_{p}_upper"] - df[f"dc_{p}_lower"]) / close * 100

    log_ret = np.log(close / close.shift(1))
    for p in [10, 20, 50]:
        df[f"hv_{p}"] = log_ret.rolling(p, min_periods=1).std() * np.sqrt(p) * 100

    return df


def calc_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula indicadores de Volumen y VWAP."""
    close = df["close"]
    volume = df["volume"]
    
    # OBV
    obv_change = np.where(close > close.shift(1), volume, np.where(close < close.shift(1), -volume, 0))
    df["obv"]          = pd.Series(obv_change, index=df.index).cumsum()
    df["obv_ema_20"]   = df["obv"].ewm(span=20, adjust=False).mean()
    df["obv_slope"]    = df["obv"].diff(5)
    df["obv_above_ema"] = df["obv"] > df["obv_ema_20"]

    df["vol_sma_20"] = volume.rolling(20, min_periods=1).mean()
    df["vol_ratio"]  = volume / (df["vol_sma_20"] + 1e-10)
    df["vol_z_20"]   = (volume - df["vol_sma_20"]) / (volume.rolling(20, min_periods=1).std() + 1e-10)
    df["vol_surge"]  = df["vol_ratio"] > 2.0

    return df


def main():
    console.print(Panel("[bold cyan]FASE 2 — Feature Engineering (FTNT)[/bold cyan]"))

    if not RAW_PATH.exists():
        raise FileNotFoundError(f"No existe el archivo raw: {RAW_PATH}")

    df = pd.read_parquet(RAW_PATH)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    console.print(f"Cargadas [green]{len(df):,}[/green] velas desde {RAW_PATH}")

    df = calc_moving_averages(df)
    df = calc_momentum(df)
    df = calc_volatility(df)
    df = calc_volume_indicators(df)

    df.dropna(subset=["ema_200"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_parquet(MASTER_PATH, index=False)
    console.print(f"[bold green]✓ Master Dataset guardado en {MASTER_PATH} ({len(df):,} filas, {len(df.columns)} columnas)[/bold green]")


if __name__ == "__main__":
    main()
