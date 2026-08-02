"""
FASE 1b — Descarga de datos de Futuros de BTCUSDT desde Binance
Fuentes:
  - Funding Rate:       /fapi/v1/fundingRate        (3 años completos cada 8h)
  - Open Interest:      /futures/data/openInterestHist (últimos 30 días 15m + historial 1h)
  - Long/Short Ratio:   /futures/data/globalLongShortAccountRatio (diario 3 años)
  - Taker L/S Ratio:    /futures/data/takerlongshortRatio (diario 3 años)
Salida: data/futures/*.parquet
"""

import requests
import pandas as pd
import numpy as np
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

BASE_FUTURES = "https://fapi.binance.com"
SYMBOL       = "BTCUSDT"
FUTURES_DIR  = Path("data/futures")
FUTURES_DIR.mkdir(parents=True, exist_ok=True)

MONTHS_BACK  = 36


def fetch_json(url: str, params: dict, retries: int = 5) -> list | dict:
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                time.sleep(int(resp.headers.get("Retry-After", 15)))
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt == retries - 1:
                console.print(f"[red]Error definitivo: {e}[/red]")
                return []
            time.sleep(2 ** attempt)
    return []


# ─────────────────────────────────────────────────────────────────────
# FUNDING RATE — cada 8 horas, 3 años disponibles
# ─────────────────────────────────────────────────────────────────────
def download_funding_rate() -> pd.DataFrame:
    console.print("\n[bold cyan]Descargando Funding Rate...[/bold cyan]")
    url = f"{BASE_FUTURES}/fapi/v1/fundingRate"

    end_dt   = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=MONTHS_BACK * 30.44)

    all_rows = []
    current  = int(start_dt.timestamp() * 1000)
    end_ms   = int(end_dt.timestamp() * 1000)
    interval_ms = 8 * 60 * 60 * 1000  # 8h en ms
    limit = 1000

    while current < end_ms:
        data = fetch_json(url, {
            "symbol":    SYMBOL,
            "startTime": current,
            "endTime":   min(current + limit * interval_ms, end_ms),
            "limit":     limit,
        })
        if not data:
            break
        all_rows.extend(data)
        current = data[-1]["fundingTime"] + interval_ms
        time.sleep(0.1)

    if not all_rows:
        console.print("[yellow]Sin datos de Funding Rate[/yellow]")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)
    df = df[["timestamp", "funding_rate"]].drop_duplicates("timestamp").sort_values("timestamp")

    # Métricas derivadas
    df["funding_rate_ann"]  = df["funding_rate"] * 3 * 365          # anualizado
    df["funding_rate_z_7d"] = (
        (df["funding_rate"] - df["funding_rate"].rolling(21).mean())  # 21×8h = 7 días
        / df["funding_rate"].rolling(21).std()
    )
    df["funding_cumul_3d"]  = df["funding_rate"].rolling(9).sum()    # 9×8h = 3 días

    out_path = FUTURES_DIR / "funding_rate.parquet"
    df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
    console.print(f"  [green]✓[/green] Funding Rate: {len(df):,} registros → {out_path}")
    return df


# ─────────────────────────────────────────────────────────────────────
# OPEN INTEREST — últimos 30 días a 15m + historial diario
# ─────────────────────────────────────────────────────────────────────
def download_open_interest() -> pd.DataFrame:
    console.print("\n[bold cyan]Descargando Open Interest...[/bold cyan]")
    url = f"{BASE_FUTURES}/futures/data/openInterestHist"

    end_dt = datetime.now(timezone.utc)

    # Primero: datos 15m de los últimos 30 días (límite de Binance)
    all_rows = []

    # Diario — para historial de 3 años
    start_daily = end_dt - timedelta(days=MONTHS_BACK * 30.44)
    current     = int(start_daily.timestamp() * 1000)
    end_ms      = int(end_dt.timestamp() * 1000)
    interval_ms = 24 * 60 * 60 * 1000   # 1 día

    while current < end_ms:
        data = fetch_json(url, {
            "symbol":    SYMBOL,
            "period":    "1d",
            "startTime": current,
            "endTime":   min(current + 500 * interval_ms, end_ms),
            "limit":     500,
        })
        if not data:
            break
        all_rows.extend(data)
        last_ts = int(data[-1]["timestamp"])
        current = last_ts + interval_ms
        time.sleep(0.1)

    # 15m — últimos 30 días (alta granularidad reciente)
    start_15m = end_dt - timedelta(days=30)
    current   = int(start_15m.timestamp() * 1000)
    interval_15m = 15 * 60 * 1000
    rows_15m  = []

    while current < end_ms:
        data = fetch_json(url, {
            "symbol":    SYMBOL,
            "period":    "15m",
            "startTime": current,
            "endTime":   min(current + 500 * interval_15m, end_ms),
            "limit":     500,
        })
        if not data:
            break
        rows_15m.extend(data)
        last_ts = int(data[-1]["timestamp"])
        current = last_ts + interval_15m
        time.sleep(0.1)

    if not all_rows and not rows_15m:
        console.print("[yellow]Sin datos de Open Interest[/yellow]")
        return pd.DataFrame()

    # Construir DataFrame diario
    df_daily = pd.DataFrame(all_rows)
    if not df_daily.empty:
        df_daily["timestamp"] = pd.to_datetime(df_daily["timestamp"].astype(int), unit="ms", utc=True)
        df_daily["open_interest"] = df_daily["sumOpenInterest"].astype(float)
        df_daily = df_daily[["timestamp", "open_interest"]].drop_duplicates("timestamp")

    # Construir DataFrame 15m
    df_15m = pd.DataFrame(rows_15m)
    if not df_15m.empty:
        df_15m["timestamp"] = pd.to_datetime(df_15m["timestamp"].astype(int), unit="ms", utc=True)
        df_15m["open_interest"] = df_15m["sumOpenInterest"].astype(float)
        df_15m = df_15m[["timestamp", "open_interest"]].drop_duplicates("timestamp")

    # Combinar (15m tiene prioridad para últimos 30 días)
    if not df_daily.empty and not df_15m.empty:
        cutoff = df_15m["timestamp"].min()
        df_daily_old = df_daily[df_daily["timestamp"] < cutoff]
        df = pd.concat([df_daily_old, df_15m], ignore_index=True)
    elif not df_15m.empty:
        df = df_15m
    else:
        df = df_daily

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Métricas derivadas
    df["oi_change_pct"]  = df["open_interest"].pct_change() * 100
    df["oi_sma_7d"]      = df["open_interest"].rolling(7).mean()
    df["oi_z_7d"]        = (
        (df["open_interest"] - df["open_interest"].rolling(30).mean())
        / df["open_interest"].rolling(30).std()
    )

    out_path = FUTURES_DIR / "open_interest.parquet"
    df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
    console.print(f"  [green]✓[/green] Open Interest: {len(df):,} registros → {out_path}")
    console.print(f"    [dim](Diario: 3 años | 15m: últimos 30 días)[/dim]")
    return df


# ─────────────────────────────────────────────────────────────────────
# LONG/SHORT RATIO — diario 3 años
# ─────────────────────────────────────────────────────────────────────
def download_ls_ratio() -> pd.DataFrame:
    console.print("\n[bold cyan]Descargando Long/Short Ratio...[/bold cyan]")
    url = f"{BASE_FUTURES}/futures/data/globalLongShortAccountRatio"

    end_dt   = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=MONTHS_BACK * 30.44)
    current  = int(start_dt.timestamp() * 1000)
    end_ms   = int(end_dt.timestamp() * 1000)
    interval_ms = 24 * 60 * 60 * 1000

    all_rows = []
    while current < end_ms:
        data = fetch_json(url, {
            "symbol":    SYMBOL,
            "period":    "1d",
            "startTime": current,
            "endTime":   min(current + 500 * interval_ms, end_ms),
            "limit":     500,
        })
        if not data:
            break
        all_rows.extend(data)
        last_ts = int(data[-1]["timestamp"])
        current = last_ts + interval_ms
        time.sleep(0.1)

    if not all_rows:
        console.print("[yellow]Sin datos de L/S Ratio[/yellow]")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["timestamp"]      = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
    df["ls_ratio"]       = df["longShortRatio"].astype(float)
    df["long_pct"]       = df["longAccount"].astype(float)
    df["short_pct"]      = df["shortAccount"].astype(float)
    df = df[["timestamp", "ls_ratio", "long_pct", "short_pct"]].drop_duplicates("timestamp").sort_values("timestamp")

    df["ls_ratio_z_30d"]   = (
        (df["ls_ratio"] - df["ls_ratio"].rolling(30).mean())
        / df["ls_ratio"].rolling(30).std()
    )
    df["ls_extreme_long"]  = df["long_pct"] > 0.65   # >65% longs = potencial reversal
    df["ls_extreme_short"] = df["short_pct"] > 0.65

    out_path = FUTURES_DIR / "ls_ratio.parquet"
    df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
    console.print(f"  [green]✓[/green] L/S Ratio: {len(df):,} registros → {out_path}")
    return df


# ─────────────────────────────────────────────────────────────────────
# TAKER BUY/SELL RATIO — diario 3 años
# ─────────────────────────────────────────────────────────────────────
def download_taker_ratio() -> pd.DataFrame:
    console.print("\n[bold cyan]Descargando Taker L/S Ratio...[/bold cyan]")
    url = f"{BASE_FUTURES}/futures/data/takerlongshortRatio"

    end_dt   = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=MONTHS_BACK * 30.44)
    current  = int(start_dt.timestamp() * 1000)
    end_ms   = int(end_dt.timestamp() * 1000)
    interval_ms = 24 * 60 * 60 * 1000

    all_rows = []
    while current < end_ms:
        data = fetch_json(url, {
            "symbol":    SYMBOL,
            "period":    "1d",
            "startTime": current,
            "endTime":   min(current + 500 * interval_ms, end_ms),
            "limit":     500,
        })
        if not data:
            break
        all_rows.extend(data)
        last_ts = int(data[-1]["timestamp"])
        current = last_ts + interval_ms
        time.sleep(0.1)

    if not all_rows:
        console.print("[yellow]Sin datos de Taker Ratio[/yellow]")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["timestamp"]       = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
    df["taker_ls_ratio"]  = df["buySellRatio"].astype(float)
    df["taker_buy_vol_f"] = df["buyVol"].astype(float)
    df["taker_sell_vol_f"]= df["sellVol"].astype(float)
    df = df[["timestamp", "taker_ls_ratio", "taker_buy_vol_f", "taker_sell_vol_f"]].drop_duplicates("timestamp").sort_values("timestamp")

    out_path = FUTURES_DIR / "taker_ratio.parquet"
    df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
    console.print(f"  [green]✓[/green] Taker Ratio: {len(df):,} registros → {out_path}")
    return df


# ─────────────────────────────────────────────────────────────────────
def main():
    console.print(Panel(
        "[bold]Descargando datos de Futuros de Binance[/bold]\n"
        "Funding Rate · Open Interest · Long/Short Ratio · Taker Ratio",
        title="FASE 1b — Datos de Futuros",
        border_style="magenta"
    ))

    results = {}
    results["funding"]  = download_funding_rate()
    results["oi"]       = download_open_interest()
    results["ls"]       = download_ls_ratio()
    results["taker"]    = download_taker_ratio()

    # ── Resumen ───────────────────────────────────────────────────────
    table = Table(title="Resumen Datos de Futuros", border_style="magenta")
    table.add_column("Fuente",       style="cyan")
    table.add_column("Registros",    style="white")
    table.add_column("Granularidad", style="dim")
    table.add_column("Estado",       style="white")

    specs = {
        "funding":  ("Funding Rate",       "8 horas"),
        "oi":       ("Open Interest",      "Diario/15m"),
        "ls":       ("Long/Short Ratio",   "Diario"),
        "taker":    ("Taker L/S Ratio",    "Diario"),
    }
    for key, (name, gran) in specs.items():
        df = results[key]
        n   = f"{len(df):,}" if not df.empty else "0"
        ok  = "[green]✓[/green]" if not df.empty else "[red]✗[/red]"
        table.add_row(name, n, gran, ok)

    console.print(table)
    console.print("\n[bold green]✓ Datos de Futuros descargados correctamente[/bold green]")


if __name__ == "__main__":
    main()
