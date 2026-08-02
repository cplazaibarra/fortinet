"""
FASE 3 — Variables de Resultado Futuro (Forward-Looking)

Calcula lo que ocurrió DESPUÉS de cada vela (para evaluar señales y entrenar ML).
Estas columnas NUNCA se usan como input en el punto de decisión.

Metodología de entrada:
  - Señal al CIERRE de la vela t
  - ENTRADA al OPEN de la vela t+1
  - Por lo tanto: ret = (precio_salida - open[t+1]) / open[t+1]

COSTOS DE TRANSACCIÓN BINANCE (incorporados en TODOS los retornos):
  Compra:         0.10%
  Venta:          0.10%
  Total (round trip): 0.20%  ← SE DESCUENTA DE CADA TRADE

Salida: data/master/master_dataset.parquet (actualizado con columnas fwd_*)
"""

import pandas as pd
import numpy as np
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

MASTER_PATH = Path("data/master/master_dataset.parquet")

# ─────────────────────────────────────────────────────────────────────
# COSTOS DE TRANSACCIÓN — Binance
# ─────────────────────────────────────────────────────────────────────
FEE_ENTRY   = 0.001   # 0.10% compra
FEE_EXIT    = 0.001   # 0.10% venta
FEE_TOTAL   = FEE_ENTRY + FEE_EXIT   # 0.20% round trip (deducido de cada retorno)


def calc_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retornos futuros netos de comisiones.

    Nomenclatura: fwd_ret_{N}c = retorno neto en las próximas N velas de 15m.
    El retorno bruto = (close[t+N] - open[t+1]) / open[t+1]
    El retorno neto  = retorno_bruto - FEE_TOTAL (0.20%)
    """
    close  = df["close"].values
    open_  = df["open"].values
    high   = df["high"].values
    low    = df["low"].values
    n      = len(df)

    # Precio de entrada para cada vela t = open[t+1]
    entry_price = np.full(n, np.nan)
    entry_price[:-1] = open_[1:]   # open de la vela siguiente

    def net_ret(horizon_candles: int) -> np.ndarray:
        """Retorno neto sobre 'horizon_candles' velas desde la entrada."""
        arr = np.full(n, np.nan)
        lim = n - horizon_candles - 1
        if lim <= 0:
            return arr
        exit_close = close[horizon_candles + 1 : n]
        entry      = entry_price[:lim]
        gross      = (exit_close - entry) / entry
        arr[:lim]  = gross - FEE_TOTAL
        return arr

    # Horizontes: 1 vela (15m), 4 (1h), 8 (2h), 16 (4h), 32 (8h), 96 (1día)
    horizons = {
        "fwd_ret_1c":  1,
        "fwd_ret_4c":  4,
        "fwd_ret_8c":  8,
        "fwd_ret_16c": 16,
        "fwd_ret_32c": 32,
        "fwd_ret_96c": 96,
    }
    for col, h in horizons.items():
        df[col] = net_ret(h)

    df["entry_price"] = entry_price
    return df


def calc_max_excursions(df: pd.DataFrame) -> pd.DataFrame:
    """
    MFE (Maximum Favorable Excursion) y MAE (Maximum Adverse Excursion)
    para horizontes de 1h (4 velas) y 4h (16 velas).

    MFE: máximo avance alcanzado en la ventana → oportunidad potencial
    MAE: máximo retroceso sufrido            → riesgo real del trade

    Ambos calculados desde el precio de entrada (open[t+1]).
    """
    high_  = df["high"].values
    low_   = df["low"].values
    entry  = df["entry_price"].values
    n      = len(df)

    for label, h in [("1h", 4), ("4h", 16)]:
        mfe = np.full(n, np.nan)
        mae = np.full(n, np.nan)

        for i in range(n - h - 1):
            e = entry[i]
            if np.isnan(e) or e == 0:
                continue
            window_high = high_[i+1 : i+h+1].max()
            window_low  = low_[i+1  : i+h+1].min()
            mfe[i] = (window_high - e) / e * 100   # %
            mae[i] = (window_low  - e) / e * 100   # % (negativo si baja)

        df[f"mfe_{label}"]     = mfe
        df[f"mae_{label}"]     = mae
        df[f"risk_reward_{label}"] = np.where(
            mae != 0, np.abs(mfe / np.clip(mae, None, -0.0001)), np.nan
        )

    return df


def calc_tp_sl_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simula salidas por TP/SL en múltiples configuraciones.
    Resultado por configuración:
       1  = TP alcanzado primero  (win)
      -1  = SL alcanzado primero  (loss)
       0  = tiempo agotado        (breakeven por fees)

    Configuraciones (TP% / SL% en relación al precio de entrada):
      A) TP=1.0% SL=0.5%  Ratio 2:1
      B) TP=1.5% SL=0.5%  Ratio 3:1
      C) TP=2.0% SL=1.0%  Ratio 2:1
      D) TP=3.0% SL=1.0%  Ratio 3:1

    Ventana máxima: 32 velas (8 horas). Después → timeout.

    COSTOS:  Se descuenta 0.20% del retorno neto en caso de TP o timeout.
             En SL, la pérdida bruta ya incluye las comisiones sumadas.
    """
    high_  = df["high"].values
    low_   = df["low"].values
    entry  = df["entry_price"].values
    n      = len(df)
    MAX_H  = 32   # 8 horas máximo

    configs = {
        "tp100_sl050": (1.00 / 100, 0.50 / 100),
        "tp150_sl050": (1.50 / 100, 0.50 / 100),
        "tp200_sl100": (2.00 / 100, 1.00 / 100),
        "tp300_sl100": (3.00 / 100, 1.00 / 100),
    }

    for cfg, (tp_pct, sl_pct) in configs.items():
        results  = np.zeros(n, dtype=np.int8)    # -1, 0, 1
        net_rets = np.full(n, np.nan)

        for i in range(n - MAX_H - 1):
            e = entry[i]
            if np.isnan(e) or e == 0:
                results[i]  = 0
                net_rets[i] = np.nan
                continue

            tp_price = e * (1 + tp_pct)
            sl_price = e * (1 - sl_pct)
            outcome  = 0        # default: timeout
            net      = -FEE_TOTAL  # al menos pagamos comisiones

            for k in range(1, MAX_H + 1):
                idx = i + k
                if idx >= n:
                    break
                h_k = high_[idx]
                l_k = low_[idx]

                # Verificar TP primero (favorable para el comprador)
                if h_k >= tp_price:
                    outcome = 1
                    net = tp_pct - FEE_TOTAL   # ganancia neta
                    break
                if l_k <= sl_price:
                    outcome = -1
                    net = -sl_pct - FEE_TOTAL  # pérdida neta (comisiones ya incluidas)
                    break

            if outcome == 0:
                # Timeout: cierra al precio actual
                close_val = df["close"].values[i + min(MAX_H, n - i - 1)]
                gross = (close_val - e) / e
                net = gross - FEE_TOTAL

            results[i]  = outcome
            net_rets[i] = net

        df[f"tpsl_{cfg}_result"]  = results
        df[f"tpsl_{cfg}_net_ret"] = net_rets
        df[f"tpsl_{cfg}_win"]     = results == 1

    return df


def calc_binary_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Targets binarios para Machine Learning.
    Siempre considerando costos de 0.20% (entry + exit).
    """
    # ¿El precio sube al menos 0.5% neto después de costos en 1h?
    # (necesita subir ≥ 0.70% bruto para ganar 0.50% neto)
    df["target_profitable_1h"]  = df["fwd_ret_4c"]  > 0          # cualquier ganancia neta
    df["target_profitable_4h"]  = df["fwd_ret_16c"] > 0
    df["target_05pct_net_1h"]   = df["fwd_ret_4c"]  >= 0.005     # +0.5% neto
    df["target_10pct_net_4h"]   = df["fwd_ret_16c"] >= 0.010     # +1.0% neto
    df["target_tp_sl_2to1"]     = df["tpsl_tp100_sl050_win"]      # TP 2:1 alcanzado

    return df


def main():
    console.print(Panel(
        "[bold]Calculando variables de resultado futuro[/bold]\n\n"
        f"[yellow]COSTOS DE TRANSACCIÓN (Binance):[/yellow]\n"
        f"  Compra:  [red]{FEE_ENTRY*100:.2f}%[/red]\n"
        f"  Venta:   [red]{FEE_EXIT*100:.2f}%[/red]\n"
        f"  [bold red]Total round-trip: {FEE_TOTAL*100:.2f}%[/bold red]\n\n"
        "Todos los retornos reportados son [bold]NETOS de comisiones[/bold].\n"
        "Una estrategia necesita generar >[bold]0.20%[/bold] por trade para ser rentable.",
        title="FASE 3 — Variables Forward-Looking",
        border_style="yellow"
    ))

    console.print(f"\nCargando {MASTER_PATH}...")
    df = pd.read_parquet(MASTER_PATH)
    console.print(f"  {len(df):,} velas | {len(df.columns):,} columnas existentes")

    console.print("\n[1/4] Calculando retornos futuros (netos de 0.20% comisiones)...")
    df = calc_forward_returns(df)

    console.print("[2/4] Calculando MFE y MAE (1h y 4h)...")
    df = calc_max_excursions(df)

    console.print("[3/4] Simulando TP/SL (4 configuraciones, ventana 8h)...")
    df = calc_tp_sl_targets(df)

    console.print("[4/4] Generando targets binarios para ML...")
    df = calc_binary_targets(df)

    # ── Guardar ───────────────────────────────────────────────────────
    df.to_parquet(MASTER_PATH, index=False, engine="pyarrow", compression="snappy")

    # ── Estadísticas de los targets ───────────────────────────────────
    table = Table(title="Distribución de Targets (% positivos)", border_style="yellow")
    table.add_column("Variable",           style="cyan")
    table.add_column("% Positivos",        style="white")
    table.add_column("Media retorno",      style="white")
    table.add_column("Descripción",        style="dim")

    target_info = [
        ("fwd_ret_4c",            "Retorno neto en 1h (continuo)"),
        ("fwd_ret_16c",           "Retorno neto en 4h (continuo)"),
        ("target_profitable_1h",  "Positivo neto en 1h"),
        ("target_profitable_4h",  "Positivo neto en 4h"),
        ("target_05pct_net_1h",   "+0.5% neto en 1h"),
        ("target_10pct_net_4h",   "+1.0% neto en 4h"),
        ("target_tp_sl_2to1",     "TP 2:1 alcanzado (1% TP / 0.5% SL)"),
    ]

    for col, desc in target_info:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.dtype == bool or (series.isin([0, 1]).all()):
            pct = f"{series.mean()*100:.1f}%"
        else:
            pct = f"—"
        mean_r = f"{series.mean()*100:.3f}%" if col.startswith("fwd_ret") else "—"
        table.add_row(col, pct, mean_r, desc)

    console.print(table)

    # TP/SL configs summary
    table2 = Table(title="Resultados TP/SL por Configuración", border_style="yellow")
    table2.add_column("Configuración", style="cyan")
    table2.add_column("Win Rate",      style="green")
    table2.add_column("Ret medio Win", style="green")
    table2.add_column("Ret medio Loss",style="red")
    table2.add_column("Expectancy",    style="white")

    for cfg in ["tp100_sl050", "tp150_sl050", "tp200_sl100", "tp300_sl100"]:
        res = df[f"tpsl_{cfg}_result"].dropna()
        net = df[f"tpsl_{cfg}_net_ret"].dropna()
        wins   = net[res == 1]
        losses = net[res == -1]
        wr     = (res == 1).mean()
        exp    = net.mean()
        table2.add_row(
            cfg,
            f"{wr*100:.1f}%",
            f"{wins.mean()*100:.2f}%" if len(wins) > 0 else "—",
            f"{losses.mean()*100:.2f}%" if len(losses) > 0 else "—",
            f"{exp*100:.3f}%",
        )
    console.print(table2)

    n_new = len([c for c in df.columns if c.startswith("fwd_") or
                 c.startswith("tpsl_") or c.startswith("target_") or
                 c.startswith("mfe_") or c.startswith("mae_")])
    console.print(f"\n[bold green]✓ {n_new} variables forward-looking añadidas[/bold green]")
    console.print(f"  Total columnas en Master Dataset: [cyan]{len(df.columns)}[/cyan]")


if __name__ == "__main__":
    main()
