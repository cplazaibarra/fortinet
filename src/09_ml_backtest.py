"""
src/09_ml_backtest.py
======================
Simula el backtest de la estrategia base (M1) con y sin el filtro de Machine Learning (Meta-Labeling).
Evalúa los resultados únicamente en el set de Test (Out-of-Sample: Jul 2025 - Jun 2026).
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

MASTER_PATH = Path("data/master/master_dataset.parquet")
MODEL_DIR   = Path("data/models")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CAPITAL_INICIAL = 10_000.0
COOLDOWN_BARS   = 32

FEE_MAKER       = 0.0002   # 0.02%
FEE_TOTAL_MAKER = FEE_MAKER * 2   # 0.04% round-trip

# Configuración TP/SL de la estrategia M1
TP_FRAC  = 0.015    # TP 1.5%
SL_FRAC  = 0.005    # SL 0.5%
FEE_RT   = FEE_TOTAL_MAKER


def build_base_signal(df: pd.DataFrame) -> np.ndarray:
    """Reconstruye la señal base (M1)."""
    adx_very_strong = df["adx_14"] > 30 if "adx_14" in df.columns else pd.Series(False, index=df.index)
    rsi = df["rsi_14"] if "rsi_14" in df.columns else pd.Series(50, index=df.index)
    rsi_zone = (rsi >= 40) & (rsi <= 65)
    close = df["close"]
    sma50 = df["sma_50"] if "sma_50" in df.columns else close
    above_sma = close > sma50
    return (adx_very_strong & rsi_zone & above_sma).fillna(False).astype(bool).values


def simulate_tp_sl(
    entry_idx: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    open_arr: np.ndarray,
    tp_frac: float,
    sl_frac: float,
    fee_rt: float,
    window: int = 32,
) -> np.ndarray:
    """Simula salidas por TP/SL."""
    n_entries = len(entry_idx)
    n_bars    = len(high_arr)
    rets      = np.full(n_entries, np.nan)

    tp_net = tp_frac - fee_rt
    sl_net = -sl_frac - fee_rt

    for k in range(n_entries):
        i = int(entry_idx[k])
        if i + 1 >= n_bars:
            continue
        entry_price = open_arr[i + 1]
        if np.isnan(entry_price) or entry_price <= 0:
            continue

        tp_price = entry_price * (1 + tp_frac)
        sl_price = entry_price * (1 - sl_frac)
        hit = None

        end = min(i + 1 + window, n_bars)
        for j in range(i + 1, end):
            h = high_arr[j]
            l = low_arr[j]
            if np.isnan(h) or np.isnan(l):
                continue
            tp_hit = h >= tp_price
            sl_hit = l <= sl_price
            if tp_hit and sl_hit:
                hit = sl_net
                break
            elif tp_hit:
                hit = tp_net
                break
            elif sl_hit:
                hit = sl_net
                break

        if hit is None:
            last = min(i + window, n_bars - 1)
            close_ret = (open_arr[last] - entry_price) / entry_price if not np.isnan(open_arr[last]) else 0.0
            hit = close_ret - fee_rt

        rets[k] = hit

    return rets


def calc_metrics(trade_rets: np.ndarray) -> dict:
    """Calcula métricas básicas de trading."""
    n = len(trade_rets)
    if n == 0:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "ret": 0.0, "mdd": 0.0, "sharpe": 0.0}

    wins = trade_rets > 0
    wr = float(wins.mean())
    
    gw = float(trade_rets[wins].sum())
    gl = float(abs(trade_rets[~wins].sum()))
    pf = gw / gl if gl > 0 else np.inf
    
    equity = CAPITAL_INICIAL * np.cumprod(1 + trade_rets)
    total_ret = (equity[-1] - CAPITAL_INICIAL) / CAPITAL_INICIAL * 100
    
    # Max Drawdown
    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / running_max
    mdd = float(dd.min()) * 100
    
    # Sharpe
    mean_r, std_r = trade_rets.mean(), trade_rets.std()
    ann_factor = np.sqrt((365 * 24 * 60 / 15) / COOLDOWN_BARS) # factor anualizado
    sharpe = mean_r / std_r * ann_factor if std_r > 0 else 0.0

    return {
        "n": n,
        "wr": wr * 100,
        "pf": pf,
        "ret": total_ret,
        "mdd": mdd,
        "sharpe": sharpe,
        "equity": equity
    }


def main():
    console.print(Panel(
        "[bold]Backtesting del Filtro de Machine Learning (Meta-Labeling)[/bold]\n\n"
        "Estrategia Base: [cyan]M1 (tp150_sl050_maker)[/cyan]\n"
        "Filtro ML: RandomForest Meta-Model\n"
        "Periodo de Evaluación: [yellow]Julio 2025 – Junio 2026 (Out-of-Sample)[/yellow]\n"
        "Fees: [green]0.04% maker round-trip[/green]",
        title="FASE 8 — Backtesting ML",
        border_style="green"
    ))

    # 1. Cargar datos y modelo
    df = pd.read_parquet(MASTER_PATH).sort_values("timestamp").reset_index(drop=True)
    
    model    = joblib.load(MODEL_DIR / "meta_label_rf.pkl")
    selector = joblib.load(MODEL_DIR / "feature_selector.pkl")
    with open(MODEL_DIR / "model_features.json", "r") as f:
        all_features = json.load(f)

    # 2. Reconstruir señales base M1 y generar trades no-solapados
    base_mask = build_base_signal(df)
    all_valid = np.where(base_mask & df["target_tp_sl_2to1"].notna())[0]
    
    entry_idx = []
    last_exit = -1
    for idx in all_valid:
        if idx > last_exit:
            entry_idx.append(idx)
            last_exit = idx + COOLDOWN_BARS - 1
    entry_idx = np.array(entry_idx)

    # Evitar segfault al rebanar datetimes
    timestamps_int = df["timestamp"].astype("int64").values[entry_idx]
    timestamps = pd.to_datetime(timestamps_int, unit="ms")

    # Split temporal (solo queremos evaluar el set de Test)
    split_date = pd.Timestamp("2025-07-01")
    test_indices = np.where(timestamps >= split_date)[0]
    test_entries = entry_idx[test_indices]
    
    if len(test_entries) == 0:
        console.print("[red]Error: No hay operaciones en el set de Test.[/red]")
        return
        
    console.print(f"Operaciones de la estrategia base en Test: {len(test_entries)}")

    # 3. Preparar variables predictoras para las operaciones de test
    df_filled = df.copy()
    from pandas.api.types import is_numeric_dtype
    for col in df_filled.columns:
        if col in ["timestamp", "year_month", "date_key"]:
            continue
        if not is_numeric_dtype(df_filled[col]):
            df_filled[col] = df_filled[col].astype("category").cat.codes
        else:
            df_filled[col] = df_filled[col].fillna(0.0)

    # Excluir no predictores
    exclude_prefixes = ["fwd_", "target_", "tpsl_", "mfe_", "mae_", "risk_reward_"]
    exclude_exact = [
        "timestamp", "open", "high", "low", "close", "volume", "quote_volume",
        "close_time", "vwap_daily", "supertrend_val", "rolling_ath_4w",
        "sma_5", "sma_9", "sma_20", "sma_50", "sma_100", "sma_200",
        "ema_9", "ema_12", "ema_20", "ema_21", "ema_26", "ema_50", "ema_100", "ema_200",
        "bb_lower", "bb_mid", "bb_upper", "kc_lower", "kc_upper",
        "dc_20_lower", "dc_20_upper", "dc_55_lower", "dc_55_upper",
        "year_month", "date_key"
    ]
    features_full = [col for col in df_filled.columns if col not in exclude_exact and not any(col.startswith(p) for p in exclude_prefixes)]

    X_test_all = df_filled.loc[test_entries, features_full].copy()
    X_test_sel = selector.transform(X_test_all)

    # 4. Predecir probabilidad de éxito
    probs = model.predict_proba(X_test_sel)[:, 1]

    # Simular retornos de todas las operaciones de test usando TP/SL
    high_arr  = df["high"].values.astype(float)
    low_arr   = df["low"].values.astype(float)
    open_arr  = df["open"].values.astype(float)
    
    test_rets = simulate_tp_sl(
        test_entries, high_arr, low_arr, open_arr,
        TP_FRAC, SL_FRAC, FEE_RT, window=COOLDOWN_BARS
    )

    # 5. Evaluar diferentes umbrales (Thresholds)
    thresholds = [0.0, 0.30, 0.33, 0.35, 0.37]
    results    = []

    for th in thresholds:
        mask = probs >= th
        filtered_rets = test_rets[mask]
        metrics = calc_metrics(filtered_rets)
        results.append({
            "threshold": th,
            "n":         metrics["n"],
            "wr":        metrics["wr"],
            "pf":        metrics["pf"],
            "ret":       metrics["ret"],
            "mdd":       metrics["mdd"],
            "sharpe":    metrics["sharpe"],
            "equity":    metrics["equity"]
        })

    # Imprimir tabla comparativa
    table = Table(title="Comparativa de Desempeño: Filtro ML en Set de Test", border_style="green")
    table.add_column("Filtro Probabilidad (Th)", style="cyan")
    table.add_column("N Ops",                   justify="right")
    table.add_column("Win Rate",                justify="right")
    table.add_column("Profit Factor",           justify="right")
    table.add_column("Retorno Total",           justify="right")
    table.add_column("Max DD",                  justify="right", style="red")
    table.add_column("Sharpe Ratio",            justify="right", style="yellow")

    for r in results:
        th_label = "Base (Sin Filtro)" if r["threshold"] == 0.0 else f"P >= {r['threshold']:.2f}"
        ret_c = "green" if r["ret"] > 0 else "red"
        table.add_row(
            th_label,
            f"{r['n']:,}",
            f"{r['wr']:.1f}%",
            f"{r['pf']:.2f}" if r["pf"] < np.inf else ">100",
            f"[{ret_c}]{r['ret']:+.1f}%[/{ret_c}]",
            f"{r['mdd']:.1f}%",
            f"{r['sharpe']:.2f}"
        )
    console.print(table)

    # 6. Generar curva de equity interactiva en un reporte HTML
    console.print("\nGenerando gráfica comparativa...")
    import plotly.graph_objects as go
    
    fig = go.Figure()
    for r in results:
        if r["n"] == 0:
            continue
        th_label = "Sin Filtro (Base)" if r["threshold"] == 0.0 else f"Filtro ML P >= {r['threshold']:.2f}"
        fig.add_trace(go.Scatter(
            y=r["equity"],
            mode="lines",
            name=th_label,
            line=dict(width=2.5 if r["threshold"] == 0.35 else 1.5)
        ))
        
    fig.update_layout(
        title="Curva de Equity Comparativa — Filtro ML (Out-of-Sample Jul 2025 - Jun 2026)",
        xaxis_title="N° Operación",
        yaxis_title="Capital USD",
        template="plotly_dark",
    )
    
    # Guardar reporte HTML de ML
    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>BTC Quant Lab — Filtro de Machine Learning</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    body {{ background: #0d1117; color: #e6edf3; font-family: sans-serif; padding: 2rem; }}
    h1 {{ color: #58a6ff; }}
    .container {{ max-width: 1100px; margin: 0 auto; }}
    .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1.5rem; margin-top: 1.5rem; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
    th, td {{ padding: 0.8rem; text-align: center; border-bottom: 1px solid #30363d; }}
    th {{ background: #21262d; }}
    .green {{ color: #3fb950; font-weight: bold; }}
    .red {{ color: #f85149; font-weight: bold; }}
  </style>
</head>
<body>
<div class="container">
  <h1>🔬 BTC Quant Lab — Backtest con Filtro de ML</h1>
  <p>Evaluación Fuera de Muestra (Out-of-Sample) desde <strong>2025-07-01</strong> hasta <strong>2026-06-28</strong>.</p>
  
  <div class="card">
    <h2>Curva de Equity Comparativa</h2>
    <div id="chart" style="height: 500px;"></div>
  </div>

  <div class="card">
    <h2>Métricas Detalladas</h2>
    <table>
      <thead>
        <tr>
          <th>Filtro de Probabilidad</th>
          <th>N° Operaciones</th>
          <th>Win Rate</th>
          <th>Profit Factor</th>
          <th>Retorno Total</th>
          <th>Max Drawdown</th>
          <th>Sharpe Ratio</th>
        </tr>
      </thead>
      <tbody>
    """
    for r in results:
        th_label = "Base (Sin Filtro)" if r["threshold"] == 0.0 else f"P >= {r['threshold']:.2f}"
        ret_class = "green" if r["ret"] > 0 else "red"
        html_content += f"""
        <tr>
          <td><strong>{th_label}</strong></td>
          <td>{r['n']}</td>
          <td>{r['wr']:.1f}%</td>
          <td>{r['pf']:.2f}</td>
          <td class="{ret_class}">{r['ret']:+.1f}%</td>
          <td class="red">{r['mdd']:.1f}%</td>
          <td>{r['sharpe']:.2f}</td>
        </tr>
        """
        
    html_content += f"""
      </tbody>
    </table>
  </div>
</div>
<script>
  Plotly.newPlot('chart', {fig.to_json()}.data, {fig.to_json()}.layout, {{responsive: true}});
</script>
</body>
</html>
"""
    
    out_path = REPORTS_DIR / "ml_backtest_report.html"
    out_path.write_text(html_content, encoding="utf-8")
    console.print(f"[bold green]✓ Reporte de ML generado: {out_path}[/bold green]")


if __name__ == "__main__":
    main()
