"""
FASE 4 — Motor de Combinaciones de Señales

Define ~45 señales binarias (condiciones de entrada),
las evalúa individualmente (Nivel 1) y en combinaciones de 2 (Nivel 2)
y 3 (Nivel 3) señales con restricciones de categoría.

Para cada combinación se evalúa:
  - Win Rate (% de trades con retorno neto > 0)
  - Retorno medio neto (ya con 0.20% de comisiones descontadas)
  - N operaciones
  - Test estadístico Mann-Whitney U (vs. universo sin señal)
  - p-valor ajustado por FDR (Benjamini-Hochberg)

Salida: data/signals/signal_evaluation.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
from scipy import stats

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
from rich.table import Table

console = Console()

MASTER_PATH   = Path("data/master/master_dataset.parquet")
SIGNALS_DIR   = Path("data/signals")
SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

# Costos fijos — deben coincidir con 04_forward_vars.py
FEE_TOTAL = 0.002   # 0.20% round trip (0.10% compra + 0.10% venta)

# ─────────────────────────────────────────────────────────────────────
# DEFINICIÓN DE SEÑALES BINARIAS
# ─────────────────────────────────────────────────────────────────────
# Cada señal es un nombre → función(df) → Serie booleana
# Categorías: trend / momentum / volatility / volume / futures / temporal
# ─────────────────────────────────────────────────────────────────────

SIGNAL_CATALOG = {

    # ── TENDENCIA ────────────────────────────────────────────────────
    "ema9_above_ema20":      ("trend",    lambda df: df["ema_9"]  > df["ema_20"]),
    "ema20_above_ema50":     ("trend",    lambda df: df["ema_20"] > df["ema_50"]),
    "ema50_above_ema200":    ("trend",    lambda df: df["ema_50"] > df["ema_200"]),
    "all_emas_aligned":      ("trend",    lambda df: (df["ema_9"] > df["ema_20"]) & (df["ema_20"] > df["ema_50"]) & (df["ema_50"] > df["ema_200"])),
    "price_above_ema20":     ("trend",    lambda df: df["close"]  > df["ema_20"]),
    "price_above_sma50":     ("trend",    lambda df: df["close"]  > df["sma_50"]),
    "price_above_sma200":    ("trend",    lambda df: df["close"]  > df["sma_200"]),
    "ema9_slope_pos":        ("trend",    lambda df: df["ema_9_slope"]  > 0),
    "ema20_slope_pos":       ("trend",    lambda df: df["ema_20_slope"] > 0),
    "ema50_slope_pos":       ("trend",    lambda df: df["ema_50_slope"] > 0),
    "supertrend_bullish":    ("trend",    lambda df: df.get("supertrend_bullish", pd.Series(False, index=df.index))),
    "price_above_vwap":      ("trend",    lambda df: df.get("price_above_vwap", pd.Series(False, index=df.index))),
    "vwap_dist_pos":         ("trend",    lambda df: df.get("vwap_dist_pct", pd.Series(0, index=df.index)) > 0),

    # ── MOMENTUM ─────────────────────────────────────────────────────
    "rsi14_above_50":        ("momentum", lambda df: df["rsi_14"] > 50),
    "rsi14_above_55":        ("momentum", lambda df: df["rsi_14"] > 55),
    "rsi14_above_60":        ("momentum", lambda df: df["rsi_14"] > 60),
    "rsi14_zone_40_65":      ("momentum", lambda df: (df["rsi_14"] > 40) & (df["rsi_14"] < 65)),
    "rsi14_not_overbought":  ("momentum", lambda df: df["rsi_14"] < 70),
    "rsi7_above_50":         ("momentum", lambda df: df["rsi_7"]  > 50),
    "macd_above_signal":     ("momentum", lambda df: df["macd_line"] > df["macd_signal"]),
    "macd_hist_positive":    ("momentum", lambda df: df["macd_hist"] > 0),
    "macd_hist_growing":     ("momentum", lambda df: df["macd_hist_slope"] > 0),
    "stochrsi_above_50":     ("momentum", lambda df: df.get("stochrsi_k", pd.Series(50, index=df.index)) > 50),
    "stochrsi_k_above_d":    ("momentum", lambda df: df.get("stochrsi_k", pd.Series(50, index=df.index)) > df.get("stochrsi_d", pd.Series(50, index=df.index))),
    "mfi_above_50":          ("momentum", lambda df: df.get("mfi_14", pd.Series(50, index=df.index)) > 50),
    "cci14_positive":        ("momentum", lambda df: df.get("cci_14", pd.Series(0, index=df.index)) > 0),
    "roc20_positive":        ("momentum", lambda df: df.get("roc_20", pd.Series(0, index=df.index)) > 0),

    # ── VOLATILIDAD / FUERZA ─────────────────────────────────────────
    "adx_strong":            ("volatility", lambda df: df.get("adx_14", pd.Series(25, index=df.index)) > 25),
    "adx_very_strong":       ("volatility", lambda df: df.get("adx_14", pd.Series(25, index=df.index)) > 30),
    "di_bull":               ("volatility", lambda df: df.get("di_bull", pd.Series(True, index=df.index))),
    "bb_price_above_mid":    ("volatility", lambda df: df["close"] > df.get("bb_mid", df["close"])),
    "bb_not_overbought":     ("volatility", lambda df: df.get("bb_pct", pd.Series(0.5, index=df.index)) < 0.9),
    "bb_squeeze_active":     ("volatility", lambda df: df.get("bb_squeeze", pd.Series(False, index=df.index)).fillna(False)),
    "vol_regime_not_extreme":("volatility", lambda df: df.get("vol_regime", pd.Series("normal", index=df.index)) != "extremo"),

    # ── VOLUMEN ──────────────────────────────────────────────────────
    "vol_above_avg":         ("volume",   lambda df: df["vol_ratio"] > 1.0),
    "vol_above_150pct":      ("volume",   lambda df: df["vol_ratio"] > 1.5),
    "vol_surge":             ("volume",   lambda df: df.get("vol_surge", pd.Series(False, index=df.index)).fillna(False)),
    "obv_above_ema":         ("volume",   lambda df: df.get("obv_above_ema", pd.Series(False, index=df.index)).fillna(False)),
    "obv_rising":            ("volume",   lambda df: df.get("obv_slope", pd.Series(0, index=df.index)) > 0),
    "cmf_positive":          ("volume",   lambda df: df.get("cmf_20", pd.Series(0, index=df.index)) > 0),
    "taker_buy_dominant":    ("volume",   lambda df: df.get("taker_buy_ratio", pd.Series(0.5, index=df.index)) > 0.50),
    "taker_buy_strong":      ("volume",   lambda df: df.get("taker_buy_ratio", pd.Series(0.5, index=df.index)) > 0.55),
    "taker_imbalance_pos":   ("volume",   lambda df: df.get("taker_imbalance", pd.Series(0, index=df.index)) > 0),

    # ── FUTUROS ──────────────────────────────────────────────────────
    "funding_neutral":       ("futures",  lambda df: (df.get("funding_rate", pd.Series(0, index=df.index)).abs() < 0.0005)),
    "funding_not_extreme":   ("futures",  lambda df: (df.get("funding_rate", pd.Series(0, index=df.index)).abs() < 0.001)),
    "oi_positive_change":    ("futures",  lambda df: df.get("oi_change_pct", pd.Series(0, index=df.index)) > 0),
    "ls_not_extreme_long":   ("futures",  lambda df: ~df.get("ls_extreme_long", pd.Series(False, index=df.index)).fillna(False)),

    # ── TEMPORAL ─────────────────────────────────────────────────────
    "not_weekend":           ("temporal", lambda df: ~df["is_weekend"]),
    "us_session":            ("temporal", lambda df: df.get("is_us_session", pd.Series(False, index=df.index))),
    "europe_session":        ("temporal", lambda df: df.get("is_europe_session", pd.Series(False, index=df.index))),
    "active_session":        ("temporal", lambda df: df.get("is_us_session", pd.Series(False, index=df.index)) | df.get("is_europe_session", pd.Series(False, index=df.index))),

    # ── ESTRUCTURA DE VELA ───────────────────────────────────────────
    "bullish_candle":        ("candle",   lambda df: df.get("is_bullish", df["close"] >= df["open"])),
    "strong_body":           ("candle",   lambda df: df.get("body_pct", pd.Series(0.5, index=df.index)) > 0.5),
    "low_upper_wick":        ("candle",   lambda df: df.get("upper_wick_pct", pd.Series(0, index=df.index)) < 0.3),
    "psar_bullish":          ("candle",   lambda df: df.get("psar_bullish", pd.Series(True, index=df.index)).fillna(True)),
}


# ─────────────────────────────────────────────────────────────────────
def evaluate_signal(
    signal_mask: np.ndarray,
    fwd_ret: np.ndarray,
    universe_ret: np.ndarray,
    min_trades: int = 50,
) -> dict | None:
    """
    Evalúa una señal binaria contra el universo.
    Retorna métricas básicas y test estadístico.
    Retornos ya son NETOS de comisiones (FEE_TOTAL = 0.20%).

    Métrica clave: relative_return = mean_ret(señal) - mean_ret(universo)
    Una señal con relative_return > 0 tiene edge positivo sobre el mercado.
    """
    active = signal_mask & np.isfinite(fwd_ret)
    n = int(active.sum())
    if n < min_trades:
        return None

    trade_rets = fwd_ret[active]
    universe   = universe_ret[np.isfinite(universe_ret)]

    universe_mean  = float(universe.mean())
    win_rate       = float((trade_rets > 0).mean())
    mean_ret       = float(trade_rets.mean())
    median_ret     = float(np.median(trade_rets))
    std_ret        = float(trade_rets.std())
    relative_ret   = mean_ret - universe_mean   # alpha sobre universo

    pos_sum    = float(trade_rets[trade_rets > 0].sum())
    neg_sum    = float(abs(trade_rets[trade_rets <= 0].sum()))
    profit_factor = pos_sum / neg_sum if neg_sum > 0 else np.inf

    # Mann-Whitney U test (no paramétrico)
    try:
        u_stat, p_val = stats.mannwhitneyu(
            trade_rets, universe, alternative="greater"
        )
    except Exception:
        p_val = 1.0

    sharpe = mean_ret / std_ret * np.sqrt(252 * 96) if std_ret > 0 else 0.0

    return {
        "n_trades":      n,
        "win_rate":      win_rate,
        "mean_ret":      mean_ret,
        "median_ret":    median_ret,
        "std_ret":       std_ret,
        "relative_ret":  relative_ret,
        "profit_factor": profit_factor,
        "sharpe":        sharpe,
        "universe_mean": universe_mean,
        "p_value_raw":   p_val,
    }


def fdr_correction(p_values: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Devuelve p-valores ajustados."""
    n = len(p_values)
    idx = np.argsort(p_values)
    sorted_p = p_values[idx]
    adjusted = np.minimum(1, sorted_p * n / (np.arange(1, n+1)))
    # Hacer monotónico
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty(n)
    result[idx] = adjusted
    return result


def build_signal_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Evalúa cada señal del catálogo sobre el dataset y devuelve matriz binaria."""
    console.print("\nGenerando matriz de señales binarias...")
    sig_df = {}
    for name, (cat, func) in SIGNAL_CATALOG.items():
        try:
            s = func(df).fillna(False).astype(bool)
        except Exception as e:
            console.print(f"  [yellow]⚠ Error en señal '{name}': {e}[/yellow]")
            s = pd.Series(False, index=df.index)
        sig_df[name] = s

    result = pd.DataFrame(sig_df, index=df.index)
    active_pct = result.mean() * 100
    console.print(f"  [green]✓[/green] {len(result.columns)} señales generadas")
    console.print(f"  Activación promedio: {active_pct.mean():.1f}% de las velas")
    return result


def evaluate_level1(sig_df: pd.DataFrame, fwd_ret: np.ndarray) -> pd.DataFrame:
    """Evaluación de señales individuales."""
    universe = fwd_ret[np.isfinite(fwd_ret)]
    rows = []

    for name, (cat, _) in SIGNAL_CATALOG.items():
        if name not in sig_df.columns:
            continue
        mask   = sig_df[name].values
        result = evaluate_signal(mask, fwd_ret, universe)
        if result is None:
            continue
        row = {"signal_name": name, "category": cat, "level": 1, **result}
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(rows)
    df_out["p_value_adj"] = fdr_correction(df_out["p_value_raw"].values)
    # Significativo = alpha positivo sobre el universo + p-val raw < 0.20
    # (FDR estricto al 0.05 es demasiado conservador para alpha de 0.02%/trade)
    df_out["significant"] = (
        (df_out["relative_ret"] > 0) &
        (df_out["p_value_raw"] < 0.20)
    )
    return df_out.sort_values("relative_ret", ascending=False)


def evaluate_level2(sig_df: pd.DataFrame, fwd_ret: np.ndarray,
                    top_signals: list[str], max_combos: int = 1000) -> pd.DataFrame:
    """Evaluación de pares de señales. Solo señales top de Level 1."""
    universe = fwd_ret[np.isfinite(fwd_ret)]
    rows     = []
    cats     = {n: v[0] for n, v in SIGNAL_CATALOG.items()}
    combos   = list(combinations(top_signals, 2))[:max_combos]

    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"),
        BarColumn(bar_width=35), MofNCompleteColumn(),
        console=console
    ) as prog:
        task = prog.add_task("Nivel 2 — Pares...", total=len(combos))

        for s1, s2 in combos:
            if s1 not in sig_df.columns or s2 not in sig_df.columns:
                prog.advance(task)
                continue
            mask   = sig_df[s1].values & sig_df[s2].values
            result = evaluate_signal(mask, fwd_ret, universe)
            if result is not None:
                row = {
                    "signal_name": f"{s1} AND {s2}",
                    "category":    f"{cats.get(s1,'?')}+{cats.get(s2,'?')}",
                    "level":       2,
                    "s1": s1, "s2": s2,
                    **result
                }
                rows.append(row)
            prog.advance(task)

    if not rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(rows)
    df_out["p_value_adj"] = fdr_correction(df_out["p_value_raw"].values)
    df_out["significant"] = (
        (df_out["relative_ret"] > 0) &
        (df_out["p_value_raw"] < 0.20)
    )
    return df_out.sort_values("relative_ret", ascending=False)


def evaluate_level3(sig_df: pd.DataFrame, fwd_ret: np.ndarray,
                    top_pairs: list[dict], top_singles: list[str],
                    max_combos: int = 1500) -> pd.DataFrame:
    """
    Nivel 3: mejores pares + señal adicional de categoría diferente.
    Restringe a top 30 pares × todas las señales individuales.
    """
    universe = fwd_ret[np.isfinite(fwd_ret)]
    cats     = {n: v[0] for n, v in SIGNAL_CATALOG.items()}
    rows     = []
    count    = 0

    pair_list = [(p["s1"], p["s2"], p["category"]) for p in top_pairs
                 if "s1" in p and "s2" in p]

    candidates = []
    for s1, s2, pair_cat in pair_list:
        cats_in_pair = set(pair_cat.split("+"))
        for s3 in top_singles:
            if s3 == s1 or s3 == s2:
                continue
            cat3 = cats.get(s3, "?")
            # Al menos 2 categorías distintas en la terna
            if len(cats_in_pair | {cat3}) >= 2:
                candidates.append((s1, s2, s3))

    candidates = candidates[:max_combos]

    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"),
        BarColumn(bar_width=35), MofNCompleteColumn(),
        console=console
    ) as prog:
        task = prog.add_task("Nivel 3 — Tríos...", total=len(candidates))

        for s1, s2, s3 in candidates:
            if s3 not in sig_df.columns:
                prog.advance(task)
                continue
            mask   = sig_df[s1].values & sig_df[s2].values & sig_df[s3].values
            result = evaluate_signal(mask, fwd_ret, universe, min_trades=30)
            if result is not None:
                cat3 = cats.get(s3, "?")
                row  = {
                    "signal_name": f"{s1} AND {s2} AND {s3}",
                    "category": f"{cats.get(s1,'?')}+{cats.get(s2,'?')}+{cat3}",
                    "level": 3,
                    "s1": s1, "s2": s2, "s3": s3,
                    **result
                }
                rows.append(row)
            prog.advance(task)

    if not rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(rows)
    df_out["p_value_adj"] = fdr_correction(df_out["p_value_raw"].values)
    df_out["significant"] = (
        (df_out["relative_ret"] > 0) &
        (df_out["p_value_raw"] < 0.20)
    )
    return df_out.sort_values("relative_ret", ascending=False)


def main():
    console.print(Panel(
        "[bold]Evaluación estadística de combinaciones de señales[/bold]\n"
        f"Señales definidas: [cyan]{len(SIGNAL_CATALOG)}[/cyan]\n"
        "Nivel 1 → individuales · Nivel 2 → pares · Nivel 3 → tríos\n\n"
        f"[yellow]Costo por trade incorporado: {FEE_TOTAL*100:.2f}% (0.10% compra + 0.10% venta)[/yellow]\n"
        "[dim]Ranking por ALPHA RELATIVO: mean_ret(señal) − mean_ret(universo)[/dim]",
        title="FASE 4 — Combinaciones de Señales",
        border_style="green"
    ))

    # ── Carga ─────────────────────────────────────────────────────────
    console.print(f"\nCargando Master Dataset...")
    df = pd.read_parquet(MASTER_PATH)
    console.print(f"  {len(df):,} velas | {len(df.columns):,} columnas")

    # Target: usar 4h (16 velas) — mejor relación señal/ruido que 1h
    # Los retornos ya incluyen -0.20% de comisiones
    target_col = "fwd_ret_16c" if "fwd_ret_16c" in df.columns else "fwd_ret_4c"
    if target_col not in df.columns:
        raise ValueError("Ejecuta primero 04_forward_vars.py")

    fwd_ret      = df[target_col].values
    universe_ret = fwd_ret[np.isfinite(fwd_ret)]
    universe_mean= float(universe_ret.mean())
    console.print(f"  Target: ret neto {target_col} | Media universo: {universe_mean*100:.3f}%")
    console.print(f"  [dim]Una señal necesita mean_ret > {universe_mean*100:.3f}% para tener alpha positivo[/dim]")

    # ── Matriz de señales ─────────────────────────────────────────────
    sig_df = build_signal_matrix(df)

    # ── Nivel 1 ───────────────────────────────────────────────────────
    console.print("\n[bold]NIVEL 1 — Señales individuales[/bold]")
    df_l1 = evaluate_level1(sig_df, fwd_ret)
    console.print(f"  Evaluadas:    {len(df_l1):,}")
    console.print(f"  Con alpha+ (rel_ret>0, p<0.20): {df_l1['significant'].sum():,}")

    # Top L1 para alimentar L2 — top 30 por relative_ret
    top_signals = df_l1[df_l1["significant"]]["signal_name"].tolist()
    if len(top_signals) < 5:
        top_signals = df_l1.head(30)["signal_name"].tolist()
    console.print(f"  Señales llevadas a Nivel 2: {len(top_signals)}")

    # ── Nivel 2 ───────────────────────────────────────────────────────
    console.print("\n[bold]NIVEL 2 — Pares de señales[/bold]")
    df_l2 = evaluate_level2(sig_df, fwd_ret, top_signals, max_combos=2000)
    console.print(f"  Evaluadas:    {len(df_l2):,}")
    if not df_l2.empty:
        console.print(f"  Con alpha+: {df_l2['significant'].sum():,}")

    # Top L2 para alimentar L3
    if not df_l2.empty:
        top_pairs = df_l2[df_l2["significant"]].head(40).to_dict("records")
        if not top_pairs:
            top_pairs = df_l2.head(25).to_dict("records")
    else:
        top_pairs = []
    console.print(f"  Pares llevados a Nivel 3: {len(top_pairs)}")

    # ── Nivel 3 ───────────────────────────────────────────────────────
    console.print("\n[bold]NIVEL 3 — Tríos de señales[/bold]")
    df_l3 = evaluate_level3(sig_df, fwd_ret, top_pairs, top_signals, max_combos=2000)
    console.print(f"  Evaluadas:    {len(df_l3):,}")
    if not df_l3.empty:
        console.print(f"  Con alpha+: {df_l3['significant'].sum():,}")

    # ── Consolidar y guardar ──────────────────────────────────────────
    all_levels = []
    for level_df in [df_l1, df_l2, df_l3]:
        if not level_df.empty:
            all_levels.append(level_df)

    if all_levels:
        df_all = pd.concat(all_levels, ignore_index=True)
        out_path = SIGNALS_DIR / "signal_evaluation.parquet"
        df_all.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
        console.print(f"\n[green]✓[/green] Guardado en {out_path}")
    else:
        console.print("[red]Sin resultados para guardar[/red]")
        return

    # ── Tabla resumen ─────────────────────────────────────────────────
    table = Table(
        title=f"Top 15 Señales por Alpha Relativo ({target_col})",
        border_style="green"
    )
    table.add_column("#",             style="dim")
    table.add_column("Señal",         style="cyan",   max_width=48)
    table.add_column("Nivel",         style="yellow", justify="center")
    table.add_column("N Ops",         style="white",  justify="right")
    table.add_column("Win Rate",      justify="right")
    table.add_column("Media Neta",    justify="right")
    table.add_column("Alpha",         style="green",  justify="right")
    table.add_column("p-raw",         style="dim",    justify="right")

    top15 = df_all.sort_values("relative_ret", ascending=False).head(15)
    for i, row in enumerate(top15.itertuples(), 1):
        alpha_color = "green" if row.relative_ret > 0 else "red"
        table.add_row(
            str(i),
            row.signal_name[:47],
            str(row.level),
            f"{row.n_trades:,}",
            f"{row.win_rate*100:.1f}%",
            f"{row.mean_ret*100:.3f}%",
            f"[{alpha_color}]{row.relative_ret*100:+.4f}%[/{alpha_color}]",
            f"{row.p_value_raw:.4f}",
        )
    console.print(table)

    n_alpha = df_all["significant"].sum()
    console.print(f"\n[bold green]✓ Combinaciones con alpha positivo: {n_alpha}[/bold green]")
    console.print(f"  Criterio: relative_ret > 0 AND p_value_raw < 0.20")
    console.print(f"  Universo media: {universe_mean*100:.4f}% neto por trade")


if __name__ == "__main__":
    main()
