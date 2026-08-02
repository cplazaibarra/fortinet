"""
FASE 5 — Backtesting Masivo Vectorizado
========================================
Metodología TP/SL real (no hold fijo):
  - Entry en open de vela t+1 tras señal en vela t
  - TP = X% desde entry   |  SL = X/2% desde entry  (RR = 2:1)
  - Si en la ventana de 8h (32 velas) toca TP primero → WIN (+TP_NET)
  - Si toca SL primero                               → LOSS (-SL_NET)
  - Si ninguno                                        → expira (+/-0 o close)

Trades NO solapados: cooldown de 32 velas tras cada entrada.

Costos Binance: 0.10% compra + 0.10% venta = 0.20% round-trip.

Salidas: data/backtests/backtest_summary.parquet
         data/backtests/monthly_breakdown.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
from rich.table import Table

console = Console()

MASTER_PATH  = Path("data/master/master_dataset.parquet")
SIGNALS_PATH = Path("data/signals/signal_evaluation.parquet")
BT_DIR       = Path("data/backtests")
BT_DIR.mkdir(parents=True, exist_ok=True)

FEE_ENTRY = 0.0005   # 0.05% compra
FEE_EXIT  = 0.0005   # 0.05% venta
FEE_TOTAL = FEE_ENTRY + FEE_EXIT  # 0.10% round trip
FEE_MAKER = 0.0001  # 0.01%
FEE_TOTAL_MAKER = FEE_MAKER * 2   # 0.02%

CAPITAL_INICIAL = 10_000.0
MIN_TRADES      = 15
COOLDOWN_BARS   = 12   # 3h @ 15m

# Configuraciones de Take Profit, Stop Loss y Trailing Stop (tp_gross, sl_gross, fee_rt)
TP_SL_CONFIGS = [
    ("tp150_sl075_stock", 0.015, 0.0075, FEE_TOTAL),
    ("tp250_sl100_stock", 0.025, 0.0100, FEE_TOTAL),
    ("tp100_sl050_stock", 0.010, 0.0050, FEE_TOTAL),
]
PRIMARY_CONFIG = "tp150_sl075_stock"


# ─────────────────────────────────────────────────────────────────────
def calc_max_drawdown(equity_curve: np.ndarray) -> float:
    running_max = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - running_max) / running_max
    return float(dd.min())


def calc_consecutive(wins_arr: np.ndarray) -> tuple[int, int]:
    max_wins = max_losses = cur_wins = cur_losses = 0
    for w in wins_arr:
        if w:
            cur_wins += 1; cur_losses = 0
            max_wins = max(max_wins, cur_wins)
        else:
            cur_losses += 1; cur_wins = 0
            max_losses = max(max_losses, cur_losses)
    return max_wins, max_losses


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
    """
    Para cada entrada i, simula salida TP/SL en ventana [i+1, i+window].
    Retorna array de retornos netos (ya descontado fee_rt).
    """
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
                # Ambos en la misma vela → la que sea más probable primero
                # Aproximación conservadora: SL primero
                hit = sl_net
                break
            elif tp_hit:
                hit = tp_net
                break
            elif sl_hit:
                hit = sl_net
                break

        if hit is None:
            # Tiempo expirado → cerrar al close de la última vela
            last = min(i + window, n_bars - 1)
            close_ret = (open_arr[last] - entry_price) / entry_price if not np.isnan(open_arr[last]) else 0.0
            hit = close_ret - fee_rt

        rets[k] = hit

    return rets[~np.isnan(rets)]


def backtest_one(trade_rets: np.ndarray, trade_months=None) -> dict:
    n = len(trade_rets)
    if n < MIN_TRADES:
        return {}

    wins   = trade_rets > 0
    losses = trade_rets <= 0

    win_rate    = float(wins.mean())
    avg_win     = float(trade_rets[wins].mean())  if wins.any()   else 0.0
    avg_loss    = float(trade_rets[losses].mean()) if losses.any() else 0.0
    gross_win   = float(trade_rets[wins].sum())
    gross_loss  = float(abs(trade_rets[losses].sum()))

    profit_factor = gross_win / gross_loss if gross_loss > 1e-10 else np.inf
    expectancy    = win_rate * avg_win + (1 - win_rate) * avg_loss
    payoff_ratio  = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

    equity = CAPITAL_INICIAL * np.cumprod(1 + trade_rets)
    max_dd = calc_max_drawdown(equity)
    total_ret = (equity[-1] - CAPITAL_INICIAL) / CAPITAL_INICIAL
    # CAGR aproximado (asume que la frecuencia de trades es variable)
    n_years = n * (COOLDOWN_BARS * 15 / (60 * 24 * 365))
    cagr = (1 + total_ret) ** (1 / max(n_years, 0.1)) - 1 if total_ret > -1 else -1.0

    mean_r, std_r = trade_rets.mean(), trade_rets.std()
    # Factor de anualización basado en cooldown real
    trades_per_year = (365 * 24 * 60 / 15) / COOLDOWN_BARS
    ann_factor = np.sqrt(trades_per_year)
    sharpe  = mean_r / std_r * ann_factor if std_r > 0 else 0.0
    down_std = trade_rets[trade_rets < 0].std()
    sortino = mean_r / down_std * ann_factor if down_std > 0 else 0.0

    calmar          = cagr / abs(max_dd) if max_dd < -1e-6 else 0.0
    recovery_factor = total_ret / abs(max_dd) if max_dd < -1e-6 else np.inf
    max_wins, max_losses = calc_consecutive(wins)

    return {
        "n_trades":          n,
        "win_rate":          win_rate,
        "avg_win_pct":       avg_win * 100,
        "avg_loss_pct":      avg_loss * 100,
        "profit_factor":     profit_factor,
        "expectancy_pct":    expectancy * 100,
        "payoff_ratio":      payoff_ratio,
        "sharpe":            sharpe,
        "sortino":           sortino,
        "calmar":            calmar,
        "max_drawdown_pct":  max_dd * 100,
        "recovery_factor":   recovery_factor,
        "total_return_pct":  total_ret * 100,
        "final_capital":     equity[-1],
        "max_capital":       equity.max(),
        "cagr_pct":          cagr * 100,
        "max_consec_wins":   max_wins,
        "max_consec_losses": max_losses,
        "fee_total_pct":     FEE_TOTAL * 100,
    }


def calc_monthly_breakdown(trade_rets, trade_months) -> pd.DataFrame:
    if len(trade_rets) == 0:
        return pd.DataFrame()
    monthly = pd.DataFrame({"month": trade_months, "ret": trade_rets})
    rows = []
    capital = CAPITAL_INICIAL
    for month, grp in monthly.groupby("month"):
        rets = grp["ret"].values
        wins = rets > 0
        gw = rets[wins].sum()
        gl = abs(rets[~wins].sum())
        month_ret = np.prod(1 + rets) - 1
        capital *= (1 + month_ret)
        rows.append({
            "month":         month,
            "n_trades":      len(rets),
            "win_rate":      float(wins.mean()),
            "month_ret_pct": month_ret * 100,
            "profit_factor": gw / gl if gl > 1e-10 else np.inf,
            "capital":       capital,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
def main():
    console.print(Panel(
        "[bold]Backtesting Masivo TP/SL (Risk:Reward = 2:1)[/bold]\n\n"
        "Metodología:\n"
        "  • Entrada en open de vela t+1 tras señal en vela t\n"
        "  • Salida por TP (+1%) o SL (-0.5%) en ventana de 8h\n"
        "  • Trades no solapados (cooldown 8h entre entradas)\n\n"
        "[yellow]Costos Binance: 0.10% compra + 0.10% venta = 0.20% round-trip[/yellow]\n"
        f"Capital inicial: ${CAPITAL_INICIAL:,.0f}",
        title="FASE 5 — Backtesting TP/SL",
        border_style="magenta"
    ))

    # ── Carga ─────────────────────────────────────────────────────────
    console.print("\nCargando Master Dataset...")
    df = pd.read_parquet(MASTER_PATH).sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    console.print(f"  {n:,} velas")

    high_arr  = df["high"].values.astype(float)
    low_arr   = df["low"].values.astype(float)
    open_arr  = df["open"].values.astype(float)

    # year_month sin timezone
    try:
        df["year_month"] = df["timestamp"].dt.tz_localize(None).dt.to_period("M").astype(str)
    except Exception:
        df["year_month"] = df["timestamp"].dt.to_period("M").astype(str)
    ym_arr = df["year_month"].values

    # ── Señales candidatas ────────────────────────────────────────────
    console.print("\nCargando señales evaluadas...")
    df_sigs = pd.read_parquet(SIGNALS_PATH)
    console.print(f"  {len(df_sigs):,} combinaciones cargadas")

    if "relative_ret" in df_sigs.columns:
        candidates = df_sigs[
            (df_sigs["relative_ret"] > 0) &
            (df_sigs["n_trades"] >= MIN_TRADES)
        ].copy()
    else:
        candidates = df_sigs[df_sigs["n_trades"] >= MIN_TRADES].nlargest(200, "mean_ret").copy()

    console.print(f"  Candidatas: {len(candidates):,}")

    # ── Importar catálogo de señales ──────────────────────────────────
    try:
        from signals_catalog import SIGNAL_CATALOG
    except ImportError:
        # Reconstruir catálogo desde 05_signals.py
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "signals", Path(__file__).parent / "05_signals.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        SIGNAL_CATALOG = mod.SIGNAL_CATALOG

    def get_sig_mask(name: str) -> np.ndarray:
        if name in SIGNAL_CATALOG:
            try:
                s = SIGNAL_CATALOG[name][1](df).fillna(False).astype(bool).values
                return s.astype(bool)
            except Exception:
                return np.zeros(n, dtype=bool)
        return np.zeros(n, dtype=bool)

    console.print("Pre-calculando máscaras de señales...")
    cached_masks = {}
    for name in SIGNAL_CATALOG:
        cached_masks[name] = get_sig_mask(name)

    def get_combined_mask(row: dict) -> np.ndarray:
        s1 = row.get("s1", row.get("signal_name", ""))
        s2 = row.get("s2", None)
        s3 = row.get("s3", None)
        mask = cached_masks.get(s1, np.zeros(n, dtype=bool))
        if isinstance(s2, str) and s2 in cached_masks:
            mask = mask & cached_masks[s2]
        if isinstance(s3, str) and s3 in cached_masks:
            mask = mask & cached_masks[s3]
        return mask

    # ── Backtesting por config TP/SL ──────────────────────────────────
    all_results = []
    all_monthly = []

    for config_name, tp_frac, sl_frac, fee_rt in TP_SL_CONFIGS:
        tp_net = (tp_frac - fee_rt) * 100
        sl_net = (sl_frac + fee_rt) * 100
        be_wr  = sl_net / (tp_net + sl_net) * 100  # break-even win rate
        console.print(
            f"\n[bold]Config: {config_name}[/bold] "
            f"| TP neto={tp_net:.2f}% | SL neto={sl_net:.2f}% "
            f"| Break-even WR={be_wr:.1f}%"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            console=console
        ) as prog:
            task = prog.add_task(f"BT {config_name}...", total=len(candidates))

            for _, cand in candidates.iterrows():
                cand_dict = cand.to_dict()
                mask = get_combined_mask(cand_dict)

                # ── Selección no solapante ────────────────────────────
                all_valid = np.where(mask)[0]
                if len(all_valid) == 0:
                    prog.advance(task)
                    continue

                entry_idx = []
                last_exit = -1
                for idx in all_valid:
                    if idx > last_exit and idx + 1 < n:
                        entry_idx.append(idx)
                        last_exit = idx + COOLDOWN_BARS - 1
                entry_idx = np.array(entry_idx)

                if len(entry_idx) < MIN_TRADES:
                    prog.advance(task)
                    continue

                # ── Simulación TP/SL ──────────────────────────────────
                trade_rets = simulate_tp_sl(
                    entry_idx, high_arr, low_arr, open_arr,
                    tp_frac, sl_frac, fee_rt, window=COOLDOWN_BARS
                )

                if len(trade_rets) < MIN_TRADES:
                    prog.advance(task)
                    continue

                # Months para breakdown
                trade_months = ym_arr[
                    np.clip(entry_idx[:len(trade_rets)] + 1, 0, n-1)
                ]

                bt = backtest_one(trade_rets, trade_months)
                if not bt:
                    prog.advance(task)
                    continue

                bt.update({
                    "signal_name": cand["signal_name"],
                    "category":    cand.get("category", ""),
                    "level":       int(cand.get("level", 1)),
                    "config":      config_name,
                    "tp_pct":      tp_frac * 100,
                    "sl_pct":      sl_frac * 100,
                    "fee_rt_pct":  fee_rt * 100,
                    "be_winrate":  sl_net / (tp_net + sl_net),
                    "alpha_pct":   cand.get("relative_ret", 0) * 100,
                })
                all_results.append(bt)

                monthly_df = calc_monthly_breakdown(trade_rets, trade_months)
                if not monthly_df.empty:
                    monthly_df["signal_name"] = cand["signal_name"]
                    monthly_df["config"]      = config_name
                    all_monthly.append(monthly_df)

                prog.advance(task)

    # ── Guardar resultados ────────────────────────────────────────────
    if not all_results:
        console.print("[red]Sin resultados.[/red]")
        return

    df_bt = pd.DataFrame(all_results)
    df_bt.to_parquet(BT_DIR / "backtest_summary.parquet", index=False)

    if all_monthly:
        df_monthly = pd.concat(all_monthly, ignore_index=True)
        df_monthly.to_parquet(BT_DIR / "monthly_breakdown.parquet", index=False)

    # ── Tabla resumen ─────────────────────────────────────────────────
    # Primario: config tp100_sl050, ordenar por expectancy
    primary = df_bt[df_bt["config"] == PRIMARY_CONFIG].copy()
    if primary.empty:
        primary = df_bt.copy()

    primary_sorted = primary.sort_values("expectancy_pct", ascending=False)

    table = Table(
        title=f"Top 15 Estrategias — {PRIMARY_CONFIG} (RR 2:1, 0.20% fees)",
        border_style="magenta"
    )
    for col, sty, just in [
        ("#", "dim", "right"), ("Señal", "cyan", "left"),
        ("N", "white", "right"), ("WR%", "green", "right"),
        ("PF", "white", "right"), ("Expect%", "green", "right"),
        ("MDD%", "red", "right"), ("Ret%", "green", "right"),
        ("Sharpe", "yellow", "right"), ("Fees", "dim", "right"),
    ]:
        table.add_column(col, style=sty, justify=just)

    top15 = primary_sorted.head(15)
    for i, row in enumerate(top15.itertuples(), 1):
        ret_color  = "green" if row.total_return_pct > 0 else "red"
        exp_color  = "green" if row.expectancy_pct > 0   else "red"
        table.add_row(
            str(i),
            row.signal_name[:55] if hasattr(row, 'signal_name') else "",
            f"{row.n_trades:,}",
            f"{row.win_rate*100:.1f}%",
            f"{row.profit_factor:.2f}" if row.profit_factor < 100 else ">100",
            f"[{exp_color}]{row.expectancy_pct:+.4f}%[/{exp_color}]",
            f"{row.max_drawdown_pct:.1f}%",
            f"[{ret_color}]{row.total_return_pct:+.1f}%[/{ret_color}]",
            f"{row.sharpe:.2f}",
            "0.2%",
        )
    console.print(table)

    # Stats globales
    n_positive = (df_bt["total_return_pct"] > 0).sum()
    n_pos_exp  = (df_bt["expectancy_pct"] > 0).sum()
    best        = primary_sorted.iloc[0]

    console.print(f"\n[bold]Resumen Global:[/bold]")
    console.print(f"  Estrategias evaluadas:       {len(df_bt):,}")
    console.print(f"  Con retorno total positivo:  {n_positive:,}")
    console.print(f"  Con expectancy positiva:     {n_pos_exp:,}")
    console.print(f"  Mejor expectancy ({PRIMARY_CONFIG}): {best['expectancy_pct']:+.4f}%/trade")
    console.print(f"  Mejor retorno total:          {primary_sorted['total_return_pct'].max():+.1f}%")
    console.print(f"\n[bold green]✓ Backtesting completado[/bold green]")
    console.print(f"  Archivos guardados en: {BT_DIR}")
    console.print(
        f"\n[bold yellow]ⓘ  Cada retorno ya descuenta {FEE_TOTAL*100:.2f}% de comisiones Binance[/bold yellow]"
        f"\n   (0.10% compra + 0.10% venta)"
    )


if __name__ == "__main__":
    main()
