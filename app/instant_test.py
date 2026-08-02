import sys
sys.path.append('/home/cplaza/app')

import database
import ml_engine
import app

klines = database.get_candles_15m_range(limit=25000)
entry_rules = database.get_strategy("entry")
exit_rules = database.get_strategy("exit")
ml_pack = ml_engine.load_ml_model()

print(f"--- ANALISIS DE UMBRAL DE CONFIANZA ML EN {len(klines)} VELAS (MUESTRA DE APORTES) ---")
for t_pct in [35, 40, 44, 50, 55, 60, 65, 70]:
    if ml_pack:
        ml_pack["threshold"] = t_pct / 100.0
    res = app.run_full_simulation(klines, entry_rules, exit_rules, initial_balance=10000.0)
    s = res["summary"]
    pnl = s["ml_total_pnl_usdt"]
    wr = s["ml_win_rate_pct"]
    trades = s["ml_completed_trades"]
    wins = s["ml_winning_trades"]
    losses = s["ml_losing_trades"]
    avoided = s["ml_filtered_losses_count"]
    print(f"Umbral {t_pct}% => Ganancia: ${pnl:,.2f} USDT | WinRate: {wr:.1f}% | Trades: {trades} ({wins}W / {losses}L) | Pérdidas Evitadas: {avoided}")
