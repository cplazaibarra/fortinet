"""
src/debug_ml.py
"""
import pandas as pd
import numpy as np

print("1. Loading dataset...")
df = pd.read_parquet("data/master/master_dataset.parquet")

print("2. Base signal...")
adx_very_strong = df["adx_14"] > 30 if "adx_14" in df.columns else pd.Series(False, index=df.index)
rsi = df["rsi_14"] if "rsi_14" in df.columns else pd.Series(50, index=df.index)
rsi_zone = (rsi >= 40) & (rsi <= 65)
close = df["close"]
sma50 = df["sma_50"] if "sma_50" in df.columns else close
above_sma = close > sma50
base_mask = adx_very_strong & rsi_zone & above_sma
base_mask = base_mask.fillna(False).astype(bool).values

all_valid = np.where(base_mask & df["target_tp_sl_2to1"].notna())[0]
entry_idx = []
last_exit = -1
for idx in all_valid:
    if idx > last_exit:
        entry_idx.append(idx)
        last_exit = idx + 32 - 1
entry_idx = np.array(entry_idx)

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
features = [col for col in df.columns if col not in exclude_exact and not any(col.startswith(p) for p in exclude_prefixes)]

X = df.loc[entry_idx, features].copy()
y = df.loc[entry_idx, "target_tp_sl_2to1"].values.astype(int)

print("7. Split starting...")
timestamps = df.loc[entry_idx, "timestamp"].values
print("  timestamps type:", type(timestamps), timestamps.dtype)

print("  Converting to datetime index...")
dt_idx = pd.to_datetime(timestamps)
print("  dt_idx created:", type(dt_idx), dt_idx.dtype)

print("  Tz-localize None...")
ts_naive = dt_idx.tz_localize(None)
print("  Tz-localize None complete!")

split_date = pd.Timestamp("2025-07-01")
train_mask = ts_naive < split_date
test_mask = ts_naive >= split_date
print("  Masks created!")

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test   = X[test_mask], y[test_mask]
print("  Data split complete!")

import lightgbm as lgb
print("8. Training LightGBM...")
model = lgb.LGBMClassifier(n_estimators=10, max_depth=3, verbosity=-1)
print("  Fitting model...")
model.fit(X_train, y_train)
print("  Fit complete!")
