"""
src/debug_overfit.py
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import json

print("1. Loading dataset...")
df = pd.read_parquet("data/master/master_dataset.parquet")

# simulate entry_idx of M1
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

# Features
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

# Handle categories (convert to numeric for logistic regression)
for cat in ["vol_regime", "market_regime"]:
    if cat in df.columns:
        df[cat] = df[cat].astype("category").cat.codes

X = df.loc[entry_idx, features].fillna(0.0).copy()
y = df.loc[entry_idx, "target_tp_sl_2to1"].values.astype(int)

# Split naive-safe datetimes
ts_int = df["timestamp"].astype("int64").values[entry_idx]
timestamps = pd.to_datetime(ts_int, unit="ms")
split_date = pd.Timestamp("2025-07-01")
train_mask = timestamps < split_date
test_mask  = timestamps >= split_date

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test   = X[test_mask], y[test_mask]

# Feature selection
selector = SelectKBest(score_func=f_classif, k=15)
X_train_sel = selector.fit_transform(X_train, y_train)
X_test_sel  = selector.transform(X_test)
selected_feats = [features[i] for i in selector.get_support(indices=True)]
print("Selected top 15 features:", selected_feats)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sel)
X_test_scaled  = scaler.transform(X_test_sel)

# Model 1: Constrained Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=2, min_samples_leaf=35, random_state=42)
rf.fit(X_train_sel, y_train)
rf_train_auc = roc_auc_score(y_train, rf.predict_proba(X_train_sel)[:, 1])
rf_test_auc  = roc_auc_score(y_test, rf.predict_proba(X_test_sel)[:, 1])
print(f"Random Forest AUC -> Train: {rf_train_auc:.4f} | Test: {rf_test_auc:.4f}")

# Model 2: Logistic Regression with L1 Penalty
lr = LogisticRegression(penalty='l1', solver='liblinear', C=0.05, random_state=42)
lr.fit(X_train_scaled, y_train)
lr_train_auc = roc_auc_score(y_train, lr.predict_proba(X_train_scaled)[:, 1])
lr_test_auc  = roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1])
print(f"Logistic Regression L1 AUC -> Train: {lr_train_auc:.4f} | Test: {lr_test_auc:.4f}")
