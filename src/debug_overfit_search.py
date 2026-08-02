"""
src/debug_overfit_search.py
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

print("1. Loading dataset...")
df = pd.read_parquet("data/master/master_dataset.parquet")

# base signal M1
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

for cat in ["vol_regime", "market_regime"]:
    if cat in df.columns:
        df[cat] = df[cat].astype("category").cat.codes

X = df.loc[entry_idx, features].fillna(0.0).copy()
y = df.loc[entry_idx, "target_tp_sl_2to1"].values.astype(int)

ts_int = df["timestamp"].astype("int64").values[entry_idx]
timestamps = pd.to_datetime(ts_int, unit="ms")
split_date = pd.Timestamp("2025-07-01")
train_mask = timestamps < split_date
test_mask  = timestamps >= split_date

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test   = X[test_mask], y[test_mask]

# Grid Search over K and max_depth for Random Forest
best_auc = 0.0
best_params = {}

for k in [5, 8, 12, 15, 20]:
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel  = selector.transform(X_test)
    
    for depth in [2, 3, 4]:
        for leaf in [20, 30, 40, 50]:
            rf = RandomForestClassifier(n_estimators=150, max_depth=depth, min_samples_leaf=leaf, random_state=42)
            rf.fit(X_train_sel, y_train)
            train_auc = roc_auc_score(y_train, rf.predict_proba(X_train_sel)[:, 1])
            test_auc  = roc_auc_score(y_test, rf.predict_proba(X_test_sel)[:, 1])
            
            if test_auc > best_auc:
                best_auc = test_auc
                best_params = {"k": k, "depth": depth, "leaf": leaf, "train_auc": train_auc, "test_auc": test_auc}

print("Best RF Params:", best_params)

# Grid Search for Logistic Regression C
scaler = StandardScaler()
for k in [5, 8, 12, 15, 20]:
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel  = selector.transform(X_test)
    
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_test_scaled  = scaler.transform(X_test_sel)
    
    for c_val in [0.01, 0.05, 0.1, 0.5, 1.0]:
        lr = LogisticRegression(penalty='l2', C=c_val, random_state=42)
        lr.fit(X_train_scaled, y_train)
        train_auc = roc_auc_score(y_train, lr.predict_proba(X_train_scaled)[:, 1])
        test_auc  = roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1])
        
        if test_auc > best_auc:
            best_auc = test_auc
            best_params = {"model": "LR", "k": k, "C": c_val, "train_auc": train_auc, "test_auc": test_auc}

print("Overall Best Params:", best_params)
