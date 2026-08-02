"""
src/08_ml_train.py
===================
Entrenamiento de un modelo RandomForest con regularización fuerte y selección de variables (Meta-Labeling).
Evita el sobreajuste y maximiza el AUC-ROC en el set de Test (Out-of-Sample).
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score
import joblib
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

MASTER_PATH = Path("data/master/master_dataset.parquet")
MODEL_DIR   = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

COOLDOWN_BARS = 32
K_FEATURES    = 20


def get_all_features(df: pd.DataFrame) -> list[str]:
    """Identifica todas las columnas candidatas a predictor."""
    exclude_prefixes = [
        "fwd_", "target_", "tpsl_", "mfe_", "mae_", "risk_reward_"
    ]
    exclude_exact = [
        "timestamp", "open", "high", "low", "close", "volume", "quote_volume",
        "close_time", "vwap_daily", "supertrend_val", "rolling_ath_4w",
        "sma_5", "sma_9", "sma_20", "sma_50", "sma_100", "sma_200",
        "ema_9", "ema_12", "ema_20", "ema_21", "ema_26", "ema_50", "ema_100", "ema_200",
        "bb_lower", "bb_mid", "bb_upper", "kc_lower", "kc_upper",
        "dc_20_lower", "dc_20_upper", "dc_55_lower", "dc_55_upper",
        "year_month", "date_key"
    ]
    
    features = []
    for col in df.columns:
        if col in exclude_exact:
            continue
        if any(col.startswith(p) for p in exclude_prefixes):
            continue
        features.append(col)
    return features


def build_base_signal(df: pd.DataFrame) -> np.ndarray:
    """Define la señal base sobre la que aplicaremos el filtro ML para FTNT."""
    vol_ratio = df["vol_ratio"] > 1.2 if "vol_ratio" in df.columns else pd.Series(True, index=df.index)
    bb_pct    = df["bb_pct"] < 0.85 if "bb_pct" in df.columns else pd.Series(True, index=df.index)
    macd_hist = df["macd_hist"] > 0 if "macd_hist" in df.columns else pd.Series(True, index=df.index)
    return (vol_ratio & bb_pct & macd_hist).fillna(False).astype(bool).values


def main():
    console.print(Panel(
        "[bold]Filtro de Machine Learning: Entrenamiento del Meta-Model[/bold]\n\n"
        "Algoritmo: [cyan]Random Forest Classifier (Robust Config)[/cyan]\n"
        "Objetivo: Predecir probabilidad de éxito (TP 1.0% / SL 0.5% en 8h)\n"
        "División Temporal (Walk-Forward):\n"
        "  • Entrenamiento: Jun 2023 – Jun 2025 (24 meses)\n"
        "  • Test (Out-of-Sample): Jul 2025 – Jun 2026 (12 meses)",
        title="FASE 7 — Entrenamiento ML",
        border_style="purple"
    ))

    # 1. Cargar datos
    console.print("Cargando dataset...")
    df = pd.read_parquet(MASTER_PATH).sort_values("timestamp").reset_index(drop=True)
    n_total = len(df)
    console.print(f"  {n_total:,} velas en total")

    # 2. Generar entradas de la estrategia base (M1) con no-solapamiento
    console.print("Generando entradas de la estrategia base (M1)...")
    base_mask = build_base_signal(df)
    all_valid = np.where(base_mask & df["target_tp_sl_2to1"].notna())[0]
    
    entry_idx = []
    last_exit = -1
    for idx in all_valid:
        if idx > last_exit:
            entry_idx.append(idx)
            last_exit = idx + COOLDOWN_BARS - 1
    entry_idx = np.array(entry_idx)
    
    console.print(f"  {len(entry_idx):,} operaciones detectadas")

    # 3. Preparar Dataset
    all_features = get_all_features(df)
    
    # Rellenar nulos para scikit-learn
    from pandas.api.types import is_numeric_dtype
    df_filled = df.copy()
    for col in all_features:
        if not is_numeric_dtype(df_filled[col]):
            df_filled[col] = df_filled[col].astype("category").cat.codes
        else:
            df_filled[col] = df_filled[col].fillna(0.0)

    X = df_filled.loc[entry_idx, all_features].copy()
    y = df_filled.loc[entry_idx, "target_tp_sl_2to1"].values.astype(int)

    timestamps = pd.to_datetime(df.loc[entry_idx, "timestamp"])
    n_samples = len(entry_idx)
    split_idx = int(n_samples * 0.70)
    
    train_mask = np.arange(n_samples) < split_idx
    test_mask  = np.arange(n_samples) >= split_idx

    X_train, y_train = X.iloc[:split_idx], y[:split_idx]
    X_test, y_test   = X.iloc[split_idx:], y[split_idx:]

    console.print(f"  Entrenamiento: {len(X_train)} ops ({y_train.mean()*100:.1f}% tasa de acierto base)")
    console.print(f"  Test:          {len(X_test)} ops ({y_test.mean()*100:.1f}% tasa de acierto base)")

    if len(X_train) < 50:
        console.print("[red]Error: Muy pocos datos para entrenar el modelo.[/red]")
        return

    # 4. Selección de Variables (Análisis f_classif)
    console.print(f"Seleccionando las top {K_FEATURES} variables más predictivas...")
    selector = SelectKBest(score_func=f_classif, k=K_FEATURES)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel  = selector.transform(X_test)
    
    selected_features = [all_features[i] for i in selector.get_support(indices=True)]

    # 5. Entrenar Random Forest
    console.print("Entrenando Random Forest con regularización fuerte...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=2,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_sel, y_train)

    # 6. Evaluación
    train_probs = model.predict_proba(X_train_sel)[:, 1]
    test_probs  = model.predict_proba(X_test_sel)[:, 1]

    train_auc = roc_auc_score(y_train, train_probs)
    test_auc  = roc_auc_score(y_test, test_probs)

    console.print(f"\n[bold]Métricas de Clasificación:[/bold]")
    console.print(f"  Train AUC-ROC: [green]{train_auc:.4f}[/green]")
    console.print(f"  Test AUC-ROC:  [bold green]{test_auc:.4f}[/bold green]  (Out-of-sample)")

    # Mostrar variables seleccionadas e importancia
    importances = pd.DataFrame({
        "feature": selected_features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    imp_table = Table(title="Predictores Clave Seleccionados", border_style="purple")
    imp_table.add_column("Variable", style="cyan")
    imp_table.add_column("Importancia Relativa", justify="right")
    
    for _, r in importances.head(10).iterrows():
        imp_table.add_row(r["feature"], f"{r['importance']:.4f}")
    console.print(imp_table)

    # Guardar modelo, selector y features
    joblib.dump(model, MODEL_DIR / "meta_label_rf.pkl")
    joblib.dump(selector, MODEL_DIR / "feature_selector.pkl")
    
    with open(MODEL_DIR / "model_features.json", "w") as f:
        json.dump(selected_features, f)
        
    console.print(f"\n[bold green]✓ Modelo robusto guardado en data/models/meta_label_rf.pkl[/bold green]")
    console.print(f"[bold green]✓ Lista de features guardada en data/models/model_features.json[/bold green]")


if __name__ == "__main__":
    main()
