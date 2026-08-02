import os
import json
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

CHILE_TZ = ZoneInfo("America/Santiago")

_MODEL_DIR = os.path.dirname(__file__)
_MODEL_PATH = os.path.join(_MODEL_DIR, 'ml_model.joblib')
_FEATURES_PATH = os.path.join(os.path.dirname(_MODEL_DIR), 'data', 'models', 'model_features.json')

_cached_ml_packs = {}

def load_ml_features():
    if os.path.exists(_FEATURES_PATH):
        try:
            with open(_FEATURES_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return ["vwap", "ema_35", "rsi_14", "ema_9_accel", "ema_12_accel", "ema_20_accel", "ema_21_accel", "rsi_7", "macd_hist_slope", "roc_10", "roc_20", "bb_squeeze", "dc_55_width", "hv_20", "obv", "obv_ema_20", "obv_above_ema", "vol_sma_20", "vol_ratio", "entry_price"]

FEATURE_NAMES = load_ml_features()

def load_ml_model(profile=1):
    if profile in _cached_ml_packs:
        return _cached_ml_packs[profile]

    model_path = os.path.join(_MODEL_DIR, f'ml_model_{profile}.joblib')
    if not os.path.exists(model_path):
        model_path = _MODEL_PATH

    if os.path.exists(model_path):
        try:
            pack = joblib.load(model_path)
            _cached_ml_packs[profile] = pack
            return pack
        except Exception as e:
            print(f"Error al cargar modelo ML Perfil {profile}: {e}")
            return None
    return None

def extract_features(candle):
    """Extrae el vector de predictores técnicos y temporales (Año, Mes, Semana ISO, Día, Día Sem, Hora, Minuto) para el modelo ML de FTNT."""
    close = float(candle.get('close', 1.0))
    if close <= 0: close = 1.0
    
    high = float(candle.get('high') or close)
    low = float(candle.get('low') or close)
    vol = float(candle.get('volume') or 0.0)
    vwap = float(candle.get('vwap') or close)

    ema9 = float(candle.get('ema9') or close)
    ema21 = float(candle.get('ema21') or close)
    ema35 = float(candle.get('ema35') or close)
    rsi14 = float(candle.get('rsi14') or 50.0)
    macd_hist = float(candle.get('macd_hist') or 0.0)

    # Extraer las 7 variables temporales ML
    ts_ms = candle.get('time') or candle.get('timestamp') or 0
    if ts_ms:
        if isinstance(ts_ms, (int, float)):
            dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=CHILE_TZ)
        else:
            dt = datetime.now(CHILE_TZ)
        iso_year, iso_week, iso_day = dt.isocalendar()
        year_val = candle.get('year_ml') or dt.year
        month_val = candle.get('month_ml') or dt.month
        week_val = candle.get('week_ml') or iso_week
        day_val = candle.get('day_ml') or dt.day
        day_of_week_val = dt.weekday()
        hour_val = candle.get('hour_ml') if candle.get('hour_ml') is not None else dt.hour
        minute_val = candle.get('minute_ml') if candle.get('minute_ml') is not None else dt.minute
    else:
        year_val, month_val, week_val, day_val, day_of_week_val, hour_val, minute_val = 2026, 8, 31, 1, 5, 12, 0

    feats = [year_val, month_val, week_val, day_val, day_of_week_val, hour_val, minute_val]
    for fname in FEATURE_NAMES:
        if fname == 'vwap': feats.append(vwap)
        elif fname == 'ema_35': feats.append(ema35)
        elif fname == 'rsi_14': feats.append(rsi14)
        elif fname == 'ema_9_accel': feats.append(float(candle.get('ema9_slope') or 0.0))
        elif fname == 'ema_12_accel': feats.append(float(candle.get('ema9_slope_pct') or 0.0))
        elif fname == 'ema_20_accel': feats.append(float(candle.get('ema21_slope') or 0.0))
        elif fname == 'ema_21_accel': feats.append(float(candle.get('ema21_slope_pct') or 0.0))
        elif fname == 'rsi_7': feats.append(rsi14)
        elif fname == 'macd_hist_slope': feats.append(macd_hist)
        elif fname == 'roc_10': feats.append(0.0)
        elif fname == 'roc_20': feats.append(0.0)
        elif fname == 'bb_squeeze': feats.append(0)
        elif fname == 'dc_55_width': feats.append(0.0)
        elif fname == 'hv_20': feats.append(0.0)
        elif fname == 'obv': feats.append(vol)
        elif fname == 'obv_ema_20': feats.append(vol)
        elif fname == 'obv_above_ema': feats.append(1)
        elif fname == 'vol_sma_20': feats.append(vol)
        elif fname == 'vol_ratio': feats.append(1.5)
        elif fname == 'entry_price': feats.append(close)
        else: feats.append(0.0)
        
    return np.nan_to_num(feats)

def get_ml_threshold(profile=1):
    try:
        import database
        settings = database.get_settings()
        threshold_key = f'ml_threshold_pct_{profile}'
        return float(settings.get(threshold_key, settings.get('ml_threshold_pct', 50.0))) / 100.0
    except Exception:
        return 0.50

def predict_candle(candle, profile=1, threshold=None):
    """Predice si el filtro ML aprueba (SI / NO) la entrada y retorna la confianza."""
    ml_pack = load_ml_model(profile)
    if not ml_pack:
        return True, 50.0

    if isinstance(ml_pack, dict):
        model = ml_pack.get('model', ml_pack)
    else:
        model = ml_pack

    if threshold is None:
        threshold = get_ml_threshold(profile)

    feats = np.array([extract_features(candle)])
    try:
        probas = model.predict_proba(feats)[0]
        prob_win = float(probas[1] * 100.0)
        approve = bool(prob_win >= (threshold * 100.0 if threshold <= 1.0 else threshold))
        return approve, round(prob_win, 1)
    except Exception as e:
        print(f"Error en predicción ML Perfil {profile}: {e}")
        return True, 50.0

def predict_candles_batch(candles, profile=1, threshold=None):
    """Predice en lote una lista de velas de compra de forma vectorizada."""
    ml_pack = load_ml_model(profile)
    if not ml_pack or not candles:
        return [(True, 50.0) for _ in candles]

    model = ml_pack.get('model', ml_pack) if isinstance(ml_pack, dict) else ml_pack
    if threshold is None:
        threshold = get_ml_threshold(profile)

    thresh_val = threshold * 100.0 if threshold <= 1.0 else threshold

    feats_list = [extract_features(c) if c else np.zeros(len(FEATURE_NAMES)) for c in candles]
    X = np.array(feats_list)

    try:
        probas = model.predict_proba(X)[:, 1] * 100.0
        approves = probas >= thresh_val
        return [(bool(appr), round(float(p), 1)) for appr, p in zip(approves, probas)]
    except Exception as e:
        print(f"Error en predicción ML en lote Perfil {profile}: {e}")
        return [(True, 50.0) for _ in candles]

def train_ml_model(klines_by_ts, trade_pairs, profile=1):
    """Hook de entrenamiento ML."""
    pass
