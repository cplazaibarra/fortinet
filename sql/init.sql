-- Inicialización de la base de datos fortinet_db para el robot de trading

CREATE TABLE IF NOT EXISTS ohlcv_15m (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(10) NOT NULL DEFAULT 'FTNT',
    open NUMERIC(12, 4) NOT NULL,
    high NUMERIC(12, 4) NOT NULL,
    low NUMERIC(12, 4) NOT NULL,
    close NUMERIC(12, 4) NOT NULL,
    volume NUMERIC(16, 4) NOT NULL,
    trade_count INTEGER,
    vwap NUMERIC(12, 4),
    
    -- Indicadores Técnicos (Mismos componentes que la BD de BTC)
    ema_9 NUMERIC(12, 4),
    slope_ema9_pct NUMERIC(10, 4),
    ema_21 NUMERIC(12, 4),
    slope_ema21_pct NUMERIC(10, 4),
    ema_35 NUMERIC(12, 4),
    slope_ema35_pct NUMERIC(10, 4),
    ema_50 NUMERIC(12, 4),
    slope_ema50_pct NUMERIC(10, 4),
    ema_100 NUMERIC(12, 4),
    slope_ema100_pct NUMERIC(10, 4),
    ema_200 NUMERIC(12, 4),
    slope_ema200_pct NUMERIC(10, 4),
    
    rsi_14 NUMERIC(8, 4),
    macd NUMERIC(12, 4),
    macd_signal NUMERIC(12, 4),
    macd_hist NUMERIC(12, 4),
    atr_14 NUMERIC(12, 4),
    
    PRIMARY KEY (symbol, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_15m_timestamp ON ohlcv_15m (timestamp DESC);
