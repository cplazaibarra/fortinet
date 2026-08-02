def calculate_ema(prices, period):
    """
    Calcula el EMA para una lista de precios y un período determinado.
    """
    if len(prices) < period:
        return [None] * len(prices)
    
    ema_list = [None] * len(prices)
    sma = sum(prices[:period]) / period
    ema_list[period - 1] = sma
    
    multiplier = 2.0 / (period + 1)
    for i in range(period, len(prices)):
        ema_list[i] = (prices[i] - ema_list[i - 1]) * multiplier + ema_list[i - 1]
        
    return ema_list

def calculate_slopes(values, period=1):
    """
    Calcula la pendiente como la diferencia entre el valor actual y el anterior.
    """
    slopes = [None] * len(values)
    for i in range(period, len(values)):
        if values[i] is not None and values[i - period] is not None:
            slopes[i] = values[i] - values[i - period]
    return slopes

def calculate_slopes_pct(values, period=1):
    """
    Calcula la pendiente porcentual.
    """
    slopes_pct = [None] * len(values)
    for i in range(period, len(values)):
        if values[i] is not None and values[i - period] is not None and values[i - period] != 0:
            slopes_pct[i] = ((values[i] - values[i - period]) / values[i - period]) * 100.0
    return slopes_pct

def calculate_rsi(prices, period=14):
    """
    Calcula el Relative Strength Index (RSI).
    """
    if len(prices) <= period:
        return [None] * len(prices)
    rsi_list = [None] * len(prices)
    gains = []
    losses = []
    for i in range(1, len(prices)):
        change = prices[i] - prices[i - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    if avg_loss == 0:
        rsi_list[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi_list[period] = 100.0 - (100.0 / (1.0 + rs))
        
    for i in range(period + 1, len(prices)):
        gain = gains[i - 1]
        loss = losses[i - 1]
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        if avg_loss == 0:
            rsi_list[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_list[i] = 100.0 - (100.0 / (1.0 + rs))
    return rsi_list

def calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
    """
    Calcula la línea MACD, línea de Señal y el Histograma.
    """
    if len(prices) < slow_period:
        return [None] * len(prices), [None] * len(prices), [None] * len(prices)
    
    fast_ema = calculate_ema(prices, fast_period)
    slow_ema = calculate_ema(prices, slow_period)
    
    macd_line = [None] * len(prices)
    for i in range(slow_period - 1, len(prices)):
        if fast_ema[i] is not None and slow_ema[i] is not None:
            macd_line[i] = fast_ema[i] - slow_ema[i]
            
    valid_macd_vals = [v for v in macd_line if v is not None]
    signal_line_valid = calculate_ema(valid_macd_vals, signal_period)
    
    signal_line = [None] * len(prices)
    macd_hist = [None] * len(prices)
    
    offset = len(prices) - len(valid_macd_vals)
    for idx, sig_val in enumerate(signal_line_valid):
        orig_idx = offset + idx
        signal_line[orig_idx] = sig_val
        if macd_line[orig_idx] is not None and sig_val is not None:
            macd_hist[orig_idx] = macd_line[orig_idx] - sig_val
            
    return macd_line, signal_line, macd_hist

def calculate_atr(klines, period=14):
    """
    Calcula el Average True Range (ATR).
    """
    if len(klines) <= period:
        return [None] * len(klines)
    tr_list = [None] * len(klines)
    for i in range(1, len(klines)):
        h = klines[i]['high']
        l = klines[i]['low']
        prev_c = klines[i - 1]['close']
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        tr_list[i] = tr
        
    atr_list = [None] * len(klines)
    valid_trs = [t for t in tr_list if t is not None]
    if len(valid_trs) < period:
        return atr_list
        
    first_atr = sum(valid_trs[:period]) / period
    atr_list[period] = first_atr
    
    for i in range(period + 1, len(klines)):
        tr_val = tr_list[i]
        if tr_val is not None and atr_list[i - 1] is not None:
            atr_list[i] = (atr_list[i - 1] * (period - 1) + tr_val) / period
            
    return atr_list

def add_indicators_to_klines(klines):
    """
    Recibe una lista de diccionarios de velas (klines) y les añade todas las métricas de trading.
    """
    if not klines:
        return []
    
    prices = [k['close'] for k in klines]
    
    periods = [9, 21, 35, 50, 100, 200]
    emas = {}
    slopes = {}
    slopes_pct = {}
    
    for p in periods:
        emas[p] = calculate_ema(prices, p)
        slopes[p] = calculate_slopes(emas[p], period=1)
        slopes_pct[p] = calculate_slopes_pct(emas[p], period=1)
        
    rsi14 = calculate_rsi(prices, 14)
    macd, macd_signal, macd_hist = calculate_macd(prices)
    atr14 = calculate_atr(klines, 14)
    
    for i in range(len(klines)):
        for p in periods:
            klines[i][f'ema{p}'] = emas[p][i]
            klines[i][f'ema{p}_slope'] = slopes[p][i]
            klines[i][f'ema{p}_slope_pct'] = slopes_pct[p][i]
            
        klines[i]['rsi14'] = rsi14[i]
        klines[i]['macd'] = macd[i]
        klines[i]['macd_signal'] = macd_signal[i]
        klines[i]['macd_hist'] = macd_hist[i]
        klines[i]['atr14'] = atr14[i]
            
    return klines
