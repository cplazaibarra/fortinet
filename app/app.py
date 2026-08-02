from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import threading
import json
import time
import os
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

CHILE_TZ = ZoneInfo("America/Santiago")

# Importar módulos del proyecto
import database
from alpaca_client import AlpacaClient
from binance_client import BinanceClient
import ml_engine
import indicators
import sync_worker

app = Flask(__name__)
app.secret_key = "btc_machine_super_secret_key"

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# Hilo del Bot de Trading
bot_thread = None

def get_variable_value(candle, var_name, klines=None):
    """Obtiene el valor de una variable de la vela (ej. 'close', 'ema9', 'ema9_slope_pct', 'vol_ratio', 'bb_pct', 'tp_pct')."""
    val = candle.get(var_name)
    if val is not None:
        return float(val)

    # Cálculo dinámico con historial para variables compuestas
    if var_name == 'vol_ratio':
        vol = float(candle.get('volume', 0.0))
        if klines and len(klines) >= 5:
            sample = klines[-20:] if len(klines) >= 20 else klines
            sma_vol = float(np.mean([float(k.get('volume', 0.0)) for k in sample]))
        else:
            sma_vol = float(candle.get('vol_sma_20') or vol or 1.0)
        return float(vol / sma_vol) if sma_vol > 0 else 1.0
    elif var_name == 'bb_pct':
        close = float(candle.get('close', 0.0))
        if klines and len(klines) >= 5:
            sample = klines[-20:] if len(klines) >= 20 else klines
            closes = [float(k.get('close', close)) for k in sample]
            mean_c = float(np.mean(closes))
            std_c = float(np.std(closes))
            if std_c > 0:
                bb_upper = mean_c + (2.0 * std_c)
                bb_lower = mean_c - (2.0 * std_c)
                denom = bb_upper - bb_lower
                return float((close - bb_lower) / denom) if denom > 0 else 0.5
        ema20 = float(candle.get('ema21') or candle.get('ema9') or close)
        atr14 = float(candle.get('atr14') or (close * 0.015))
        bb_upper = ema20 + (atr14 * 2.0)
        bb_lower = ema20 - (atr14 * 2.0)
        denom = bb_upper - bb_lower
        return float((close - bb_lower) / denom) if denom > 0 else 0.5
    elif var_name in ['tp_pct', 'pnl_pct']:
        entry_p = float(candle.get('buy_price', candle.get('close', 1.0)))
        close = float(candle.get('close', entry_p))
        return float(((close - entry_p) / entry_p) * 100.0)
    elif var_name == 'trailing_drop_pct':
        peak_p = float(candle.get('peak_price', candle.get('close', 1.0)))
        close = float(candle.get('close', peak_p))
        return float(((close - peak_p) / peak_p) * 100.0) if peak_p > 0 else 0.0

    return None

def format_rule_description(rule):
    """Genera una descripción legible de una regla técnica cumplida."""
    op_map = {
        'gt': '>',
        'lt': '<',
        'gte': '>=',
        'lte': '<=',
        'cross_above': 'Cruza por encima de',
        'cross_below': 'Cruza por debajo de'
    }
    var_map = {
        'trailing_drop_pct': '% Caída desde Máximo (Trailing Stop)'
    }
    v1_raw = rule['var1']
    v1 = var_map.get(v1_raw, v1_raw.upper())
    op_str = op_map.get(rule['op'], rule['op'])
    v2 = rule['var2'].upper() if rule['compare_type'] == 'indicator' else str(rule['val2'])
    return f"{v1} {op_str} {v2}"

def evaluate_rules(klines, rules, is_exit=False):
    """
    Evalúa una lista de reglas de estrategia sobre los datos de las velas.
    klines[-1] es la vela actual, klines[-2] es la vela anterior.
    Retorna tuple (passed, reason_text).
    """
    if not rules:
        return False, "Sin reglas configuradas"
        
    if len(klines) < 3:
        return False, "Velas insuficientes"
        
    c0 = klines[-1] # Vela actual
    c1 = klines[-2] # Vela anterior (para crossovers)
    
    results = []
    matched_descriptions = []
    
    for rule in rules:
        var1 = rule['var1']
        op = rule['op']
        compare_type = rule['compare_type']
        var2 = rule['var2']
        val2_raw = rule.get('val2')
        val2 = float(val2_raw) if (val2_raw is not None and str(val2_raw).strip() != '') else 0.0
        
        # Obtener valores actuales (c0)
        v1_c0 = get_variable_value(c0, var1, klines=klines)
        v2_c0 = get_variable_value(c0, var2, klines=klines) if compare_type == 'indicator' else val2
        
        # Obtener valores anteriores (c1)
        v1_c1 = get_variable_value(c1, var1, klines=klines[:-1])
        v2_c1 = get_variable_value(c1, var2, klines=klines[:-1]) if compare_type == 'indicator' else val2
        
        if v1_c0 is None or v2_c0 is None:
            results.append(False)
            continue
            
        rule_passed = False
        
        # Manejo especial para Trailing Stop Loss (% Caída desde Máximo)
        if var1 == 'trailing_drop_pct':
            drop_mag = abs(v1_c0)
            target_drop = abs(val2)
            rule_passed = (drop_mag >= target_drop) if op in ['gte', 'gt', 'lte', 'lt'] else False
        elif op == 'gt': # Mayor que
            rule_passed = v1_c0 > v2_c0
        elif op == 'lt': # Menor que
            rule_passed = v1_c0 < v2_c0
        elif op == 'gte': # Mayor o igual que
            rule_passed = v1_c0 >= v2_c0
        elif op == 'lte': # Menor o igual que
            rule_passed = v1_c0 <= v2_c0
        elif op == 'cross_above': # Cruza por encima de
            if v1_c1 is not None and v2_c1 is not None:
                rule_passed = (v1_c1 <= v2_c1) and (v1_c0 > v2_c0)
        elif op == 'cross_below': # Cruza por debajo de
            if v1_c1 is not None and v2_c1 is not None:
                rule_passed = (v1_c1 >= v2_c1) and (v1_c0 < v2_c0)
                
        results.append(rule_passed)
        if rule_passed:
            matched_descriptions.append(format_rule_description(rule))
        
    if is_exit:
        # Estrategia de Salida: OR (se cumple al menos una)
        passed = any(results)
        reason = f"Regla de Salida [{', '.join(matched_descriptions)}]" if passed else "Sin activación de salida"
        return passed, reason
    else:
        # Estrategia de Entrada: Doble Disparador (Cruce Inicial OR Re-Entrada en Tendencia)
        c0 = klines[-1]
        c1 = klines[-2]
        macd_c0 = get_variable_value(c0, 'macd', klines=klines) or 0
        sig_c0 = get_variable_value(c0, 'macd_signal', klines=klines) or 0
        macd_c1 = get_variable_value(c1, 'macd', klines=klines[:-1]) or 0
        sig_c1 = get_variable_value(c1, 'macd_signal', klines=klines[:-1]) or 0
        rsi_c0 = get_variable_value(c0, 'rsi14', klines=klines) or 50
        rsi_c1 = get_variable_value(c1, 'rsi14', klines=klines[:-1]) or 50
        close_c0 = get_variable_value(c0, 'close', klines=klines) or 0
        open_c0 = get_variable_value(c0, 'open', klines=klines) or 0
        ema9_c0 = get_variable_value(c0, 'ema9', klines=klines) or 0
        ema21_c0 = get_variable_value(c0, 'ema21', klines=klines) or 0

        # Disparador 1: Cruce Inicial MACD Line por encima de Signal + RSI > 45
        trig1 = (macd_c1 <= sig_c1) and (macd_c0 > sig_c0) and (rsi_c0 > 45.0)

        # Disparador 2: Re-Entrada por Impulso en Tendencia Alcista (MACD > Signal + EMA9 > EMA21 + RSI Cruza por encima de 50.0)
        trig2 = (macd_c0 > sig_c0) and (ema9_c0 > ema21_c0) and (rsi_c1 <= 50.0) and (rsi_c0 > 50.0)

        # Disparador 3: Ruptura por Impulso Alcista (MACD > Signal + Precio > EMA9 > EMA21 + Vela Verde Impulso + RSI > 48.0)
        trig3 = (macd_c0 > sig_c0) and (close_c0 > ema9_c0) and (ema9_c0 > ema21_c0) and (close_c0 > open_c0 * 1.003) and (rsi_c0 > 48.0)

        if trig1:
            return True, "Disparador 1: Cruce Inicial MACD + RSI > 45"
        elif trig2:
            return True, "Disparador 2: Re-Entrada por Impulso en Tendencia (MACD > Signal + EMA9 > EMA21 + RSI Cruza 50)"
        elif trig3:
            return True, "Disparador 3: Ruptura por Impulso Alcista (MACD > Signal + Precio > EMA9 > EMA21 + Impulso Verde + RSI > 48)"
        else:
            return False, "No se cumplen las reglas de entrada (RAMA 1, RAMA 2 o RAMA 3)"

def get_open_position(min_usdt_threshold=2.0):
    """Busca si hay una operación de compra abierta (que no tenga venta asociada).
    Si el valor acumulado en USDT de la posición es menor a $2.0 USDT (polvo/residuo),
    se considera como NINGUNA posición abierta para permitir nuevas entradas."""
    trades = database.get_trades()
    buys = {}
    sells = {}
    for t in trades:
        tg = t.get('trade_group')
        if tg is not None:
            if t.get('type') == 'BUY':
                buys[tg] = t
            elif t.get('type') == 'SELL':
                sells[tg] = t
                
    for tg, buy_trade in buys.items():
        if tg not in sells:
            usdt_val = float(buy_trade.get('usdt_value', 0.0))
            if usdt_val < min_usdt_threshold:
                continue
            return buy_trade
    return None

def bot_worker():
    """Hilo secundario que ejecuta el bot de trading en segundo plano."""
    print("Hilo secundario del Bot de Trading iniciado.")
    database.add_log("Hilo de trading del bot iniciado en segundo plano", "INFO")
    
    last_processed_entry_ts = {}
    
    while True:
        try:
            # Obtener configuraciones actualizadas
            settings = database.get_settings()
            
            if settings.get('bot_active') != 'true':
                time.sleep(10)
                continue
                
            # Inicializar cliente de Binance
            api_key = settings.get('binance_api_key', '')
            api_secret = settings.get('binance_api_secret', '')
            use_testnet = settings.get('binance_testnet', 'true') == 'true'
            
            client = BinanceClient(api_key=api_key, api_secret=api_secret, use_testnet=use_testnet)
            
            # 1. Obtener perfil de estrategia activo (1 o 2)
            active_profile = int(settings.get('active_strategy_profile', '1'))

            # 2. Obtener velas de 15m almacenadas en PostgreSQL
            klines = database.get_candles_15m(limit=300)
            if not klines:
                time.sleep(15)
                continue
                
            current_candle = klines[-1]
            current_price = current_candle['close']
            current_time_ms = current_candle['time']
            
            # 3. Comprobar si hay una posición abierta
            open_position = get_open_position()
            
            if open_position:
                # Buscar reglas de salida del perfil activo (Evaluación en TIEMPO REAL cada 15s)
                exit_rules = database.get_strategy('exit', profile=active_profile)
                
                # Calcular precio máximo alcanzado durante la posición activa y % caída desde el pico
                trade_start_ts = open_position.get('timestamp') or open_position.get('buy_time')
                peak_price = float(open_position.get('price', current_price))
                if trade_start_ts:
                    for k in klines:
                        if k['time'] >= trade_start_ts:
                            peak_price = max(peak_price, float(k.get('high', k['close'])))
                
                current_candle['buy_price'] = open_position['price']
                current_candle['peak_price'] = peak_price
                current_candle['pnl_pct'] = ((current_price - open_position['price']) / open_position['price']) * 100.0
                current_candle['trailing_drop_pct'] = ((current_price - peak_price) / peak_price) * 100.0 if peak_price > 0 else 0.0
                
                # Evaluar salida en tiempo real
                exit_passed, exit_reason = evaluate_rules(klines, exit_rules, is_exit=True)
                if exit_passed:
                    print(f"--- ¡CONDICION DE SALIDA (PERFIL {active_profile}) CUMPLIDA! ({exit_reason}) ---")
                    execute_sell(client, current_price, open_position, reason=f"P{active_profile}: {exit_reason}")
            else:
                # ENTRADA: Evaluar ÚNICAMENTE con velas de 15m 100% CERRADAS
                now_ms = int(time.time() * 1000)
                closed_klines = [k for k in klines if (int(k['time']) + 900000) <= now_ms]
                
                if closed_klines:
                    last_closed_candle = closed_klines[-1]
                    closed_ts = int(last_closed_candle['time'])
                    
                    # Evitar re-evaluar repetidamente la misma vela cerrada
                    if last_processed_entry_ts.get(active_profile) != closed_ts:
                        entry_rules = database.get_strategy('entry', profile=active_profile)
                        
                        # Evaluar entrada sobre el historial de velas cerradas
                        entry_passed, entry_reason = evaluate_rules(closed_klines, entry_rules, is_exit=False)
                        if entry_passed:
                            # Consultar el filtro de Machine Learning del perfil activo sobre la vela cerrada
                            ml_approve, ml_conf = ml_engine.predict_candle(last_closed_candle, profile=active_profile)
                            if ml_approve:
                                full_entry_reason = f"P{active_profile}: {entry_reason} + ML P{active_profile} Aprobado ({ml_conf}% conf) [Vela Cerrada]"
                                print(f"--- ¡CONDICION DE ENTRADA Y FILTRO ML (PERFIL {active_profile}) APROBADOS EN VELA CERRADA! ({full_entry_reason}) ---")
                                database.add_log(f"Señal de COMPRA (Perfil {active_profile}) detectada en VELA CERRADA: {full_entry_reason}.", "INFO")
                                execute_buy(client, current_price, reason=full_entry_reason)
                            else:
                                filter_reason = f"P{active_profile}: {entry_reason} | Omitida por ML P{active_profile} ({ml_conf}% conf) [Vela Cerrada]"
                                database.add_log(f"Señal técnica Perfil {active_profile} detectada en vela cerrada ({entry_reason}) pero FILTRADA por ML Perfil {active_profile} ({ml_conf}% confianza). Compra omitida.", "WARNING")
                                
                                last_trade = database.get_last_trade()
                                if not last_trade or last_trade.get('type') != 'SIGNAL_REJECTED' or (now_ms - last_trade.get('timestamp', 0)) > 900000:
                                    trade_size_usd = float(settings.get('trade_size_usd', 100.0))
                                    amount_ftnt = round(trade_size_usd / current_price, 5) if current_price > 0 else 0.0
                                    database.add_trade('SIGNAL_REJECTED', closed_ts, current_price, amount_ftnt, trade_size_usd, trade_group=now_ms, mode='AUTO', reason=filter_reason)
                        
                        # Registrar esta vela cerrada como procesada para no duplicar en los siguientes chequeos de 15s
                        last_processed_entry_ts[active_profile] = closed_ts
                    
        except Exception as e:
            print(f"Error en bot_worker: {e}")
            try:
                database.add_log(f"Error en la ejecución del bot: {str(e)}", "ERROR")
            except Exception:
                pass
                
        time.sleep(15) # Revisar cada 15 segundos

def execute_buy(client, price, reason="Reglas de Indicadores Técnicos"):
    """Ejecuta una orden de compra (simulada o real)."""
    settings = database.get_settings()
    sim_mode = settings.get('simulation_mode') == 'true'
    trade_size_usd = float(settings.get('trade_size_usd', 100.0))
    
    # El trade group será el timestamp actual en milisegundos
    trade_group = int(time.time() * 1000)
    amount_ftnt = round(trade_size_usd / price, 5)
    if amount_ftnt <= 0:
        amount_ftnt = 0.00001
    
    if sim_mode:
        sim_balance = float(settings.get('simulation_balance', 10000.0))
        if sim_balance < trade_size_usd:
            database.add_log(f"Fallo al comprar: Saldo de simulación insuficiente ({sim_balance:.2f} USDT < {trade_size_usd} USDT)", "WARNING")
            return
            
        new_balance = sim_balance - trade_size_usd
        new_btc_balance = float(settings.get('simulation_btc_balance', 0.0)) + amount_ftnt
        
        database.set_setting('simulation_balance', new_balance)
        database.set_setting('simulation_btc_balance', new_btc_balance)
        
        database.add_trade('BUY', trade_group, price, amount_ftnt, trade_size_usd, trade_group=trade_group, mode='AUTO', reason=reason)
        database.add_log(f"COMPRA SIMULADA: Compra de {amount_ftnt:.6f} FTNT a ${price:.2f} USDT (Motivo: {reason})", "INFO")
    else:
        # Modo Real Testnet
        try:
            # Obtener balance real de USDT en testnet
            balances = client.get_balances()
            usdt_bal = balances.get('USDT', 0.0)
            if usdt_bal < trade_size_usd:
                database.add_log(f"Fallo en Testnet: Balance insuficiente en testnet ({usdt_bal:.2f} USDT < {trade_size_usd} USDT)", "WARNING")
                return
                
            # Colocar orden de compra a mercado
            order = client.place_market_order(symbol="FTNT", side="BUY", quantity=amount_ftnt)
            executed_qty = float(order.get('executedQty', amount_ftnt))
            cummulative_quote_qty = float(order.get('cummulativeQuoteQty', trade_size_usd))
            # Calcular precio medio de ejecución
            avg_price = cummulative_quote_qty / executed_qty if executed_qty > 0 else price
            
            database.add_trade('BUY', int(order['transactTime']), avg_price, executed_qty, cummulative_quote_qty, trade_group=trade_group, mode='AUTO', reason=reason)
            database.add_log(f"COMPRA TESTNET: Compra de {executed_qty:.6f} FTNT a ${avg_price:.2f} USDT (Motivo: {reason})", "INFO")
        except Exception as e:
            database.add_log(f"Error al ejecutar orden de compra en Testnet: {e}", "ERROR")

def execute_sell(client, price, open_position, reason="Reglas de Indicadores Técnicos"):
    """Ejecuta una orden de venta para cerrar una posición."""
    settings = database.get_settings()
    sim_mode = settings.get('simulation_mode') == 'true'
    trade_group = open_position['trade_group']
    amount_ftnt = open_position['amount']
    
    if sim_mode:
        sim_btc_balance = float(settings.get('simulation_btc_balance', 0.0))
        if sim_btc_balance < amount_ftnt:
            amount_ftnt = sim_btc_balance # Ajustar a lo disponible por redondeos
            
        if amount_ftnt <= 0:
            database.add_log("Fallo al vender: Saldo de FTNT en simulación es 0", "WARNING")
            return
            
        usdt_value = amount_ftnt * price
        pnl = usdt_value - open_position['usdt_value']
        
        new_balance = float(settings.get('simulation_balance', 10000.0)) + usdt_value
        new_btc_balance = sim_btc_balance - amount_ftnt
        
        database.set_setting('simulation_balance', new_balance)
        database.set_setting('simulation_btc_balance', new_btc_balance)
        
        database.add_trade('SELL', int(time.time() * 1000), price, amount_ftnt, usdt_value, pnl=pnl, trade_group=trade_group, mode='AUTO', reason=reason)
        database.add_log(f"VENTA SIMULADA: Venta de {amount_ftnt:.6f} FTNT a ${price:.2f} USDT (Motivo: {reason}). Ganancia: ${pnl:.2f} USDT", "INFO")
    else:
        # Modo Real Testnet
        try:
            # Obtener balance real de FTNT en testnet
            balances = client.get_balances()
            btc_bal = balances.get('FTNT', 0.0)
            if btc_bal < amount_ftnt:
                amount_ftnt = btc_bal
                
            if amount_ftnt <= 0:
                database.add_log("Fallo en Testnet: No hay saldo FTNT disponible para vender", "WARNING")
                return
                
            # Colocar orden de venta a mercado
            order = client.place_market_order(symbol="FTNT", side="SELL", quantity=amount_ftnt)
            executed_qty = float(order.get('executedQty', amount_ftnt))
            cummulative_quote_qty = float(order.get('cummulativeQuoteQty', executed_qty * price))
            avg_price = cummulative_quote_qty / executed_qty if executed_qty > 0 else price
            
            pnl = cummulative_quote_qty - open_position['usdt_value']
            
            database.add_trade('SELL', int(order['transactTime']), avg_price, executed_qty, cummulative_quote_qty, pnl=pnl, trade_group=trade_group, mode='AUTO', reason=reason)
            database.add_log(f"VENTA TESTNET: Venta de {executed_qty:.6f} FTNT a ${avg_price:.2f} USDT (Motivo: {reason}). Ganancia: ${pnl:.2f} USDT", "INFO")
        except Exception as e:
            database.add_log(f"Error al ejecutar orden de venta en Testnet: {e}", "ERROR")

def get_current_balances():
    """Obtiene los balances actuales de USDT y FTNT (de Simulación o de Binance Testnet)."""
    settings = database.get_settings()
    sim_mode = settings.get('simulation_mode') == 'true'

    if sim_mode:
        usdt_bal = float(settings.get('simulation_balance', 10000.0))
        btc_bal = float(settings.get('simulation_btc_balance', 0.0))
    else:
        try:
            api_key = settings.get('binance_api_key', '')
            api_secret = settings.get('binance_api_secret', '')
            use_testnet = settings.get('binance_testnet', 'true') == 'true'
            client = BinanceClient(api_key=api_key, api_secret=api_secret, use_testnet=use_testnet)
            bals = client.get_balances()
            usdt_bal = float(bals.get('USDT', 0.0))
            btc_bal = float(bals.get('FTNT', 0.0))
        except Exception as e:
            print(f"Error consultando balances de Binance Testnet: {e}")
            usdt_bal = float(settings.get('simulation_balance', 10000.0))
            btc_bal = float(settings.get('simulation_btc_balance', 0.0))

    return {
        'usdt': round(usdt_bal, 2),
        'btc': round(btc_bal, 5),
        'simulation_mode': sim_mode
    }

# ================= RUTA WEB FLASK =================

@app.route('/')
@app.route('/dashboard')
def dashboard():
    settings = database.get_settings()
    trades = database.get_trades() or []
    balances = get_current_balances()
    
    # Identificar cuáles trade_groups de compras tienen ya su venta asociada
    sells_by_tg = set(t['trade_group'] for t in trades if t.get('type') == 'SELL' and t.get('trade_group') is not None)
    manual_tgs = set(t['trade_group'] for t in trades if t.get('mode') == 'MANUAL' and t.get('trade_group') is not None)

    # Formatear timestamps y evaluar si la posición está ACTIVA (en curso) o CERRADA y si fue MANUAL u AUTOMATICA
    for t in trades:
        if t.get('timestamp'):
            t['date_str'] = datetime.fromtimestamp(t['timestamp'] / 1000, tz=CHILE_TZ).strftime('%Y-%m-%d %H:%M:%S')

        if t.get('type') == 'BUY':
            if t.get('trade_group') and t.get('trade_group') not in sells_by_tg:
                t['is_active'] = True
            else:
                t['is_active'] = False
        else:
            t['is_active'] = False

        if t.get('mode') == 'MANUAL' or (t.get('trade_group') and t.get('trade_group') in manual_tgs):
            t['is_manual'] = True
        else:
            t['is_manual'] = False

    logs = database.get_logs(limit=30) or []
    for l in logs:
        if isinstance(l.get('timestamp'), (int, float)):
            l['timestamp'] = datetime.fromtimestamp(l['timestamp'] / 1000, tz=CHILE_TZ).strftime('%Y-%m-%d %H:%M:%S')
        
    return render_template('index.html', settings=settings, trades=trades, logs=logs, balances=balances, active_page='dashboard')

@app.route('/strategy/entry')
@app.route('/strategy/entry/<int:profile>')
def strategy_entry(profile=1):
    settings = database.get_settings()
    rules = database.get_strategy('entry', profile=profile)
    template = 'strategy_entry_2.html' if profile == 2 else 'strategy_entry.html'
    return render_template(template, rules=rules, settings=settings, profile=profile, active_page=f'strategy_entry_{profile}')

@app.route('/strategy/exit')
@app.route('/strategy/exit/<int:profile>')
def strategy_exit(profile=1):
    settings = database.get_settings()
    rules = database.get_strategy('exit', profile=profile)
    template = 'strategy_exit_2.html' if profile == 2 else 'strategy_exit.html'
    return render_template(template, rules=rules, settings=settings, profile=profile, active_page=f'strategy_exit_{profile}')

@app.route('/chart')
def chart_page():
    settings = database.get_settings()
    return render_template('chart.html', settings=settings, active_page='chart')

@app.route('/chart/3d')
def chart_3d():
    return redirect(url_for('chart_page'))

@app.route('/settings')
def settings_page():
    settings = database.get_settings()
    return render_template('settings.html', settings=settings, active_page='settings')

@app.route('/logs')
def logs_page():
    """Página dedicada para explorar los registros de actividad del sistema (Logs)."""
    settings = database.get_settings()
    limit = request.args.get('limit', default=20, type=int)
    if limit not in [20, 30, 60, 100]:
        limit = 20
    logs = database.get_logs(limit=limit) or []
    for l in logs:
        if isinstance(l.get('timestamp'), (int, float)):
            l['timestamp'] = datetime.fromtimestamp(l['timestamp'] / 1000, tz=CHILE_TZ).strftime('%Y-%m-%d %H:%M:%S')
        elif l.get('timestamp') and hasattr(l['timestamp'], 'strftime'):
            l['timestamp'] = l['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        else:
            l['timestamp'] = str(l.get('timestamp', ''))

    return render_template('logs.html', settings=settings, logs=logs, current_limit=limit, active_page='logs')

# ================= ACCIONES DE CONFIGURACION =================

@app.route('/settings/save', methods=['POST'])
def save_settings():
    candle_interval = request.form.get('candle_interval')
    simulation_mode = request.form.get('simulation_mode')
    simulation_balance = request.form.get('simulation_balance')
    simulation_btc_balance = request.form.get('simulation_btc_balance')
    trade_size_usd = request.form.get('trade_size_usd')
    
    database.set_settings({
        'candle_interval': candle_interval,
        'simulation_mode': simulation_mode,
        'simulation_balance': simulation_balance,
        'simulation_btc_balance': simulation_btc_balance,
        'trade_size_usd': trade_size_usd
    })
    
    flash("Configuración guardada correctamente.", "success")
    database.add_log(f"Configuración de parámetros actualizada (Intervalo: {candle_interval}, Modo Simulación: {simulation_mode})", "INFO")
    return redirect(url_for('settings_page'))

@app.route('/settings/save_keys', methods=['POST'])
def save_keys():
    api_key = request.form.get('binance_api_key')
    api_secret = request.form.get('binance_api_secret')
    
    database.set_settings({
        'binance_api_key': api_key,
        'binance_api_secret': api_secret
    })
    
    flash("Llaves API de Binance guardadas correctamente.", "success")
    database.add_log("API Keys de Binance actualizadas por el usuario", "INFO")
    return redirect(url_for('settings_page'))

@app.route('/settings/reset', methods=['POST'])
def reset_database():
    database.clear_logs_and_trades()
    flash("Base de datos y saldos reiniciados correctamente.", "success")
    return redirect(url_for('settings_page'))

@app.route('/bot/toggle', methods=['POST'])
def toggle_bot():
    settings = database.get_settings()
    current_state = settings.get('bot_active', 'false')
    
    new_state = 'true' if current_state == 'false' else 'false'
    database.set_setting('bot_active', new_state)
    
    msg = "Bot de trading INICIADO." if new_state == 'true' else "Bot de trading DETENIDO."
    flash(msg, "success")
    database.add_log(msg, "INFO")
    
    return redirect(url_for('dashboard'))

# ================= ENDPOINTS DE LA API =================

@app.route('/api/strategy/save/<strategy_type>', methods=['POST'])
def save_strategy_api(strategy_type):
    if strategy_type not in ['entry', 'exit']:
        return jsonify({'success': False, 'message': 'Tipo de estrategia inválido'})
    try:
        data = request.get_json() or {}
        rules = data.get('rules', [])
        profile = int(data.get('profile', request.args.get('profile', 1)))
        
        database.save_strategy(strategy_type, rules, profile=profile)
        database.add_log(f"Estrategia de {strategy_type} (Perfil {profile}) guardada con {len(rules)} reglas", "INFO")
        
        # Recalcular y guardar la simulación del perfil correspondiente en segundo plano
        def bg_update():
            try:
                update_persistent_simulation(profile=profile)
            except Exception as e:
                print(f"Advertencia al actualizar simulación P{profile} tras guardar estrategia: {e}")

        threading.Thread(target=bg_update, daemon=True).start()

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/bot/set_profile', methods=['POST'])
def set_active_profile_api():
    """Permite al usuario cambiar el perfil de estrategia activo para el bot en vivo."""
    try:
        data = request.get_json() or {}
        profile = int(data.get('profile', 1))
        if profile not in [1, 2]:
            return jsonify({'success': False, 'message': 'Perfil inválido'})
            
        database.set_setting('active_strategy_profile', str(profile))
        msg = f"Perfil de estrategia del Bot cambiado a: Perfil {profile}"
        database.add_log(msg, "INFO")
        flash(msg, "success")
        return jsonify({'success': True, 'active_profile': profile})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/test-keys', methods=['POST'])
def test_keys():
    try:
        data = request.get_json()
        api_key = data.get('api_key')
        api_secret = data.get('api_secret')
        
        client = BinanceClient(api_key=api_key, api_secret=api_secret, use_testnet=True)
        success, msg = client.test_api_keys()
        
        return jsonify({'success': success, 'message': msg})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/database')
def database_page():
    """Página para explorar la base de datos PostgreSQL de velas de 15 minutos."""
    settings = database.get_settings()
    return render_template('database_view.html', settings=settings, active_page='database_view')

@app.route('/simulation')
def simulation_page():
    """Página para explorar el historial de operaciones simuladas."""
    settings = database.get_settings()
    return render_template('simulation_view.html', settings=settings, active_page='simulation_view')

@app.route('/api/settings/trade-size', methods=['POST'])
def api_set_trade_size():
    """Actualiza la cantidad de dinero en USD a colocar en cada entrada."""
    try:
        data = request.get_json() or {}
        trade_size = float(data.get('trade_size_usd', 1000.0))
        if trade_size <= 0:
            return jsonify({'success': False, 'message': 'El monto por entrada debe ser mayor a 0 USD.'})
        
        database.set_setting('trade_size_usd', str(trade_size))
        database.add_log(f"Monto por entrada actualizado a ${trade_size:,.2f} USDT por el usuario", "INFO")
        return jsonify({'success': True, 'trade_size_usd': trade_size})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/settings/ml-threshold', methods=['POST'])
def api_set_ml_threshold():
    """Actualiza el umbral de confianza de ML (en %) configurado por el usuario y recalcula la simulación en segundo plano."""
    try:
        data = request.get_json() or {}
        profile = int(data.get('profile', 1))
        ml_thresh = float(data.get('ml_threshold_pct', 50.0))
        ml_thresh = max(10.0, min(95.0, ml_thresh))
        
        setting_key = f'ml_threshold_pct_{profile}' if profile == 2 else 'ml_threshold_pct'
        database.set_setting(setting_key, str(ml_thresh))
        if profile == 1:
            database.set_setting('ml_threshold_pct_1', str(ml_thresh))

        database.add_log(f"Umbral de Confianza ML Perfil {profile} actualizado a {ml_thresh:.1f}% por el usuario", "INFO")

        # Iniciar recálculo de simulación en segundo plano para respuesta instantánea
        def bg_update():
            try:
                update_persistent_simulation(profile=profile)
            except Exception as sim_err:
                print(f"Error actualizando simulación P{profile} en segundo plano: {sim_err}")

        threading.Thread(target=bg_update, daemon=True).start()

        return jsonify({'success': True, 'ml_threshold_pct': ml_thresh, 'profile': profile})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/trade/manual-buy', methods=['POST'])
def api_manual_buy():
    """Ejecuta una compra manual especificada por el usuario en USD."""
    try:
        data = request.get_json() or {}
        amount_usd = float(data.get('amount_usd', 10.0))
        if amount_usd <= 0:
            return jsonify({'success': False, 'message': 'El monto a comprar debe ser mayor a 0 USD.'})

        settings = database.get_settings()
        sim_mode = settings.get('simulation_mode') == 'true'
        api_key = settings.get('binance_api_key', '')
        api_secret = settings.get('binance_api_secret', '')
        use_testnet = settings.get('binance_testnet', 'true') == 'true'
        client = BinanceClient(api_key=api_key, api_secret=api_secret, use_testnet=use_testnet)

        # Obtener precio actual de FTNT
        klines = database.get_candles_15m(limit=5)
        if not klines:
            return jsonify({'success': False, 'message': 'No hay datos de precio disponibles.'})
        current_price = klines[-1]['close']

        # Verificar saldo suficiente
        if sim_mode:
            sim_balance = float(settings.get('simulation_balance', 10000.0))
            if sim_balance < amount_usd:
                return jsonify({'success': False, 'message': f'Saldo de simulación insuficiente ({sim_balance:.2f} USDT < {amount_usd:.2f} USDT)'})
        else:
            balances = client.get_balances()
            usdt_bal = balances.get('USDT', 0.0)
            if usdt_bal < amount_usd:
                return jsonify({'success': False, 'message': f'Balance en Testnet insuficiente ({usdt_bal:.2f} USDT < {amount_usd:.2f} USDT)'})

        # Ejecutar compra manual
        trade_group = int(time.time() * 1000)
        amount_ftnt = round(amount_usd / current_price, 5)
        if amount_ftnt <= 0:
            amount_ftnt = 0.00001

        if sim_mode:
            new_balance = sim_balance - amount_usd
            new_btc_balance = float(settings.get('simulation_btc_balance', 0.0)) + amount_ftnt
            database.set_setting('simulation_balance', str(new_balance))
            database.set_setting('simulation_btc_balance', str(new_btc_balance))
            database.add_trade('BUY', trade_group, current_price, amount_ftnt, amount_usd, trade_group=trade_group, mode='MANUAL')
            database.add_log(f"COMPRA MANUAL SIMULADA: Inversión de ${amount_usd:.2f} USDT ({amount_ftnt:.5f} FTNT) a ${current_price:,.2f} USDT", "INFO")
        else:
            order = client.place_market_order(symbol="FTNT", side="BUY", quantity=amount_ftnt)
            executed_qty = float(order.get('executedQty', amount_ftnt))
            cummulative_quote = float(order.get('cummulativeQuoteQty', amount_usd))
            avg_price = cummulative_quote / executed_qty if executed_qty > 0 else current_price
            database.add_trade('BUY', int(order['transactTime']), avg_price, executed_qty, cummulative_quote, trade_group=trade_group, mode='MANUAL')
            database.add_log(f"COMPRA MANUAL TESTNET: Compra de ${cummulative_quote:.2f} USDT ({executed_qty:.5f} FTNT) a ${avg_price:,.2f} USDT", "INFO")

        # Recalcular simulación en segundo plano (asíncrono para respuesta inmediata)
        try:
            threading.Thread(target=update_persistent_simulation, daemon=True).start()
        except Exception:
            pass

        return jsonify({
            'success': True,
            'message': f'Compra manual por ${amount_usd:.2f} USDT realizada exitosamente a ${current_price:,.2f} USDT.'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/trade/manual-sell-all', methods=['POST'])
def api_manual_sell_all():
    """Ejecuta la salida manual vendiendo el 100% de la posición/tenencia activa de FTNT."""
    try:
        settings = database.get_settings()
        sim_mode = settings.get('simulation_mode') == 'true'
        api_key = settings.get('binance_api_key', '')
        api_secret = settings.get('binance_api_secret', '')
        use_testnet = settings.get('binance_testnet', 'true') == 'true'
        client = BinanceClient(api_key=api_key, api_secret=api_secret, use_testnet=use_testnet)

        # Obtener precio actual de FTNT
        klines = database.get_candles_15m(limit=5)
        if not klines:
            return jsonify({'success': False, 'message': 'No hay datos de precio disponibles.'})
        current_price = klines[-1]['close']

        open_position = get_open_position()

        if sim_mode:
            btc_bal = float(settings.get('simulation_btc_balance', 0.0))
            if btc_bal <= 0:
                if open_position:
                    database.add_trade('SELL', int(time.time() * 1000), current_price, open_position.get('amount', 0.0), open_position.get('usdt_value', 0.0), pnl=0.0, trade_group=open_position['trade_group'], mode='MANUAL', reason='Cerrada por Venta Previa')
                    return jsonify({'success': True, 'message': 'La posición de FTNT ya había sido cerrada previamente.'})
                return jsonify({'success': False, 'message': 'No tienes saldo de FTNT en la posición actual.'})

            usdt_value = btc_bal * current_price
            pnl = (usdt_value - open_position['usdt_value']) if open_position else 0.0
            trade_group = open_position['trade_group'] if open_position else int(time.time() * 1000)

            new_balance = float(settings.get('simulation_balance', 10000.0)) + usdt_value
            database.set_setting('simulation_balance', str(new_balance))
            database.set_setting('simulation_btc_balance', '0.0')

            reason = "Venta Manual Total Directa (100% Posición)"
            database.add_trade('SELL', int(time.time() * 1000), current_price, btc_bal, usdt_value, pnl=pnl, trade_group=trade_group, mode='MANUAL', reason=reason)
            database.add_log(f"VENTA MANUAL TOTAL SIMULADA (100%): Venta de {btc_bal:.5f} FTNT a ${current_price:,.2f} USDT (+${usdt_value:.2f} USDT). PnL: ${pnl:+.2f} USDT", "INFO")
        else:
            balances = client.get_balances()
            btc_bal = float(balances.get('FTNT', 0.0))
            btc_bal = round(btc_bal, 5)
            if btc_bal <= 0.00001:
                if open_position:
                    database.add_trade('SELL', int(time.time() * 1000), current_price, open_position.get('amount', 0.0), open_position.get('usdt_value', 0.0), pnl=0.0, trade_group=open_position['trade_group'], mode='MANUAL', reason='Cerrada por Venta Previa Automática')
                    database.add_log("SALIDA MANUAL: La posición de FTNT ya había sido vendida automáticamente en Testnet por la Estrategia de Salida.", "INFO")
                    return jsonify({'success': True, 'message': 'La posición de FTNT ya había sido vendida automáticamente por las reglas de la Estrategia de Salida del bot.'})
                return jsonify({'success': False, 'message': 'No hay balance activo de FTNT disponible en la cuenta de Testnet.'})

            trade_group = open_position['trade_group'] if open_position else int(time.time() * 1000)
            order = client.place_market_order(symbol="FTNT", side="SELL", quantity=btc_bal)
            executed_qty = float(order.get('executedQty', btc_bal))
            cummulative_quote = float(order.get('cummulativeQuoteQty', btc_bal * current_price))
            avg_price = cummulative_quote / executed_qty if executed_qty > 0 else current_price
            pnl = (cummulative_quote - open_position['usdt_value']) if open_position else 0.0

            database.add_trade('SELL', int(order['transactTime']), avg_price, executed_qty, cummulative_quote, pnl=pnl, trade_group=trade_group, mode='MANUAL')
            database.add_log(f"VENTA MANUAL TOTAL TESTNET (100%): Venta de {executed_qty:.5f} FTNT a ${avg_price:,.2f} USDT (+${cummulative_quote:.2f} USDT). PnL: ${pnl:+.2f} USDT", "INFO")

        # Recalcular simulación en segundo plano (asíncrono para respuesta inmediata)
        try:
            threading.Thread(target=update_persistent_simulation, daemon=True).start()
        except Exception:
            pass

        return jsonify({
            'success': True,
            'message': f'Venta del 100% de la tenencia realizada exitosamente a ${current_price:,.2f} USDT.'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/database/candles')
def api_database_candles():
    """Retorna datos paginados de la tabla candles_15m de PostgreSQL."""
    try:
        page = request.args.get('page', type=int, default=1)
        per_page = request.args.get('per_page', type=int, default=25)
        data = database.get_candles_15m_paginated(page=page, per_page=per_page)
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/ml/retrain', methods=['POST'])
def api_ml_retrain():
    """Fuerza un re-entrenamiento del modelo ML desde la interfaz web."""
    try:
        data = request.get_json(silent=True) or {}
        profile = int(data.get('profile', request.args.get('profile', 1)))
        
        metrics = ml_engine.retrain_model(profile=profile)
        if metrics.get('status') == 'success':
            database.add_log(f"Modelo de Machine Learning Perfil {profile} re-entrenado manualmente con éxito", "INFO")
            return jsonify({'success': True, 'metrics': metrics, 'profile': profile})
        else:
            return jsonify({'success': False, 'message': metrics.get('message', 'Error al re-entrenar')})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/ml/predict', methods=['POST'])
def api_ml_predict():
    """Endpoint API independiente para consultar predicciones de Machine Learning para señales de compra."""
    try:
        data = request.get_json() or {}
        candle = data.get('candle', data)
        approve, confidence = ml_engine.predict_candle(candle)
        return jsonify({
            'success': True,
            'approve': approve,
            'confidence_pct': confidence,
            'recommendation': 'EJECUTAR_COMPRA' if approve else 'FILTRAR_OPERACION'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/logs/clear', methods=['POST'])
def clear_logs_api():
    """Limpia los registros de logs desde la interfaz web."""
    try:
        database.clear_logs()
        database.add_log("Registros de actividad (logs) limpiados por el usuario desde la plataforma web", "INFO")
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/simulation/run')
def api_simulation_run():
    """Retorna las operaciones simuladas guardadas en la tabla simulated_trades de PostgreSQL,
    filtrando por perfil de estrategia y rango de fechas si se especifica."""
    try:
        profile = request.args.get('profile', type=int, default=1)
        if profile not in [1, 2]: profile = 1
        
        start_ts = request.args.get('start_ts', type=int)
        end_ts = request.args.get('end_ts', type=int)
        page = request.args.get('page', type=int, default=1)
        per_page = request.args.get('per_page', type=int, default=50)

        # Asegurar que hay datos precalculados en PostgreSQL para este perfil
        if not database.get_saved_simulation_summary(profile=profile):
            update_persistent_simulation(profile=profile)

        # Calcular resumen directamente en SQL (correcto para cualquier rango de fechas y perfil)
        summary = database.get_simulated_summary_by_range(start_ts=start_ts, end_ts=end_ts, profile=profile)

        # Incluir resumen legible de reglas activas e umbral ML
        entry_rules = database.get_strategy('entry', profile=profile)
        exit_rules = database.get_strategy('exit', profile=profile)
        settings = database.get_settings()
        thresh_key = f'ml_threshold_pct_{profile}' if profile == 2 else 'ml_threshold_pct'
        ml_thresh_val = float(settings.get(thresh_key, settings.get('ml_threshold_pct', '50.0')))

        entry_desc = " AND ".join([format_rule_description(r) for r in entry_rules]) if entry_rules else "Sin reglas de entrada"
        exit_desc = " OR ".join([format_rule_description(r) for r in exit_rules]) if exit_rules else "Sin reglas de salida"

        trailing_rule = next((r for r in exit_rules if r.get('var1') == 'trailing_drop_pct'), None)
        if trailing_rule:
            val = float(trailing_rule.get('val2', 0))
            trailing_val_str = f"-{abs(val):.1f}%"
        else:
            trailing_val_str = "No activo"

        summary['entry_rules_text'] = entry_desc
        summary['exit_rules_text'] = exit_desc
        summary['ml_threshold_pct'] = ml_thresh_val
        summary['trailing_stop_val'] = trailing_val_str

        # Leer operaciones paginadas desde PostgreSQL
        paged_data = database.get_simulated_trades_paginated(
            page=page, per_page=per_page,
            start_ts=start_ts, end_ts=end_ts,
            profile=profile
        )

        return jsonify({
            'success': True,
            'profile': profile,
            'summary': summary,
            'pagination': {
                'page': paged_data['page'],
                'per_page': paged_data['per_page'],
                'total_records': paged_data['total_records'],
                'total_pages': paged_data['total_pages']
            },
            'trades': paged_data['records']
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

def build_rules_checklist(klines, rules, is_exit=False, open_position=None):
    """Evalúa individualmente cada regla de estrategia sobre la última vela para construir una lista de verificación en vivo."""
    if not rules or len(klines) < 3:
        return []

    c0 = dict(klines[-1])
    c1 = dict(klines[-2])

    if open_position:
        current_price = float(c0['close'])
        buy_price = float(open_position.get('price', current_price))
        trade_start_ts = open_position.get('timestamp') or open_position.get('buy_time')
        peak_price = buy_price
        if trade_start_ts:
            for k in klines:
                if k['time'] >= trade_start_ts:
                    peak_price = max(peak_price, float(k.get('high', k['close'])))
        
        c0['buy_price'] = buy_price
        c0['peak_price'] = peak_price
        c0['pnl_pct'] = ((current_price - buy_price) / buy_price) * 100.0 if buy_price > 0 else 0.0
        c0['trailing_drop_pct'] = ((current_price - peak_price) / peak_price) * 100.0 if peak_price > 0 else 0.0

    var_labels = {
        'close': 'Precio Actual',
        'open': 'Precio Apertura',
        'high': 'Precio Máximo',
        'low': 'Precio Mínimo',
        'ema9': 'EMA 9',
        'ema21': 'EMA 21',
        'ema35': 'EMA 35',
        'ema50': 'EMA 50',
        'ema100': 'EMA 100',
        'ema200': 'EMA 200',
        'ema9_slope_pct': 'Pendiente % EMA 9',
        'ema21_slope_pct': 'Pendiente % EMA 21',
        'rsi14': 'RSI (14)',
        'macd': 'MACD Line',
        'macd_signal': 'MACD Signal',
        'macd_hist': 'MACD Hist',
        'atr14': 'ATR (14)',
        'pnl_pct': '% PnL Posición',
        'trailing_drop_pct': '% Caída desde Máximo',
        'buy_price': 'Precio de Compra'
    }

    op_labels = {
        'gt': '>',
        'lt': '<',
        'gte': '>=',
        'lte': '<=',
        'cross_above': 'Cruza por encima de',
        'cross_below': 'Cruza por debajo de'
    }

    checklist = []
    for rule in rules:
        var1 = rule['var1']
        op = rule['op']
        compare_type = rule['compare_type']
        var2 = rule['var2']
        val2 = float(rule['val2'])

        v1_c0 = get_variable_value(c0, var1)
        v2_c0 = get_variable_value(c0, var2) if compare_type == 'indicator' else val2
        v1_c1 = get_variable_value(c1, var1)
        v2_c1 = get_variable_value(c1, var2) if compare_type == 'indicator' else val2

        if v1_c0 is None or v2_c0 is None:
            rule_passed = False
        else:
            if var1 == 'trailing_drop_pct':
                drop_limit = -abs(val2)
                if op in ['lt', 'lte']:
                    rule_passed = v1_c0 <= drop_limit
                elif op in ['gt', 'gte']:
                    rule_passed = v1_c0 >= drop_limit
            elif op == 'gt':
                rule_passed = v1_c0 > v2_c0
            elif op == 'lt':
                rule_passed = v1_c0 < v2_c0
            elif op == 'gte':
                rule_passed = v1_c0 >= v2_c0
            elif op == 'lte':
                rule_passed = v1_c0 <= v2_c0
            elif op == 'cross_above':
                rule_passed = (v1_c1 is not None and v2_c1 is not None) and (v1_c1 <= v2_c1 and v1_c0 > v2_c0)
            elif op == 'cross_below':
                rule_passed = (v1_c1 is not None and v2_c1 is not None) and (v1_c1 >= v2_c1 and v1_c0 < v2_c0)
            else:
                rule_passed = False

        v1_label = var_labels.get(var1, var1.upper())
        v2_label = var_labels.get(var2, var2.upper()) if compare_type == 'indicator' else str(val2)

        def fmt_val(var_name, val):
            if val is None: return "N/A"
            if 'price' in var_name or var_name in ['close', 'open', 'high', 'low', 'ema9', 'ema21', 'ema35', 'ema50', 'ema100', 'ema200']:
                return f"${val:,.2f}"
            elif 'pct' in var_name:
                return f"{val:+.2f}%"
            else:
                return f"{val:.2f}"

        val1_str = fmt_val(var1, v1_c0)
        val2_str = fmt_val(var2, v2_c0) if compare_type == 'indicator' else (f"-{abs(val2):.1f}%" if var1 == 'trailing_drop_pct' else f"{val2:.2f}")

        checklist.append({
            'var1_label': v1_label,
            'op_str': op_labels.get(op, op),
            'v2_label': v2_label,
            'val1_str': val1_str,
            'val2_str': val2_str,
            'passed': rule_passed,
            'description': f"{v1_label} ({val1_str}) {op_labels.get(op, op)} {v2_label} ({val2_str})"
        })

    return checklist

@app.route('/api/dashboard/status')
def api_dashboard_status():
    """Endpoint API en vivo para el Dashboard que retorna la posición activa, 
    la checklist en tiempo real de la última vela y las últimas operaciones."""
    try:
        settings = database.get_settings()
        profile = int(settings.get('active_strategy_profile', '1'))
        
        klines = database.get_candles_15m(limit=300)
        if not klines:
            return jsonify({'success': False, 'message': 'No hay velas cargadas'})

        last_candle = klines[-1]
        last_ts = int(last_candle['time'])
        last_date_str = datetime.fromtimestamp(last_ts / 1000, tz=CHILE_TZ).strftime('%Y-%m-%d %H:%M')

        open_pos = get_open_position()
        has_open = open_pos is not None

        open_pos_info = None
        if has_open:
            current_price = float(last_candle['close'])
            buy_price = float(open_pos['price'])
            buy_ts = open_pos.get('timestamp') or open_pos.get('buy_time')
            buy_date_str = datetime.fromtimestamp(buy_ts / 1000, tz=CHILE_TZ).strftime('%Y-%m-%d %H:%M') if buy_ts else 'N/A'
            amount_ftnt = float(open_pos.get('amount', 0.0))
            invested_usdt = float(open_pos.get('usdt_value', 0.0))
            pnl_usdt = (current_price - buy_price) * amount_ftnt
            pnl_pct = ((current_price - buy_price) / buy_price) * 100.0 if buy_price > 0 else 0.0

            peak_price = buy_price
            if buy_ts:
                for k in klines:
                    if k['time'] >= buy_ts:
                        peak_price = max(peak_price, float(k.get('high', k['close'])))

            trailing_drop = ((current_price - peak_price) / peak_price) * 100.0 if peak_price > 0 else 0.0

            open_pos_info = {
                'buy_price': buy_price,
                'buy_date_str': buy_date_str,
                'amount_ftnt': amount_ftnt,
                'invested_usdt': invested_usdt,
                'current_price': current_price,
                'pnl_usdt': round(pnl_usdt, 2),
                'pnl_pct': round(pnl_pct, 2),
                'peak_price': round(peak_price, 2),
                'trailing_drop_pct': round(trailing_drop, 2)
            }

        entry_rules = database.get_strategy('entry', profile=profile)
        exit_rules = database.get_strategy('exit', profile=profile)

        entry_checklist = build_rules_checklist(klines, entry_rules, is_exit=False)
        exit_checklist = build_rules_checklist(klines, exit_rules, is_exit=True, open_position=open_pos)

        entry_all_passed, entry_eval_reason = evaluate_rules(klines, entry_rules, is_exit=False)
        exit_any_passed = any(item['passed'] for item in exit_checklist) if exit_checklist else False

        ml_approve, ml_conf = ml_engine.predict_candle(last_candle, profile=profile)
        thresh_key = f'ml_threshold_pct_{profile}' if profile == 2 else 'ml_threshold_pct'
        ml_thresh = float(settings.get(thresh_key, settings.get('ml_threshold_pct', 50.0)))

        # Evaluaciones detalladas para la UI estructurada en RAMA 1 y RAMA 2
        c0 = klines[-1]
        c1 = klines[-2]
        macd_c0 = get_variable_value(c0, 'macd', klines=klines) or 0
        sig_c0 = get_variable_value(c0, 'macd_signal', klines=klines) or 0
        macd_c1 = get_variable_value(c1, 'macd', klines=klines[:-1]) or 0
        sig_c1 = get_variable_value(c1, 'macd_signal', klines=klines[:-1]) or 0
        rsi_c0 = get_variable_value(c0, 'rsi14', klines=klines) or 50
        rsi_c1 = get_variable_value(c1, 'rsi14', klines=klines[:-1]) or 50
        ema9_c0 = get_variable_value(c0, 'ema9', klines=klines) or 0
        ema21_c0 = get_variable_value(c0, 'ema21', klines=klines) or 0

        r1_c1 = (macd_c1 <= sig_c1) and (macd_c0 > sig_c0)
        r1_c2 = rsi_c0 > 45.0
        rama1_passed = r1_c1 and r1_c2

        r2_c1 = macd_c0 > sig_c0
        r2_c2 = ema9_c0 > ema21_c0
        r2_c3 = (rsi_c1 <= 50.0) and (rsi_c0 > 50.0)
        rama2_passed = r2_c1 and r2_c2 and r2_c3

        r3_c1 = macd_c0 > sig_c0
        r3_c2 = (close_c0 > ema9_c0) and (ema9_c0 > ema21_c0)
        r3_c3 = close_c0 > (open_c0 * 1.003)
        r3_c4 = rsi_c0 > 48.0
        rama3_passed = r3_c1 and r3_c2 and r3_c3 and r3_c4

        entry_branches = {
            'rama1': {
                'title': 'RAMA 1: Entrada por Cruce Inicial',
                'passed': rama1_passed,
                'items': [
                    {'label': 'MACD Line Cruza por encima de Signal', 'val_str': f"{macd_c0:.2f} vs Signal ({sig_c0:.2f})", 'passed': r1_c1},
                    {'label': 'RSI (14) > 45.0', 'val_str': f"{rsi_c0:.2f} vs 45.0", 'passed': r1_c2}
                ]
            },
            'rama2': {
                'title': 'RAMA 2: Re-Entrada por Impulso en Tendencia',
                'passed': rama2_passed,
                'items': [
                    {'label': 'MACD Line por encima de Signal (Tendencia)', 'val_str': f"{macd_c0:.2f} > {sig_c0:.2f}", 'passed': r2_c1},
                    {'label': 'Alineación de Medias (EMA9 > EMA21)', 'val_str': f"${ema9_c0:.2f} > ${ema21_c0:.2f}", 'passed': r2_c2},
                    {'label': 'RSI (14) Cruza por encima de 50.0', 'val_str': f"{rsi_c0:.2f} (ant: {rsi_c1:.2f})", 'passed': r2_c3}
                ]
            },
            'rama3': {
                'title': 'RAMA 3: Ruptura por Impulso Alcista',
                'passed': rama3_passed,
                'items': [
                    {'label': 'Precio por encima de EMA9 y EMA21', 'val_str': f"${close_c0:.2f} > ${ema9_c0:.2f}", 'passed': r3_c2},
                    {'label': 'Vela Verde Impulso (> +0.3%)', 'val_str': f"Cierre: ${close_c0:.2f} vs Ap: ${open_c0:.2f}", 'passed': r3_c3},
                    {'label': 'RSI (14) > 48.0', 'val_str': f"{rsi_c0:.2f} vs 48.0", 'passed': r3_c4}
                ]
            },
            'ml_filter': {
                'title': 'Filtro Machine Learning (IA)',
                'passed': ml_approve,
                'confidence': round(ml_conf, 1),
                'threshold': ml_thresh
            }
        }

        raw_trades = database.get_trades()
        recent_trades = []
        for t in raw_trades:
            ts = t.get('timestamp', 0)
            date_str = datetime.fromtimestamp(ts / 1000, tz=CHILE_TZ).strftime('%Y-%m-%d %H:%M') if ts else 'N/A'
            recent_trades.append({
                'id': t.get('id'),
                'type': t.get('type'),
                'date_str': date_str,
                'price': float(t.get('price', 0.0)),
                'amount_ftnt': float(t.get('amount', 0.0)),
                'usdt_value': float(t.get('usdt_value', 0.0)),
                'pnl': float(t.get('pnl', 0.0)) if t.get('pnl') is not None else None,
                'mode': t.get('mode', 'AUTO'),
                'reason': t.get('reason', '')
            })

        return jsonify({
            'success': True,
            'profile': profile,
            'candle_date_str': last_date_str,
            'current_price': float(last_candle['close']),
            'has_open_position': has_open,
            'open_position': open_pos_info,
            'entry_checklist': entry_checklist,
            'entry_branches': entry_branches,
            'entry_all_passed': entry_all_passed,
            'ml_prediction': {
                'approve': ml_approve,
                'confidence': ml_conf,
                'threshold': ml_thresh
            },
            'exit_checklist': exit_checklist,
            'exit_any_passed': exit_any_passed,
            'recent_trades': recent_trades
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/sync/status')
def get_sync_status_api():
    """Retorna el estado en vivo de la sincronización de PostgreSQL y el monitoreo de las últimas 3 velas."""
    try:
        status_data = database.get_sync_monitoring_status()
        return jsonify({'success': True, 'sync': status_data})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/data')
def get_market_data():
    """Retorna las velas de mercado almacenadas en PostgreSQL en el rango especificado con sus indicadores y trades."""
    try:
        settings = database.get_settings()
        interval = settings.get('candle_interval', '15m')
        
        start_ts = request.args.get('start_ts', type=int)
        end_ts = request.args.get('end_ts', type=int)
        limit = request.args.get('limit', type=int, default=300)
        
        if start_ts is not None and end_ts is not None:
            klines = database.get_candles_15m_range(start_ts=start_ts, end_ts=end_ts)
        elif start_ts is not None:
            klines = database.get_candles_15m_range(start_ts=start_ts)
        elif end_ts is not None:
            klines = database.get_candles_15m_range(end_ts=end_ts, limit=limit)
        else:
            klines = database.get_candles_15m(limit=limit)
            
        trades = database.get_trades()
        balances = get_current_balances()

        # Adjuntar predicción de ML para cada compra si no la especifica
        klines_by_ts = {k['time']: k for k in klines} if klines else {}
        for t in trades:
            if 'ml_approve' not in t:
                if t.get('type') == 'BUY':
                    buy_candle = klines_by_ts.get(t.get('timestamp'))
                    if buy_candle:
                        approve, conf = ml_engine.predict_candle(buy_candle)
                        t['ml_approve'] = 'SI' if approve else 'NO'
                        t['ml_confidence'] = round(conf, 1)
                    else:
                        t['ml_approve'] = 'SI'
                else:
                    t['ml_approve'] = 'SI'

        return jsonify({
            'success': True,
            'interval': interval,
            'klines': klines,
            'trades': trades,
            'balances': balances
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

def run_simulation_3d(klines, entry_rules, exit_rules, trade_size_usd=100.0):
    """
    Ejecuta un backtest/simulación histórica sobre las velas recibidas.
    Evalúa las reglas de entrada y salida ordenadas cronológicamente para simular los trades.
    """
    if not klines or len(klines) < 3:
        return {'trades': [], 'summary': {'completed_trades': 0, 'total_pnl_usd': 0.0, 'total_pnl_pct': 0.0, 'winning_trades': 0, 'losing_trades': 0, 'win_rate_pct': 0.0, 'has_open_position': False}}

    simulated_trades = []
    open_pos = None
    trade_id_counter = 1

    for i in range(2, len(klines)):
        sub_klines = klines[0:i+1]
        candle = klines[i]
        price = float(candle['close'])
        timestamp = int(candle['time'])

        if open_pos is None:
            # Evaluar entrada
            passed, _ = evaluate_rules(sub_klines, entry_rules, is_exit=False)
            if passed:
                trade_group = timestamp
                amount_ftnt = trade_size_usd / price if price > 0 else 0.0
                open_pos = {
                    'trade_group': trade_group,
                    'buy_price': price,
                    'buy_time': timestamp,
                    'amount': amount_ftnt,
                    'usdt_value': trade_size_usd
                }
                simulated_trades.append({
                    'id': trade_id_counter,
                    'type': 'BUY',
                    'timestamp': timestamp,
                    'price': price,
                    'amount': amount_ftnt,
                    'usdt_value': trade_size_usd,
                    'trade_group': trade_group,
                    'pnl': None
                })
                trade_id_counter += 1
        else:
            # Evaluar salida
            passed, _ = evaluate_rules(sub_klines, exit_rules, is_exit=True)
            if passed:
                sell_usdt_value = open_pos['amount'] * price
                pnl = sell_usdt_value - open_pos['usdt_value']
                simulated_trades.append({
                    'id': trade_id_counter,
                    'type': 'SELL',
                    'timestamp': timestamp,
                    'price': price,
                    'amount': open_pos['amount'],
                    'usdt_value': sell_usdt_value,
                    'trade_group': open_pos['trade_group'],
                    'pnl': pnl
                })
                trade_id_counter += 1
                open_pos = None

    # Calcular estadísticas del resumen de la simulación
    grouped = {}
    for t in simulated_trades:
        tg = t['trade_group']
        if tg not in grouped:
            grouped[tg] = {}
        grouped[tg][t['type']] = t

    completed_trades = 0
    winning_trades = 0
    losing_trades = 0
    total_pnl_usd = 0.0
    total_invested_usd = 0.0

    for tg, group in grouped.items():
        if 'BUY' in group and 'SELL' in group:
            completed_trades += 1
            pnl = float(group['SELL'].get('pnl', 0.0))
            total_pnl_usd += pnl
            total_invested_usd += float(group['BUY']['usdt_value'])
            if pnl > 0:
                winning_trades += 1
            elif pnl < 0:
                losing_trades += 1

    win_rate = (winning_trades / completed_trades * 100.0) if completed_trades > 0 else 0.0
    total_pnl_pct = (total_pnl_usd / total_invested_usd * 100.0) if total_invested_usd > 0 else 0.0

    summary = {
        'completed_trades': completed_trades,
        'total_pnl_usd': round(total_pnl_usd, 2),
        'total_pnl_pct': round(total_pnl_pct, 2),
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate_pct': round(win_rate, 1),
        'has_open_position': open_pos is not None
    }

    return {'trades': simulated_trades, 'summary': summary}

def run_full_simulation(klines, entry_rules, exit_rules, initial_balance=10000.0, profile=1):
    """
    Ejecuta una simulación completa de trading sobre las velas leídas de PostgreSQL.
    Retorna un resumen de métricas y la lista detallada de pares de operaciones (Entrada -> Salida).
    profile: 1 o 2
    """
    if not klines or len(klines) < 3:
        return {
            'summary': {
                'initial_balance': initial_balance,
                'final_balance': initial_balance,
                'total_pnl_usdt': 0.0,
                'total_return_pct': 0.0,
                'completed_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate_pct': 0.0
            },
            'trade_pairs': []
        }

    klines_by_ts = {int(c['time']): c for c in klines}
    balance_usdt = initial_balance
    position = None
    trade_pairs = []
    trade_num = 1

    for i in range(2, len(klines)):
        sub_klines = klines[0:i+1]
        c = klines[i]
        price = float(c['close'])
        ts = int(c['time'])

        if position is not None:
            # Actualizar precio máximo alcanzado durante el trade
            high_price = float(c.get('high', price))
            if high_price > position['peak_price']:
                position['peak_price'] = high_price
            
            c['buy_price'] = position['buy_price']
            c['peak_price'] = position['peak_price']
            c['pnl_pct'] = ((price - position['buy_price']) / position['buy_price']) * 100.0
            c['trailing_drop_pct'] = ((price - position['peak_price']) / position['peak_price']) * 100.0 if position['peak_price'] > 0 else 0.0

            # Evaluar salida
            passed, _ = evaluate_rules(sub_klines, exit_rules, is_exit=True)
            if passed:
                sell_usdt = position['amount_ftnt'] * price
                pnl_usdt = sell_usdt - position['invested_usdt']
                pnl_pct = (pnl_usdt / position['invested_usdt']) * 100.0
                balance_usdt = balance_usdt - position['invested_usdt'] + sell_usdt

                buy_d = datetime.fromtimestamp(position['buy_time'] / 1000, tz=CHILE_TZ).strftime('%Y-%m-%d %H:%M')
                sell_d = datetime.fromtimestamp(ts / 1000, tz=CHILE_TZ).strftime('%Y-%m-%d %H:%M')

                trade_pairs.append({
                    'trade_num': trade_num,
                    'buy_time': position['buy_time'],
                    'buy_date_str': buy_d,
                    'buy_price': position['buy_price'],
                    'sell_time': ts,
                    'sell_date_str': sell_d,
                    'sell_price': price,
                    'amount_ftnt': position['amount_ftnt'],
                    'invested_usdt': round(position['invested_usdt'], 2),
                    'returned_usdt': round(sell_usdt, 2),
                    'pnl_usdt': round(pnl_usdt, 2),
                    'pnl_pct': round(pnl_pct, 2),
                    'cumulative_balance': round(balance_usdt, 2),
                    'status': 'Ganancia' if pnl_usdt >= 0 else 'Pérdida'
                })
                trade_num += 1
                position = None
        else:
            # Evaluar entrada
            passed, _ = evaluate_rules(sub_klines, entry_rules, is_exit=False)
            if passed:
                invested = balance_usdt
                amount_ftnt = invested / price if price > 0 else 0.0
                position = {
                    'buy_time': ts,
                    'buy_price': price,
                    'peak_price': float(c.get('high', price)),
                    'amount_ftnt': amount_ftnt,
                    'invested_usdt': invested
                }

    # Si queda posición abierta al final
    if position is not None:
        last_c = klines[-1]
        last_p = float(last_c['close'])
        last_ts = int(last_c['time'])
        sell_usdt = position['amount_ftnt'] * last_p
        pnl_usdt = sell_usdt - position['invested_usdt']
        pnl_pct = (pnl_usdt / position['invested_usdt']) * 100.0
        balance_usdt = balance_usdt - position['invested_usdt'] + sell_usdt

        buy_d = datetime.fromtimestamp(position['buy_time'] / 1000, tz=CHILE_TZ).strftime('%Y-%m-%d %H:%M')
        sell_d = datetime.fromtimestamp(last_ts / 1000, tz=CHILE_TZ).strftime('%Y-%m-%d %H:%M')

        trade_pairs.append({
            'trade_num': trade_num,
            'buy_time': position['buy_time'],
            'buy_date_str': buy_d,
            'buy_price': position['buy_price'],
            'sell_time': last_ts,
            'sell_date_str': sell_d + " (Abierta)",
            'sell_price': last_p,
            'amount_ftnt': position['amount_ftnt'],
            'invested_usdt': round(position['invested_usdt'], 2),
            'returned_usdt': round(sell_usdt, 2),
            'pnl_usdt': round(pnl_usdt, 2),
            'pnl_pct': round(pnl_pct, 2),
            'cumulative_balance': round(balance_usdt, 2),
            'status': 'Ganancia' if pnl_usdt >= 0 else 'Pérdida'
        })

    completed = len(trade_pairs)
    winning = sum(1 for t in trade_pairs if t['pnl_usdt'] >= 0)
    losing = sum(1 for t in trade_pairs if t['pnl_usdt'] < 0)
    total_pnl = balance_usdt - initial_balance
    total_return_pct = (total_pnl / initial_balance) * 100.0 if initial_balance > 0 else 0.0
    win_rate = (winning / completed * 100.0) if completed > 0 else 0.0

    # --- ENTRENAMIENTO Y EVALUACION DE MACHINE LEARNING (POR PERFIL) ---
    ml_engine.train_ml_model(klines_by_ts, trade_pairs, profile=profile)

    ml_balance = initial_balance
    ml_completed_trades = 0
    ml_winning_trades = 0
    ml_losing_trades = 0
    ml_filtered_losses_count = 0
    ml_filtered_losses_val = 0.0
    ml_filtered_wins_count = 0
    ml_filtered_wins_val = 0.0

    # Predicción vectorizada en lote para máximo rendimiento
    buy_candles = [klines_by_ts.get(t['buy_time']) for t in trade_pairs]
    ml_threshold = ml_engine.get_ml_threshold(profile)
    predictions = ml_engine.predict_candles_batch(buy_candles, profile=profile, threshold=ml_threshold)

    for idx, t in enumerate(trade_pairs):
        approve, confidence = predictions[idx] if idx < len(predictions) else (True, 50.0)

        t['ml_approve'] = 'SI' if approve else 'NO'
        t['ml_confidence'] = round(confidence, 1)

        if approve:
            ml_completed_trades += 1
            invested = t['invested_usdt']
            returned = t['returned_usdt']
            trade_pnl = returned - invested
            ml_balance = ml_balance - invested + returned
            t['ml_pnl_usdt'] = round(trade_pnl, 2)
            t['ml_pnl_pct'] = t['pnl_pct']
            if trade_pnl > 0:
                ml_winning_trades += 1
            else:
                ml_losing_trades += 1
        else:
            t['ml_pnl_usdt'] = 0.0
            t['ml_pnl_pct'] = 0.0
            if t['pnl_usdt'] < 0:
                ml_filtered_losses_count += 1
                ml_filtered_losses_val += abs(t['pnl_usdt'])
            else:
                ml_filtered_wins_count += 1
                ml_filtered_wins_val += t['pnl_usdt']

        t['ml_cumulative_balance'] = round(ml_balance, 2)

    ml_total_pnl = ml_balance - initial_balance
    ml_total_return_pct = (ml_total_pnl / initial_balance) * 100.0 if initial_balance > 0 else 0.0
    ml_win_rate = (ml_winning_trades / ml_completed_trades * 100.0) if ml_completed_trades > 0 else 0.0

    summary = {
        'initial_balance': initial_balance,
        'final_balance': round(balance_usdt, 2),
        'total_pnl_usdt': round(total_pnl, 2),
        'total_return_pct': round(total_return_pct, 2),
        'completed_trades': completed,
        'winning_trades': winning,
        'losing_trades': losing,
        'win_rate_pct': round(win_rate, 1),

        'ml_final_balance': round(ml_balance, 2),
        'ml_total_pnl_usdt': round(ml_total_pnl, 2),
        'ml_total_return_pct': round(ml_total_return_pct, 2),
        'ml_completed_trades': ml_completed_trades,
        'ml_winning_trades': ml_winning_trades,
        'ml_losing_trades': ml_losing_trades,
        'ml_win_rate_pct': round(ml_win_rate, 1),
        'ml_filtered_losses_count': ml_filtered_losses_count,
        'ml_filtered_losses_val': round(ml_filtered_losses_val, 2),
        'ml_filtered_wins_count': ml_filtered_wins_count,
        'ml_filtered_wins_val': round(ml_filtered_wins_val, 2)
    }

    return {'summary': summary, 'trade_pairs': trade_pairs}

def update_persistent_simulation(profile=1):
    """
    Recalcula la simulación completa del perfil indicado (1 o 2) y la guarda en PostgreSQL.
    """
    try:
        klines = database.get_candles_15m_range(limit=75000)
        entry_rules = database.get_strategy('entry', profile=profile)
        exit_rules = database.get_strategy('exit', profile=profile)

        sim_res = run_full_simulation(klines, entry_rules, exit_rules, initial_balance=10000.0, profile=profile)
        database.save_simulated_trades(sim_res['trade_pairs'], sim_res['summary'], profile=profile)
        print(f"Simulación Perfil {profile} guardada en PostgreSQL: {len(sim_res['trade_pairs'])} operaciones almacenadas.")
    except Exception as e:
        print(f"Error al actualizar simulación persistente Perfil {profile}: {e}")

@app.route('/api/data/3d')
def get_market_data_3d():
    """Retorna las velas de los últimos 3 días (288 velas) leídas directamente desde PostgreSQL con la simulación de estrategia."""
    try:
        settings = database.get_settings()
        trade_size_usd = float(settings.get('trade_size_usd', 100.0))
        
        # Leer 288 velas directamente desde la base de datos PostgreSQL
        klines = database.get_candles_15m(limit=288)
        
        entry_rules = database.get_strategy('entry')
        exit_rules = database.get_strategy('exit')
        
        sim_result = run_simulation_3d(klines, entry_rules, exit_rules, trade_size_usd)
        
        return jsonify({
            'success': True,
            'klines': klines,
            'trades': sim_result['trades'],
            'simulation_summary': sim_result['summary'],
            'entry_rules_count': len(entry_rules),
            'exit_rules_count': len(exit_rules)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

def update_all_persistent_simulations():
    """Recalcula las simulaciones de ambos perfiles."""
    update_persistent_simulation(profile=1)
    update_persistent_simulation(profile=2)

def weekly_ml_retrain_worker():
    """
    Hilo en segundo plano que re-entrena los modelos de Machine Learning y actualiza las simulaciones
    automáticamente todos los Domingos a las 17:00 hs (5:00 PM Hora Chile / America/Santiago).
    """
    print("Hilo de re-entrenamiento semanal de ML (Domingos 17:00 hs Chile) iniciado.")
    while True:
        try:
            now = datetime.now(CHILE_TZ)
            # Sunday = 6 en datetime de Python (0=Lunes, 6=Domingo)
            if now.weekday() == 6 and now.hour == 17 and now.minute == 0:
                database.add_log("Iniciando re-entrenamiento semanal programado de ML (Perfil 1 y Perfil 2)...", "INFO")
                update_all_persistent_simulations()
                database.add_log("Re-entrenamiento semanal de ML (ambos perfiles) completado con éxito.", "INFO")
                time.sleep(65)
        except Exception as e:
            print(f"Error en weekly_ml_retrain_worker: {e}")
            try:
                database.add_log(f"Error en el re-entrenamiento semanal programado de ML: {e}", "ERROR")
            except Exception:
                pass
        time.sleep(30)

# ================= INICIALIZACIÓN =================

if __name__ == '__main__':
    # Inicializar Base de Datos primero
    database.init_db()
    
    # Arrancar el hilo autónomo de sincronización en segundo plano
    sync_thread = threading.Thread(target=sync_worker.run_sync_worker, daemon=True)
    sync_thread.start()
    
    # Generar y guardar la simulación persistente de ambos perfiles en PostgreSQL
    sim_thread = threading.Thread(target=update_all_persistent_simulations, daemon=True)
    sim_thread.start()

    # Arrancar el hilo secundario del Bot
    bot_thread = threading.Thread(target=bot_worker, daemon=True)
    bot_thread.start()
    
    # Arrancar el hilo de re-entrenamiento semanal de ML (Domingos 5:00 PM Chile)
    weekly_retrain_thread = threading.Thread(target=weekly_ml_retrain_worker, daemon=True)
    weekly_retrain_thread.start()
    
    # Arrancar Servidor Flask en el puerto 5050 y accesible para toda la red
    app.run(host='0.0.0.0', port=5050, debug=False)
