import os
import requests
import time
from datetime import datetime, timezone

class AlpacaClient:
    """Cliente para la API de Alpaca (Market Data & Trading para Fortinet FTNT)."""

    def __init__(self, api_key=None, api_secret=None, is_paper=True):
        self.api_key = api_key or os.getenv("ALPACA_API_KEY", "")
        self.api_secret = api_secret or os.getenv("ALPACA_SECRET_KEY", "")
        self.is_paper = is_paper
        
        self.data_base_url = "https://data.alpaca.markets/v2"
        self.trading_base_url = "https://paper-api.alpaca.markets/v2" if is_paper else "https://api.alpaca.markets/v2"
        
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "accept": "application/json"
        }

    def get_klines(self, symbol="FTNT", interval="15m", limit=1000, startTime=None, endTime=None):
        """
        Obtiene velas históricas de 15 min de FTNT desde Alpaca.
        Retorna una lista de diccionarios en el formato esperado por el bot:
        [{'time': ms, 'open': float, 'high': float, 'low': float, 'close': float, 'volume': float, 'vwap': float, 'trade_count': int}]
        """
        url = f"{self.data_base_url}/stocks/bars"
        
        tf = "15Min" if interval == "15m" else "1Min"
        
        params = {
            "symbols": symbol,
            "timeframe": tf,
            "limit": min(limit, 10000),
            "adjustment": "split",
            "feed": "iex"
        }
        
        if startTime:
            dt_start = datetime.fromtimestamp(startTime / 1000.0, tz=timezone.utc)
            params["start"] = dt_start.strftime("%Y-%m-%dT%H:%M:%SZ")
            
        if endTime:
            dt_end = datetime.fromtimestamp(endTime / 1000.0, tz=timezone.utc)
            params["end"] = dt_end.strftime("%Y-%m-%dT%H:%M:%SZ")

        try:
            resp = requests.get(url, headers=self.headers, params=params, timeout=20)
            if resp.status_code != 200:
                # Intentar con feed sip
                params["feed"] = "sip"
                resp = requests.get(url, headers=self.headers, params=params, timeout=20)
                
            resp.raise_for_status()
            data = resp.json()
            bars = data.get("bars", {}).get(symbol, [])
            
            klines = []
            for b in bars:
                # t es ISO String UTC ej 2026-07-31T19:45:00Z
                dt = datetime.fromisoformat(b['t'].replace('Z', '+00:00'))
                time_ms = int(dt.timestamp() * 1000)
                
                klines.append({
                    'time': time_ms,
                    'open': float(b['o']),
                    'high': float(b['h']),
                    'low': float(b['l']),
                    'close': float(b['c']),
                    'volume': float(b['v']),
                    'vwap': float(b.get('vw', b['c'])),
                    'trade_count': int(b.get('n', 0))
                })
            
            return klines
        except Exception as e:
            print(f"Error obteniendo velas de Alpaca para {symbol}: {e}")
            return []

    def get_account(self):
        """Retorna información de la cuenta en Alpaca."""
        url = f"{self.trading_base_url}/account"
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"Error obteniendo cuenta de Alpaca: {e}")
            return None

    def get_positions(self):
        """Retorna las posiciones abiertas en Alpaca."""
        url = f"{self.trading_base_url}/positions"
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"Error obteniendo posiciones de Alpaca: {e}")
            return []

    def create_order(self, symbol="FTNT", qty=1, side="buy", type="market", time_in_force="gtc"):
        """Crea una orden de compra o venta en Alpaca."""
        url = f"{self.trading_base_url}/orders"
        payload = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side,
            "type": type,
            "time_in_force": time_in_force
        }
        try:
            resp = requests.post(url, headers=self.headers, json=payload, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"Error creando orden en Alpaca: {e}")
            return None
