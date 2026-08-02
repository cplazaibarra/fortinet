import time
import hmac
import hashlib
import requests
from urllib.parse import urlencode

class BinanceClient:
    def __init__(self, api_key='', api_secret='', use_testnet=True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.use_testnet = use_testnet
        
        if use_testnet:
            self.base_url = "https://testnet.binance.vision"
        else:
            self.base_url = "https://api.binance.com"
            
    def get_klines(self, symbol="FTNT", interval="5m", limit=500, startTime=None, endTime=None):
        """
        Descarga velas (Klines) de Binance usando endpoints públicos de alta disponibilidad.
        """
        endpoints = [
            "https://api.binance.com/api/v3/klines",
            "https://api1.binance.com/api/v3/klines",
            "https://api2.binance.com/api/v3/klines",
            "https://api3.binance.com/api/v3/klines"
        ]
        
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        if startTime is not None:
            params["startTime"] = startTime
        if endTime is not None:
            params["endTime"] = endTime

        last_error = None
        for url in endpoints:
            try:
                response = requests.get(url, params=params, timeout=8)
                response.raise_for_status()
                data = response.json()
                
                klines = []
                for item in data:
                    klines.append({
                        'time': int(item[0]),         # Open time
                        'open': float(item[1]),        # Open
                        'high': float(item[2]),        # High
                        'low': float(item[3]),         # Low
                        'close': float(item[4]),       # Close
                        'volume': float(item[5])       # Volume
                    })
                return klines
            except Exception as e:
                last_error = e

        raise Exception(f"No se pudieron descargar velas desde ningún servidor de Binance: {last_error}")

    def _sign_request(self, params):
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        params['signature'] = signature
        return params

    def _get_headers(self):
        return {
            'X-MBX-APIKEY': self.api_key
        }

    def test_connectivity(self):
        """
        Prueba la conexión pública con Binance.
        """
        url = f"{self.base_url}/api/v3/ping"
        try:
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def test_api_keys(self):
        """
        Prueba si las llaves API de Binance Testnet son válidas obteniendo los datos de la cuenta.
        """
        if not self.api_key or not self.api_secret:
            return False, "Llaves API no configuradas"
        try:
            account_info = self.get_account_info()
            return True, "Conexión exitosa"
        except Exception as e:
            return False, str(e)

    def get_account_info(self):
        """
        Obtiene los datos de la cuenta (balances) de Binance.
        Requiere autenticación.
        """
        if not self.api_key or not self.api_secret:
            raise Exception("API Key y Secret son requeridos para operaciones privadas")
            
        url = f"{self.base_url}/api/v3/account"
        params = {
            'timestamp': int(time.time() * 1000),
            'recvWindow': 5000
        }
        self._sign_request(params)
        headers = self._get_headers()
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        if response.status_code != 200:
            raise Exception(f"Binance Error: {response.text}")
        return response.json()

    def get_balances(self):
        """
        Retorna los balances de USDT y FTNT de la cuenta.
        """
        info = self.get_account_info()
        balances = info.get('balances', [])
        
        usdt_bal = 0.0
        btc_bal = 0.0
        
        for bal in balances:
            asset = bal['asset']
            if asset == 'USDT':
                usdt_bal = float(bal['free'])
            elif asset == 'FTNT':
                btc_bal = float(bal['free'])
                
        return {'USDT': usdt_bal, 'FTNT': btc_bal}

    def place_market_order(self, symbol="FTNT", side="BUY", quantity=0.0):
        """
        Crea una orden a mercado en la red (BUY o SELL).
        Nota: Binance requiere 'quantity' en el activo base (FTNT).
        """
        if not self.api_key or not self.api_secret:
            raise Exception("API Key y Secret son requeridos para colocar órdenes")
            
        url = f"{self.base_url}/api/v3/order"
        params = {
            'symbol': symbol,
            'side': side,
            'type': 'MARKET',
            'quantity': f"{quantity:.6f}",
            'timestamp': int(time.time() * 1000),
            'recvWindow': 5000
        }
        self._sign_request(params)
        headers = self._get_headers()
        
        response = requests.post(url, data=params, headers=headers, timeout=10)
        if response.status_code != 200:
            raise Exception(f"Binance Error al colocar orden: {response.text}")
        return response.json()
