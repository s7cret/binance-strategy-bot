"""
Binance API Client using aiohttp for async requests
"""
import aiohttp
import asyncio
import json
import hmac
import hashlib
from typing import Optional, Dict, Any, List
from urllib.parse import urlencode


class BinanceClient:
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """
        Initialize Binance client
        
        Args:
            api_key: Binance API key (optional for public endpoints)
            api_secret: Binance API secret (optional for public endpoints)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.binance.com"
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.session:
            await self.session.close()
    
    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """Generate HMAC SHA256 signature for private endpoints"""
        query_string = urlencode(params)
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    async def _make_request(self, method: str, endpoint: str, params: Optional[Dict[str, Any]] = None, 
                           signed: bool = False) -> Dict[str, Any]:
        """Make HTTP request to Binance API"""
        if not self.session:
            raise RuntimeError("Client session not initialized. Use as context manager or initialize manually.")
        
        url = f"{self.base_url}{endpoint}"
        
        # Prepare parameters
        req_params = params.copy() if params else {}
        
        # Add signature if required
        if signed and self.api_secret:
            req_params['signature'] = self._generate_signature(req_params)
        
        # Set headers
        headers = {}
        if self.api_key:
            headers['X-MBX-APIKEY'] = self.api_key
        
        # Make the request
        async with self.session.request(method, url, params=req_params, headers=headers) as response:
            if response.status != 200:
                text = await response.text()
                raise Exception(f"Binance API error {response.status}: {text}")
            
            return await response.json()
    
    async def get_klines(self, symbol: str, interval: str, start_time: Optional[int] = None, 
                         end_time: Optional[int] = None, limit: int = 500) -> List[List[Any]]:
        """
        Get klines/candlestick data
        
        Args:
            symbol: Trading pair (e.g. 'BTCUSDT')
            interval: Candle interval (e.g. '1m', '5m', '1h', '1d')
            start_time: Start time in milliseconds since epoch
            end_time: End time in milliseconds since epoch
            limit: Number of klines returned (max 1000)
            
        Returns:
            List of klines in format [timestamp, open, high, low, close, volume, ...]
        """
        params = {
            'symbol': symbol.upper(),
            'interval': interval,
            'limit': min(limit, 1000)  # Binance max limit is 1000
        }
        
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
            
        return await self._make_request('GET', '/api/v3/klines', params=params)
    
    async def get_exchange_info(self) -> Dict[str, Any]:
        """
        Get current exchange trading rules and symbol information
        
        Returns:
            Exchange information including symbols, filters, etc.
        """
        return await self._make_request('GET', '/api/v3/exchangeInfo')
    
    async def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get information for a specific symbol
        
        Args:
            symbol: Trading pair (e.g. 'BTCUSDT')
            
        Returns:
            Symbol information or None if symbol not found
        """
        exchange_info = await self.get_exchange_info()
        for sym in exchange_info.get('symbols', []):
            if sym['symbol'].upper() == symbol.upper():
                return sym
        return None
    
    async def get_server_time(self) -> int:
        """
        Test connectivity to the Rest API and get server time
        
        Returns:
            Server timestamp in milliseconds
        """
        result = await self._make_request('GET', '/api/v3/time')
        return result['serverTime']
    
    async def get_ticker_price(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get latest price for a symbol or all symbols
        
        Args:
            symbol: Trading pair (optional, returns all if not provided)
            
        Returns:
            Price information for specified symbol(s)
        """
        params = {}
        if symbol:
            params['symbol'] = symbol.upper()
        
        return await self._make_request('GET', '/api/v3/ticker/price', params=params)
    
    async def get_ticker_24hr(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get 24 hour rolling window price change statistics
        
        Args:
            symbol: Trading pair (optional, returns all if not provided)
            
        Returns:
            24 hour price statistics for specified symbol(s)
        """
        params = {}
        if symbol:
            params['symbol'] = symbol.upper()
        
        return await self._make_request('GET', '/api/v3/ticker/24hr', params=params)


# Example usage
async def main():
    # Initialize client without credentials for public endpoints
    async with BinanceClient() as client:
        # Get exchange info
        exchange_info = await client.get_exchange_info()
        print(f"Exchange info retrieved. Available symbols: {len(exchange_info['symbols'])}")
        
        # Get BTCUSDT klines (last 100 1-hour candles)
        klines = await client.get_klines('BTCUSDT', '1h', limit=100)
        print(f"Retrieved {len(klines)} klines")
        
        # Get server time
        server_time = await client.get_server_time()
        print(f"Server time: {server_time}")


if __name__ == "__main__":
    asyncio.run(main())