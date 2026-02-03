"""
Data ingestion module for Binance klines
Handles fetching data from Binance API and storing it in SQLite database
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.data.binance_client import BinanceClient
from src.storage.db import get_db
from src.core.config import settings


logger = logging.getLogger(__name__)


def transform_kline_for_db(kline: List[Any], symbol: str, interval: str) -> Dict[str, Any]:
    """
    Transform a Binance kline to match the database schema
    
    Args:
        kline: Raw kline data from Binance API
               [open_time, open, high, low, close, volume, close_time, ...]
        symbol: Trading pair symbol
        interval: Candle interval
        
    Returns:
        Dictionary matching the candles table schema
    """
    return {
        'symbol': symbol.upper(),
        'interval': interval,
        'open_time': kline[0],  # Open time
        'open': float(kline[1]),  # Open price
        'high': float(kline[2]),  # High price
        'low': float(kline[3]),  # Low price
        'close': float(kline[4]),  # Close price
        'volume': float(kline[5]),  # Volume
        'close_time': kline[6]  # Close time
    }


async def save_klines_to_db(klines: List[Dict[str, Any]]) -> int:
    """
    Save kline data to SQLite database
    
    Args:
        klines: List of transformed kline data ready for DB insertion
        
    Returns:
        Number of records inserted
    """
    db = await get_db()
    
    try:
        inserted_count = 0
        
        for kline in klines:
            await db.execute('''
                INSERT OR REPLACE INTO candles 
                (symbol, interval, open_time, open, high, low, close, volume, close_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                kline['symbol'],
                kline['interval'], 
                kline['open_time'],
                kline['open'],
                kline['high'],
                kline['low'],
                kline['close'],
                kline['volume'],
                kline['close_time']
            ))
            inserted_count += 1
        
        await db.commit()
        logger.info(f"Successfully saved {inserted_count} klines to database")
        return inserted_count
        
    except Exception as e:
        logger.error(f"Error saving klines to database: {e}")
        await db.rollback()
        raise
    finally:
        await db.close()


async def fetch_and_store_klines(symbol: str, interval: str, limit: int = 500, 
                                start_time: Optional[int] = None, 
                                end_time: Optional[int] = None) -> int:
    """
    Fetch klines from Binance and store them in the database
    
    Args:
        symbol: Trading pair (e.g. 'BTCUSDT')
        interval: Candle interval (e.g. '1m', '5m', '1h', '1d')
        limit: Number of klines to fetch (max 1000 per request)
        start_time: Start time in milliseconds since epoch
        end_time: End time in milliseconds since epoch
        
    Returns:
        Number of records stored
    """
    logger.info(f"Fetching {limit} klines for {symbol} at {interval} interval")
    
    async with BinanceClient() as client:
        # Fetch klines from Binance
        raw_klines = await client.get_klines(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        
        # Transform klines for database insertion
        transformed_klines = [
            transform_kline_for_db(kline, symbol, interval) 
            for kline in raw_klines
        ]
        
        # Save to database
        saved_count = await save_klines_to_db(transformed_klines)
        
        logger.info(f"Fetched and stored {saved_count} klines for {symbol}")
        return saved_count


async def fetch_and_store_exchange_info() -> Dict[str, Any]:
    """
    Fetch exchange information and store relevant data in the database
    
    Returns:
        Exchange information dictionary
    """
    logger.info("Fetching exchange information")
    
    async with BinanceClient() as client:
        exchange_info = await client.get_exchange_info()
        
        db = await get_db()
        try:
            # Insert/update market information for all symbols
            for symbol_info in exchange_info['symbols']:
                symbol = symbol_info['symbol']
                
                # Insert or update market info
                await db.execute('''
                    INSERT OR REPLACE INTO markets 
                    (symbol, base_asset, quote_asset, status)
                    VALUES (?, ?, ?, ?)
                ''', (
                    symbol,
                    symbol_info['baseAsset'],
                    symbol_info['quoteAsset'],
                    symbol_info['status']
                ))
            
            await db.commit()
            logger.info(f"Updated market information for {len(exchange_info['symbols'])} symbols")
            
        except Exception as e:
            logger.error(f"Error saving exchange info to database: {e}")
            await db.rollback()
            raise
        finally:
            await db.close()
        
        return exchange_info


async def sync_recent_klines(symbol: str, interval: str, lookback_hours: int = 24) -> int:
    """
    Sync recent klines based on a lookback period
    
    Args:
        symbol: Trading pair (e.g. 'BTCUSDT')
        interval: Candle interval (e.g. '1m', '5m', '1h')
        lookback_hours: How many hours back to fetch data
        
    Returns:
        Number of records synced
    """
    import time
    
    # Calculate start time based on lookback hours
    now = int(time.time() * 1000)  # Current time in milliseconds
    lookback_ms = lookback_hours * 60 * 60 * 1000  # Convert hours to milliseconds
    start_time = now - lookback_ms
    
    # Determine appropriate limit based on interval and lookback period
    interval_multipliers = {
        '1m': 60 * 1000,      # 1 minute in ms
        '5m': 5 * 60 * 1000,  # 5 minutes in ms
        '15m': 15 * 60 * 1000, # 15 minutes in ms
        '30m': 30 * 60 * 1000, # 30 minutes in ms
        '1h': 60 * 60 * 1000,  # 1 hour in ms
        '2h': 2 * 60 * 60 * 1000, # 2 hours in ms
        '4h': 4 * 60 * 60 * 1000, # 4 hours in ms
        '6h': 6 * 60 * 60 * 1000, # 6 hours in ms
        '12h': 12 * 60 * 60 * 1000, # 12 hours in ms
        '1d': 24 * 60 * 60 * 1000, # 1 day in ms
        '3d': 3 * 24 * 60 * 60 * 1000, # 3 days in ms
        '1w': 7 * 24 * 60 * 60 * 1000, # 1 week in ms
    }
    
    multiplier = interval_multipliers.get(interval, 60 * 1000)  # Default to 1 minute
    estimated_records = lookback_ms // multiplier
    limit = min(estimated_records, 1000)  # Binance API limit
    
    logger.info(f"Syncing {limit} klines for {symbol} at {interval} interval (last {lookback_hours} hours)")
    
    return await fetch_and_store_klines(
        symbol=symbol,
        interval=interval,
        limit=min(limit, 1000),
        start_time=start_time
    )


async def batch_sync_symbols(symbols_intervals: List[tuple], lookback_hours: int = 24) -> Dict[str, int]:
    """
    Batch sync multiple symbols and intervals
    
    Args:
        symbols_intervals: List of tuples (symbol, interval)
        lookback_hours: How many hours back to fetch data
        
    Returns:
        Dictionary mapping (symbol, interval) to number of records synced
    """
    results = {}
    
    for symbol, interval in symbols_intervals:
        try:
            count = await sync_recent_klines(symbol, interval, lookback_hours)
            results[f"{symbol}_{interval}"] = count
        except Exception as e:
            logger.error(f"Error syncing {symbol} {interval}: {e}")
            results[f"{symbol}_{interval}"] = 0
    
    return results


# Example usage and testing
async def main():
    """
    Example usage of the ingestion functions
    """
    logging.basicConfig(level=logging.INFO)
    
    # Fetch and store exchange info
    await fetch_and_store_exchange_info()
    
    # Fetch and store some BTCUSDT klines
    await fetch_and_store_klines('BTCUSDT', '1h', limit=100)
    
    # Sync recent klines for multiple pairs
    symbols_intervals = [
        ('BTCUSDT', '1h'),
        ('ETHUSDT', '1h'),
        ('BNBUSDT', '1h'),
        ('ADAUSDT', '1h')
    ]
    
    results = await batch_sync_symbols(symbols_intervals, lookback_hours=24)
    print("Batch sync results:", results)


if __name__ == "__main__":
    asyncio.run(main())