import pandas as pd
import numpy as np
from typing import Union


def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI) indicator.
    
    Args:
        data: Price data series
        period: Period for RSI calculation (default 14)
        
    Returns:
        Series with RSI values
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_ema(data: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA).
    
    Args:
        data: Price data series
        period: Period for EMA calculation
        
    Returns:
        Series with EMA values
    """
    ema = data.ewm(span=period, adjust=False).mean()
    return ema


def calculate_sma(data: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Simple Moving Average (SMA).
    
    Args:
        data: Price data series
        period: Period for SMA calculation
        
    Returns:
        Series with SMA values
    """
    sma = data.rolling(window=period).mean()
    return sma


def calculate_macd(data: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple:
    """
    Calculate MACD indicator.
    
    Args:
        data: Price data series
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line EMA period (default 9)
        
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    fast_ema = calculate_ema(data, fast_period)
    slow_ema = calculate_ema(data, slow_period)
    
    macd_line = fast_ema - slow_ema
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(data: pd.Series, period: int = 20, std_dev: int = 2) -> tuple:
    """
    Calculate Bollinger Bands.
    
    Args:
        data: Price data series
        period: Period for middle band (default 20)
        std_dev: Standard deviation multiplier (default 2)
        
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    middle_band = calculate_sma(data, period)
    rolling_std = data.rolling(window=period).std()
    
    upper_band = middle_band + (rolling_std * std_dev)
    lower_band = middle_band - (rolling_std * std_dev)
    
    return upper_band, middle_band, lower_band


def calculate_stochastic(data: pd.Series, high: pd.Series, low: pd.Series, k_period: int = 14, d_period: int = 3) -> tuple:
    """
    Calculate Stochastic Oscillator.
    
    Args:
        data: Close price data series
        high: High price data series
        low: Low price data series
        k_period: %K period (default 14)
        d_period: %D period (default 3)
        
    Returns:
        Tuple of (%K, %D)
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    k_value = 100 * ((data - lowest_low) / (highest_high - lowest_low))
    d_value = k_value.rolling(window=d_period).mean()
    
    return k_value, d_value