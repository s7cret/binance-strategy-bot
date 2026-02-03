"""
Strategies package for the Binance Strategy Bot.

This package contains various trading strategies that implement the Strategy interface.
"""

from .base import Strategy, Signal
from .indicators import (
    calculate_rsi,
    calculate_ema,
    calculate_sma,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_stochastic
)

# Import specific strategies
from .ema_crossover import EMACrossoverStrategy
from .rsi_reversion import RSIVersionStrategy

__all__ = [
    'Strategy',
    'Signal',
    'EMACrossoverStrategy',
    'RSIVersionStrategy',
    # Indicators
    'calculate_rsi',
    'calculate_ema',
    'calculate_sma',
    'calculate_macd',
    'calculate_bollinger_bands',
    'calculate_stochastic'
]