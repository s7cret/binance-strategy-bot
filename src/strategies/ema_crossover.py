import pandas as pd
from typing import Dict, Any
from .base import Strategy, Signal
from .indicators import calculate_ema


class EMACrossoverStrategy(Strategy):
    """
    EMA Crossover Strategy - generates buy/sell signals based on EMA crossovers.
    A buy signal is generated when the fast EMA crosses above the slow EMA.
    A sell signal is generated when the fast EMA crosses below the slow EMA.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("EMA_Crossover_Strategy", config)
        self.fast_period = config.get('fast_period', 12)
        self.slow_period = config.get('slow_period', 26)
    
    def prepare_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data by calculating EMA indicators.
        
        Args:
            raw_data: Raw market data
            
        Returns:
            Data with calculated EMA indicators
        """
        # Assuming raw_data has 'close' column
        df = raw_data.copy()
        
        # Calculate EMAs
        df['ema_fast'] = calculate_ema(df['close'], self.fast_period)
        df['ema_slow'] = calculate_ema(df['close'], self.slow_period)
        
        # Calculate crossover signals
        df['ema_diff'] = df['ema_fast'] - df['ema_slow']
        df['ema_prev_diff'] = df['ema_diff'].shift(1)
        
        # Detect crossovers
        df['bullish_cross'] = (df['ema_prev_diff'] <= 0) & (df['ema_diff'] > 0)
        df['bearish_cross'] = (df['ema_prev_diff'] >= 0) & (df['ema_diff'] < 0)
        
        return df
    
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signals based on EMA crossovers.
        
        Args:
            data: Market data with calculated indicators
            
        Returns:
            Trading signal dictionary
        """
        if len(data) < 2:
            return {'signal': Signal(Signal.HOLD)}
        
        # Get the latest row
        latest = data.iloc[-1]
        previous = data.iloc[-2]
        
        # Check for crossover signals
        if latest['bullish_cross']:
            # Fast EMA crossed above slow EMA - Buy signal
            signal = Signal(
                signal_type=Signal.BUY,
                strength=1.0,
                price=latest['close'],
                confidence=0.8
            )
        elif latest['bearish_cross']:
            # Fast EMA crossed below slow EMA - Sell signal
            signal = Signal(
                signal_type=Signal.SELL,
                strength=1.0,
                price=latest['close'],
                confidence=0.8
            )
        else:
            # No crossover - Hold
            signal = Signal(
                signal_type=Signal.HOLD,
                strength=0.0,
                price=latest['close'],
                confidence=0.5
            )
        
        return {
            'signal': signal,
            'ema_fast': latest['ema_fast'],
            'ema_slow': latest['ema_slow'],
            'timestamp': latest.name if hasattr(latest, 'name') else None
        }


# Example usage and testing
if __name__ == "__main__":
    import numpy as np
    
    # Create sample data for testing
    dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
    prices = 100 + np.cumsum(np.random.randn(50) * 0.5)
    
    sample_data = pd.DataFrame({
        'close': prices
    }, index=dates)
    
    # Test the strategy
    config = {
        'fast_period': 5,
        'slow_period': 10
    }
    
    strategy = EMACrossoverStrategy(config)
    prepared_data = strategy.prepare_data(sample_data)
    signal_result = strategy.generate_signal(prepared_data)
    
    print(f"Strategy: {strategy.get_name()}")
    print(f"Signal: {signal_result['signal']}")
    print(f"Fast EMA: {signal_result['ema_fast']:.2f}")
    print(f"Slow EMA: {signal_result['ema_slow']:.2f}")