import pandas as pd
from typing import Dict, Any
from .base import Strategy, Signal
from .indicators import calculate_rsi


class RSIVersionStrategy(Strategy):
    """
    RSI Reversion Strategy - generates buy/sell signals based on RSI oversold/overbought levels.
    A buy signal is generated when RSI falls below oversold threshold and starts rising.
    A sell signal is generated when RSI rises above overbought threshold and starts falling.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("RSI_Reversion_Strategy", config)
        self.rsi_period = config.get('rsi_period', 14)
        self.overbought_level = config.get('overbought_level', 70)
        self.oversold_level = config.get('oversold_level', 30)
    
    def prepare_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data by calculating RSI indicator.
        
        Args:
            raw_data: Raw market data
            
        Returns:
            Data with calculated RSI indicator
        """
        df = raw_data.copy()
        
        # Calculate RSI
        df['rsi'] = calculate_rsi(df['close'], self.rsi_period)
        
        # Calculate RSI direction
        df['rsi_direction'] = df['rsi'].diff()
        df['rsi_prev'] = df['rsi'].shift(1)
        
        # Identify overbought/oversold conditions
        df['is_overbought'] = df['rsi'] > self.overbought_level
        df['is_oversold'] = df['rsi'] < self.oversold_level
        
        # Identify reversion points
        df['oversold_reversion'] = (df['is_oversold']) & (df['rsi_direction'] > 0)
        df['overbought_reversion'] = (df['is_overbought']) & (df['rsi_direction'] < 0)
        
        return df
    
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signals based on RSI reversion.
        
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
        
        # Check for RSI reversion signals
        if latest['oversold_reversion']:
            # RSI was oversold and is now rising - Buy signal
            signal = Signal(
                signal_type=Signal.BUY,
                strength=self._calculate_strength(latest['rsi']),
                price=latest['close'],
                confidence=0.7
            )
        elif latest['overbought_reversion']:
            # RSI was overbought and is now falling - Sell signal
            signal = Signal(
                signal_type=Signal.SELL,
                strength=self._calculate_strength(latest['rsi']),
                price=latest['close'],
                confidence=0.7
            )
        else:
            # No reversion signal
            signal = Signal(
                signal_type=Signal.HOLD,
                strength=0.0,
                price=latest['close'],
                confidence=0.5
            )
        
        return {
            'signal': signal,
            'rsi': latest['rsi'],
            'rsi_direction': latest['rsi_direction'],
            'timestamp': latest.name if hasattr(latest, 'name') else None
        }
    
    def _calculate_strength(self, rsi_value: float) -> float:
        """
        Calculate signal strength based on how far RSI is from center line.
        
        Args:
            rsi_value: Current RSI value
            
        Returns:
            Signal strength between 0 and 1
        """
        # Strength increases as RSI moves further from 50 (center)
        distance_from_center = abs(rsi_value - 50)
        strength = min(distance_from_center / 50, 1.0)  # Normalize to 0-1 range
        return strength


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
        'rsi_period': 14,
        'overbought_level': 70,
        'oversold_level': 30
    }
    
    strategy = RSIVersionStrategy(config)
    prepared_data = strategy.prepare_data(sample_data)
    signal_result = strategy.generate_signal(prepared_data)
    
    print(f"Strategy: {strategy.get_name()}")
    print(f"Signal: {signal_result['signal']}")
    print(f"RSI: {signal_result['rsi']:.2f}")
    print(f"RSI Direction: {signal_result['rsi_direction']:.2f}")