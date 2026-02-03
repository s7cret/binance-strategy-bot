from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pandas as pd


class Strategy(ABC):
    """
    Base Strategy interface that all trading strategies should inherit from.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize the strategy with a name and configuration.
        
        Args:
            name: Name of the strategy
            config: Configuration dictionary containing strategy parameters
        """
        self.name = name
        self.config = config
    
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signal based on market data.
        
        Args:
            data: Market data as a pandas DataFrame
            
        Returns:
            Dictionary containing signal information (e.g., buy/sell/hold, position size, etc.)
        """
        pass
    
    @abstractmethod
    def prepare_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare raw market data for strategy calculations.
        
        Args:
            raw_data: Raw market data as a pandas DataFrame
            
        Returns:
            Prepared data suitable for strategy calculations
        """
        pass
    
    def get_name(self) -> str:
        """Return the name of the strategy."""
        return self.name
    
    def get_config(self) -> Dict[str, Any]:
        """Return the strategy configuration."""
        return self.config


class Signal:
    """
    Class representing a trading signal.
    """
    
    BUY = 'BUY'
    SELL = 'SELL'
    HOLD = 'HOLD'
    
    def __init__(self, signal_type: str, strength: float = 1.0, 
                 price: Optional[float] = None, 
                 confidence: Optional[float] = None):
        """
        Initialize a trading signal.
        
        Args:
            signal_type: Type of signal (BUY, SELL, HOLD)
            strength: Strength of the signal (0.0 to 1.0)
            price: Price at which signal was generated
            confidence: Confidence level of the signal (0.0 to 1.0)
        """
        self.signal_type = signal_type
        self.strength = strength
        self.price = price
        self.confidence = confidence
    
    def __repr__(self):
        return f"Signal(type={self.signal_type}, strength={self.strength}, confidence={self.confidence})"