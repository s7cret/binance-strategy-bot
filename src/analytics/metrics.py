import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional


def calculate_sharpe_ratio(returns: Union[pd.Series, List[float]], risk_free_rate: float = 0.0) -> float:
    """
    Calculate the Sharpe ratio of a strategy's returns.
    
    Args:
        returns: Series or list of returns
        risk_free_rate: Risk-free rate of return (annualized)
        
    Returns:
        Sharpe ratio
    """
    if isinstance(returns, list):
        returns = pd.Series(returns)
    
    if len(returns) == 0:
        return 0.0
    
    # Convert annualized risk_free_rate to the same frequency as returns
    # For daily returns, we typically assume 252 trading days per year
    if len(returns) > 0:
        period_returns = returns
        excess_returns = period_returns - risk_free_rate
        
        if excess_returns.std() == 0:
            return 0.0
        
        sharpe_ratio = excess_returns.mean() / excess_returns.std()
        
        # Annualize if we have daily returns
        # Assuming daily returns, multiply by sqrt(252)
        if len(returns) > 1:
            # Only annualize if we have reasonable amount of data
            sharpe_ratio = sharpe_ratio * np.sqrt(len(returns)) if len(returns) > 1 else sharpe_ratio
            
        return sharpe_ratio
    else:
        return 0.0


def calculate_max_drawdown(equity_curve: Union[pd.Series, List[float]]) -> float:
    """
    Calculate the maximum drawdown of a strategy.
    
    Args:
        equity_curve: Series or list of equity values
        
    Returns:
        Maximum drawdown as a percentage (negative value)
    """
    if isinstance(equity_curve, list):
        equity_curve = pd.Series(equity_curve)
    
    if len(equity_curve) == 0:
        return 0.0
    
    # Calculate the running maximum
    running_max = equity_curve.expanding().max()
    
    # Calculate drawdown as percentage drop from running max
    drawdown = (equity_curve - running_max) / running_max
    
    # Return the minimum (most negative) drawdown value
    max_drawdown = drawdown.min()
    
    return max_drawdown


def calculate_total_return(initial_capital: float, final_capital: float) -> float:
    """
    Calculate total return of a strategy.
    
    Args:
        initial_capital: Initial capital
        final_capital: Final capital
        
    Returns:
        Total return as a percentage
    """
    if initial_capital == 0:
        return 0.0
    
    return (final_capital - initial_capital) / initial_capital


def calculate_volatility(returns: Union[pd.Series, List[float]]) -> float:
    """
    Calculate the volatility of returns (standard deviation).
    
    Args:
        returns: Series or list of returns
        
    Returns:
        Volatility (annualized if daily returns)
    """
    if isinstance(returns, list):
        returns = pd.Series(returns)
    
    if len(returns) == 0:
        return 0.0
    
    volatility = returns.std()
    
    # Annualize if we have daily returns (multiply by sqrt(252))
    # We'll assume daily returns if we have many data points
    if len(returns) > 10:
        volatility = volatility * np.sqrt(len(returns))
    
    return volatility


def calculate_win_rate(trades: List[Dict]) -> float:
    """
    Calculate win rate of trades.
    
    Args:
        trades: List of trade dictionaries with 'pnl' key
        
    Returns:
        Win rate as a percentage
    """
    if not trades:
        return 0.0
    
    wins = [trade for trade in trades if trade.get('pnl', 0) > 0]
    win_rate = len(wins) / len(trades)
    
    return win_rate


def calculate_profit_factor(trades: List[Dict]) -> float:
    """
    Calculate profit factor of trades.
    
    Args:
        trades: List of trade dictionaries with 'pnl' key
        
    Returns:
        Profit factor (gains / losses)
    """
    if not trades:
        return 0.0
    
    gains = sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) > 0)
    losses = abs(sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) < 0))
    
    if losses == 0:
        return float('inf') if gains > 0 else 0.0
    
    return gains / losses


def calculate_metrics(
    returns: Union[pd.Series, List[float]],
    equity_curve: Optional[Union[pd.Series, List[float]]] = None,
    trades: Optional[List[Dict]] = None,
    risk_free_rate: float = 0.0
) -> Dict[str, float]:
    """
    Calculate a comprehensive set of performance metrics.
    
    Args:
        returns: Series or list of returns
        equity_curve: Series or list of equity values
        trades: List of trade dictionaries
        risk_free_rate: Risk-free rate for Sharpe ratio
        
    Returns:
        Dictionary of performance metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['sharpe_ratio'] = calculate_sharpe_ratio(returns, risk_free_rate)
    metrics['volatility'] = calculate_volatility(returns)
    
    # Drawdown metrics
    if equity_curve is not None:
        metrics['max_drawdown'] = calculate_max_drawdown(equity_curve)
    
    # Trade-based metrics
    if trades is not None:
        metrics['win_rate'] = calculate_win_rate(trades)
        metrics['profit_factor'] = calculate_profit_factor(trades)
    
    # Additional metrics
    if isinstance(returns, list):
        returns_series = pd.Series(returns)
    else:
        returns_series = returns
    
    if len(returns_series) > 0:
        metrics['total_return'] = returns_series.sum()
        metrics['average_return'] = returns_series.mean()
        metrics['best_return'] = returns_series.max()
        metrics['worst_return'] = returns_series.min()
    
    return metrics


# Example usage
if __name__ == "__main__":
    # Sample data for testing
    returns = [0.02, -0.01, 0.03, -0.005, 0.01, 0.02, -0.03, 0.015, 0.02, -0.01]
    equity_curve = [100, 102, 100.98, 103.96, 103.44, 104.47, 106.56, 103.36, 104.91, 106.96, 105.90]
    
    # Calculate individual metrics
    sharpe = calculate_sharpe_ratio(returns)
    max_dd = calculate_max_drawdown(equity_curve)
    
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.2%}")
    
    # Calculate all metrics
    all_metrics = calculate_metrics(returns, equity_curve)
    for metric, value in all_metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.2f}")
        else:
            print(f"{metric}: {value}")