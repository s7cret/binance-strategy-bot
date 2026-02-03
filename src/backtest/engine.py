"""
Backtesting Engine Module

A basic backtesting engine that simulates trading strategies against historical data.
"""

import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum


@dataclass
class Order:
    """Represents a trading order"""
    id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    timestamp: datetime.datetime
    order_type: str = 'MARKET'


@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    timestamp: datetime.datetime
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class Trade:
    """Represents a completed trade"""
    id: str
    symbol: str
    side: str
    quantity: float
    entry_price: float
    exit_price: float
    entry_time: datetime.datetime
    exit_time: datetime.datetime
    pnl: float


@dataclass
class PortfolioState:
    """Current state of the portfolio"""
    cash: float
    positions: Dict[str, Position]
    total_value: float
    timestamp: datetime.datetime


@dataclass
class BacktestResult:
    """Result of a backtest run"""
    initial_capital: float
    final_capital: float
    total_return: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    max_drawdown: float
    sharpe_ratio: float
    trades: List[Trade]
    portfolio_history: List[PortfolioState]


class BacktestEngine:
    """
    Basic backtesting engine that processes historical data and executes trading strategies
    """
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_history = []
        self.order_history = []
        self.current_time = None
        
    def run_backtest(self, strategy, data_source, start_date: datetime.date, end_date: datetime.date) -> BacktestResult:
        """
        Run a backtest with the given strategy and data source
        """
        print(f"Starting backtest from {start_date} to {end_date}")
        print(f"Initial capital: ${self.initial_capital}")
        
        # Initialize portfolio state
        initial_state = PortfolioState(
            cash=self.initial_capital,
            positions={},
            total_value=self.initial_capital,
            timestamp=datetime.datetime.combine(start_date, datetime.time(0, 0))
        )
        self.portfolio_history.append(initial_state)
        
        # Process historical data
        for candle_data in data_source.get_data(start_date, end_date):
            self.current_time = candle_data['timestamp']
            
            # Execute strategy logic
            orders = strategy.generate_signals(candle_data, self.positions, self.current_capital)
            
            # Execute orders
            for order in orders:
                self.execute_order(order, candle_data['close'])
                
            # Update portfolio value
            portfolio_value = self.calculate_portfolio_value(candle_data)
            
            # Record portfolio state
            current_state = PortfolioState(
                cash=self.current_capital,
                positions=self.positions.copy(),
                total_value=portfolio_value,
                timestamp=self.current_time
            )
            self.portfolio_history.append(current_state)
        
        # Calculate final results
        final_capital = self.calculate_portfolio_value(data_source.get_latest_data())
        total_return = ((final_capital - self.initial_capital) / self.initial_capital) * 100
        
        result = BacktestResult(
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            total_trades=len(self.trades),
            winning_trades=len([t for t in self.trades if t.pnl > 0]),
            losing_trades=len([t for t in self.trades if t.pnl < 0]),
            max_drawdown=self.calculate_max_drawdown(),
            sharpe_ratio=self.calculate_sharpe_ratio(),
            trades=self.trades,
            portfolio_history=self.portfolio_history
        )
        
        return result
    
    def execute_order(self, order: Order, market_price: float):
        """
        Execute an order at the given market price
        """
        # Add to order history
        self.order_history.append(order)
        
        # For now, execute all orders at market price
        if order.side == 'BUY':
            cost = order.quantity * market_price
            if cost <= self.current_capital:
                self.current_capital -= cost
                
                # Update or create position
                if order.symbol in self.positions:
                    pos = self.positions[order.symbol]
                    # Average down
                    total_quantity = pos.quantity + order.quantity
                    avg_price = ((pos.quantity * pos.entry_price) + (order.quantity * market_price)) / total_quantity
                    pos.quantity = total_quantity
                    pos.entry_price = avg_price
                    pos.current_price = market_price
                else:
                    self.positions[order.symbol] = Position(
                        symbol=order.symbol,
                        quantity=order.quantity,
                        entry_price=market_price,
                        current_price=market_price,
                        timestamp=order.timestamp
                    )
        elif order.side == 'SELL':
            if order.symbol in self.positions and self.positions[order.symbol].quantity >= order.quantity:
                pos = self.positions[order.symbol]
                sell_value = order.quantity * market_price
                self.current_capital += sell_value
                
                # Calculate PnL
                pnl = (market_price - pos.entry_price) * order.quantity
                
                # Create trade record
                trade = Trade(
                    id=f"trade_{len(self.trades)}",
                    symbol=order.symbol,
                    side='BUY',  # Original direction
                    quantity=order.quantity,
                    entry_price=pos.entry_price,
                    exit_price=market_price,
                    entry_time=pos.timestamp,
                    exit_time=order.timestamp,
                    pnl=pnl
                )
                self.trades.append(trade)
                
                # Update position
                pos.quantity -= order.quantity
                pos.realized_pnl += pnl
                
                # Remove position if fully closed
                if pos.quantity == 0:
                    del self.positions[order.symbol]
    
    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value based on current prices
        """
        value = self.current_capital
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                value += position.quantity * current_prices[symbol]
        return value
    
    def calculate_max_drawdown(self) -> float:
        """
        Calculate maximum drawdown during the backtest period
        """
        if not self.portfolio_history:
            return 0.0
            
        peak = self.portfolio_history[0].total_value
        max_dd = 0.0
        
        for state in self.portfolio_history:
            if state.total_value > peak:
                peak = state.total_value
            dd = (peak - state.total_value) / peak * 100
            if dd > max_dd:
                max_dd = dd
                
        return max_dd
    
    def calculate_sharpe_ratio(self) -> float:
        """
        Calculate Sharpe ratio (simplified version)
        """
        if len(self.portfolio_history) < 2:
            return 0.0
            
        # Calculate returns
        returns = []
        for i in range(1, len(self.portfolio_history)):
            prev_val = self.portfolio_history[i-1].total_value
            curr_val = self.portfolio_history[i].total_value
            if prev_val != 0:
                ret = (curr_val - prev_val) / prev_val
                returns.append(ret)
        
        if not returns:
            return 0.0
            
        # Calculate average return and volatility
        avg_return = sum(returns) / len(returns)
        volatility = (sum((r - avg_return)**2 for r in returns) / len(returns))**0.5
        
        if volatility == 0:
            return 0.0
            
        # Assuming risk-free rate of 0 for simplicity
        sharpe = avg_return / volatility
        
        # Annualize the ratio (assuming daily data)
        return sharpe * (252**0.5)  # 252 trading days per year


# Example usage and testing
if __name__ == "__main__":
    print("Backtest engine initialized")
    engine = BacktestEngine(initial_capital=10000.0)
    print(f"Engine ready with initial capital: ${engine.initial_capital}")