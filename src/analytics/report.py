"""
Analytics Report Module

Generates reports and visualizations for trading strategy performance analysis.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import sys

from .metrics import calculate_drawdown_series, calculate_sortino_ratio, calculate_sharpe_ratio


@dataclass
class PerformanceMetrics:
    """Performance metrics for a trading strategy"""
    total_return: float
    total_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    total_volume: float


class AnalyticsReport:
    """
    Analytics report generator for trading strategy performance
    """
    
    def __init__(self):
        self.metrics = {}
        self.charts = {}
        
    def generate_report(self, backtest_result: Any) -> Dict[str, Any]:
        """
        Generate a comprehensive analytics report from backtest results
        """
        print("Generating analytics report...")

        equity_curve = self._generate_equity_curve(backtest_result)
        drawdown_curve = self._generate_drawdown_curve(backtest_result)
        returns_series = self._generate_returns_series(backtest_result)
        monthly_returns = self._generate_monthly_returns(returns_series)

        # Extract key information from backtest result
        report_data = {
            'summary': self._generate_summary(backtest_result),
            'performance_metrics': self._calculate_performance_metrics(backtest_result),
            'trade_analysis': self._analyze_trades(backtest_result),
            'risk_metrics': self._calculate_risk_metrics(backtest_result, returns_series),
            'equity_curve': equity_curve,
            'drawdown_curve': drawdown_curve,
            'returns_series': returns_series,
            'monthly_returns': monthly_returns,
        }

        return report_data
    
    def _generate_summary(self, backtest_result: Any) -> Dict[str, Any]:
        """Generate summary statistics"""
        total_return_pct = backtest_result.total_return
        net_profit = backtest_result.final_capital - backtest_result.initial_capital
        total_trades = backtest_result.total_trades
        winning_trades = backtest_result.winning_trades
        losing_trades = backtest_result.losing_trades
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'initial_capital': backtest_result.initial_capital,
            'final_capital': backtest_result.final_capital,
            'net_profit': net_profit,
            'total_return_pct': total_return_pct,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate, 2),
            'max_drawdown': round(backtest_result.max_drawdown, 2),
            'sharpe_ratio': round(backtest_result.sharpe_ratio, 2)
        }
    
    def _calculate_performance_metrics(self, backtest_result: Any) -> PerformanceMetrics:
        """Calculate detailed performance metrics"""
        trades = backtest_result.trades
        
        if not trades:
            return PerformanceMetrics(
                total_return=backtest_result.total_return,
                total_trades=0,
                win_rate=0.0,
                profit_factor=0.0,
                max_drawdown=backtest_result.max_drawdown,
                sharpe_ratio=backtest_result.sharpe_ratio,
                avg_win=0.0,
                avg_loss=0.0,
                best_trade=0.0,
                worst_trade=0.0,
                total_volume=0.0
            )
        
        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl < 0]
        
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        profit_factor = total_wins / total_losses if total_losses != 0 else float('inf')
        
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        best_trade = max([t.pnl for t in trades]) if trades else 0
        worst_trade = min([t.pnl for t in trades]) if trades else 0
        
        total_volume = sum([abs(t.pnl) for t in trades])
        
        win_rate = len(wins) / len(trades) * 100 if trades else 0
        
        return PerformanceMetrics(
            total_return=backtest_result.total_return,
            total_trades=len(trades),
            win_rate=round(win_rate, 2),
            profit_factor=round(profit_factor, 2) if profit_factor != float('inf') else float('inf'),
            max_drawdown=round(backtest_result.max_drawdown, 2),
            sharpe_ratio=round(backtest_result.sharpe_ratio, 2),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            best_trade=round(best_trade, 2),
            worst_trade=round(worst_trade, 2),
            total_volume=round(total_volume, 2)
        )
    
    def _analyze_trades(self, backtest_result: Any) -> Dict[str, Any]:
        """Analyze individual trades"""
        trades = backtest_result.trades
        
        if not trades:
            return {
                'trade_count': 0,
                'avg_duration': 0,
                'pnl_distribution': [],
                'monthly_returns': {}
            }
        
        # Calculate trade durations
        durations = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in trades]  # hours
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        # PnL distribution
        pnl_values = [t.pnl for t in trades]
        
        # Monthly returns
        monthly_returns = {}
        for trade in trades:
            month_key = trade.exit_time.strftime('%Y-%m')
            if month_key not in monthly_returns:
                monthly_returns[month_key] = 0
            monthly_returns[month_key] += trade.pnl
        
        return {
            'trade_count': len(trades),
            'avg_duration_hours': round(avg_duration, 2),
            'pnl_distribution': pnl_values,
            'monthly_returns': monthly_returns,
            'largest_win': max(pnl_values) if pnl_values else 0,
            'largest_loss': min(pnl_values) if pnl_values else 0
        }
    
    def _calculate_risk_metrics(self, backtest_result: Any, returns_series: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate risk metrics"""
        if not returns_series:
            return {
                'max_drawdown': backtest_result.max_drawdown,
                'volatility': 0.0,
                'var_95': 0.0,
                'calmar_ratio': 0.0,
                'sortino_ratio': 0.0
            }

        returns = [r['return'] for r in returns_series]
        volatility = np.std(returns) * 100
        sorted_returns = sorted(returns)
        var_index = int(len(sorted_returns) * 0.05)
        var_95 = sorted_returns[var_index] if sorted_returns else 0
        calmar_ratio = (backtest_result.total_return / backtest_result.max_drawdown) if backtest_result.max_drawdown != 0 else 0
        sortino = calculate_sortino_ratio(returns)

        return {
            'max_drawdown': round(backtest_result.max_drawdown, 2),
            'volatility': round(volatility, 2),
            'var_95': round(var_95 * 100, 2),
            'calmar_ratio': round(calmar_ratio, 2),
            'sortino_ratio': round(sortino, 2)
        }
    
    def _generate_equity_curve(self, backtest_result: Any) -> List[Dict[str, Any]]:
        """Generate equity curve data points"""
        equity_points = []
        for state in backtest_result.portfolio_history:
            equity_points.append({
                'timestamp': state.timestamp.isoformat(),
                'value': state.total_value,
                'cash': state.cash
            })
        return equity_points

    def _generate_returns_series(self, backtest_result: Any) -> List[Dict[str, Any]]:
        """Generate per-period returns from portfolio history"""
        series = []
        history = backtest_result.portfolio_history
        for i in range(1, len(history)):
            prev = history[i-1].total_value
            cur = history[i].total_value
            if prev == 0:
                continue
            r = (cur - prev) / prev
            series.append({
                'timestamp': history[i].timestamp.isoformat(),
                'return': r
            })
        return series

    def _generate_drawdown_curve(self, backtest_result: Any) -> List[Dict[str, Any]]:
        """Generate drawdown series"""
        equity = [s.total_value for s in backtest_result.portfolio_history]
        dd = calculate_drawdown_series(equity)
        curve = []
        for idx, state in enumerate(backtest_result.portfolio_history):
            curve.append({
                'timestamp': state.timestamp.isoformat(),
                'drawdown': float(dd.iloc[idx]) if len(dd) > idx else 0.0
            })
        return curve

    def _generate_monthly_returns(self, returns_series: List[Dict[str, Any]]) -> Dict[str, float]:
        if not returns_series:
            return {}
        df = pd.DataFrame(returns_series)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['month'] = df['timestamp'].dt.to_period('M').astype(str)
        return df.groupby('month')['return'].sum().to_dict()
    
    def export_report(self, report_data: Dict[str, Any], filename: str = None) -> str:
        """
        Export the report to various formats (JSON, CSV, HTML)
        """
        if filename is None:
            filename = f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Export to JSON
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"Report exported to {filename}")
        return filename
    
    def generate_chart(self, chart_type: str, data: Any, title: str = "", save_path: str = None):
        """
        Generate a chart based on the specified type
        """
        plt.figure(figsize=(12, 6))

        if chart_type == "equity_curve":
            timestamps = [point['timestamp'] for point in data]
            values = [point['value'] for point in data]
            plt.plot(timestamps, values)
            plt.title(title or "Equity Curve")
            plt.xlabel("Date")
            plt.ylabel("Portfolio Value ($)")
            plt.xticks(rotation=45)

        elif chart_type == "drawdown_curve":
            timestamps = [point['timestamp'] for point in data]
            values = [point['drawdown'] * 100 for point in data]
            plt.fill_between(timestamps, values, 0, alpha=0.4)
            plt.title(title or "Drawdown (%)")
            plt.xlabel("Date")
            plt.ylabel("Drawdown (%)")
            plt.xticks(rotation=45)

        elif chart_type == "returns_histogram":
            returns = [r['return'] for r in data]
            plt.hist(returns, bins=30, edgecolor='black')
            plt.title(title or "Returns Distribution")
            plt.xlabel("Return")
            plt.ylabel("Frequency")

        elif chart_type == "monthly_returns":
            months = list(data.keys())
            returns = list(data.values())
            plt.bar(months, returns)
            plt.title(title or "Monthly Returns")
            plt.xlabel("Month")
            plt.ylabel("Return")
            plt.xticks(rotation=45)

        if save_path:
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            print(f"Chart saved to {save_path}")
        else:
            plt.show()

    def generate_default_charts(self, report_data: Dict[str, Any], output_dir: str):
        """Generate default chart pack to output_dir"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        self.generate_chart("equity_curve", report_data['equity_curve'], save_path=f"{output_dir}/equity_curve.png")
        self.generate_chart("drawdown_curve", report_data['drawdown_curve'], save_path=f"{output_dir}/drawdown_curve.png")
        self.generate_chart("returns_histogram", report_data['returns_series'], save_path=f"{output_dir}/returns_histogram.png")
        self.generate_chart("monthly_returns", report_data['monthly_returns'], save_path=f"{output_dir}/monthly_returns.png")


def print_report_summary(report_data: Dict[str, Any]):
    """
    Print a formatted summary of the report
    """
    summary = report_data['summary']
    metrics = report_data['performance_metrics']
    
    print("="*60)
    print("BACKTEST PERFORMANCE REPORT")
    print("="*60)
    print(f"Initial Capital:     ${summary['initial_capital']:,.2f}")
    print(f"Final Capital:       ${summary['final_capital']:,.2f}")
    print(f"Net Profit:          ${summary['net_profit']:,.2f}")
    print(f"Total Return:        {summary['total_return_pct']:.2f}%")
    print("-"*60)
    print(f"Total Trades:        {summary['total_trades']}")
    print(f"Win Rate:            {summary['win_rate']:.2f}%")
    print(f"Winning Trades:      {summary['winning_trades']}")
    print(f"Losing Trades:       {summary['losing_trades']}")
    print("-"*60)
    print(f"Max Drawdown:        {summary['max_drawdown']:.2f}%")
    print(f"Sharpe Ratio:        {summary['sharpe_ratio']:.2f}")
    print(f"Profit Factor:       {metrics.profit_factor}")
    print(f"Best Trade:          ${metrics.best_trade:.2f}")
    print(f"Worst Trade:         ${metrics.worst_trade:.2f}")
    print("="*60)


# Example usage
if __name__ == "__main__":
    print("Analytics report module loaded")
    analyzer = AnalyticsReport()
    print("Analyzer ready for generating reports")