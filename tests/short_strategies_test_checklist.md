# Short Strategies/Metrics Test Checklist

1) SMA indicator
- Input: [1,2,3,4,5], window=3 → expected [None,None,2,3,4]

2) RSI bounds
- Ensure RSI in [0,100] for random series

3) EMA crossover signals
- Generate BUY on fast EMA crossing above slow EMA

4) RSI reversion signals
- Oversold (<30) → BUY, overbought (>70) → SELL

5) Sharpe ratio
- Known returns → expected Sharpe

6) Max drawdown
- Known equity curve → expected drawdown

7) Multi-indicator confirmation
- Signal only when EMA crossover + RSI condition both satisfied

8) Edge cases
- Empty data / too few points returns no signals
