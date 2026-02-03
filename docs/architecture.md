# Architecture

## Overview
The bot is split into independent modules with clear responsibilities.

## Modules
- **core/**: config, shared utilities, app lifecycle
- **data/**: Binance client, data ingestion, market data normalization
- **storage/**: SQLite access layer, schema, migrations
- **strategies/**: strategy plugins, signal generation
- **backtest/**: backtesting engine, portfolio simulation
- **learn/**: parameter optimization / self‑tuning
- **analytics/**: metrics, reports, statistics
- **telegram/**: bot commands and UX

## Data flow
1) Ingest market data → SQLite
2) Strategy computes signals → backtest engine
3) Engine writes results → analytics
4) Analytics generates reports → Telegram

## Tech
- Python (async)
- aiogram (Telegram)
- SQLite (local DB)
