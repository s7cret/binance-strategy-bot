-- SQLite schema

PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS markets (
  id INTEGER PRIMARY KEY,
  symbol TEXT NOT NULL,
  base_asset TEXT,
  quote_asset TEXT,
  status TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS candles (
  id INTEGER PRIMARY KEY,
  symbol TEXT NOT NULL,
  interval TEXT NOT NULL,
  open_time INTEGER NOT NULL,
  open REAL, high REAL, low REAL, close REAL,
  volume REAL,
  close_time INTEGER,
  UNIQUE(symbol, interval, open_time)
);

CREATE TABLE IF NOT EXISTS strategies (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  params_json TEXT NOT NULL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS backtest_runs (
  id INTEGER PRIMARY KEY,
  strategy_id INTEGER NOT NULL,
  symbol TEXT NOT NULL,
  interval TEXT NOT NULL,
  start_ts INTEGER,
  end_ts INTEGER,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(strategy_id) REFERENCES strategies(id)
);

CREATE TABLE IF NOT EXISTS backtest_trades (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL,
  ts INTEGER NOT NULL,
  side TEXT,
  price REAL,
  qty REAL,
  fee REAL,
  pnl REAL,
  FOREIGN KEY(run_id) REFERENCES backtest_runs(id)
);

CREATE TABLE IF NOT EXISTS metrics (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL,
  name TEXT NOT NULL,
  value REAL,
  FOREIGN KEY(run_id) REFERENCES backtest_runs(id)
);

CREATE TABLE IF NOT EXISTS optimization_runs (
  id INTEGER PRIMARY KEY,
  strategy_id INTEGER NOT NULL,
  method TEXT NOT NULL,
  best_params_json TEXT,
  score REAL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(strategy_id) REFERENCES strategies(id)
);
