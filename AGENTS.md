# AGENTS.md

## Cursor Cloud specific instructions

Autoportfolio is a single-product **Python quant library** (portfolio optimization +
rolling backtesting). There is no web server, database, or long-running service —
everything is invoke-and-exit via Python. See `README.md` for the module map and
usage examples.

### Environment

- Run on Python 3.10–3.12 inside a virtualenv at `.venv/` (system Python is PEP 668
  "externally managed", so a venv is required).
- Always use the venv interpreter: `.venv/bin/python` and `.venv/bin/pip` (or
  `source .venv/bin/activate` first). Do NOT `pip install` against system Python.
- Install: `python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -r requirements.txt`
- `requirements.txt` uses compatible ranges (not exact pins). Keep `plotly` in that
  file; the HTML dashboard (`src/dashboard_results.py`) depends on it.

### Test / run

- Tests: `.venv/bin/python -m pytest tests/ -q` (offline/synthetic data; no network).
- Experiments (`python -m experiments.experiment_*`), the README quick-start, and the
  notebook fetch live prices from Yahoo Finance via `yfinance`. If downloads return
  empty, `data_client` raises "All data sources exhausted". Demonstrate/verify core
  functionality with synthetic data instead (this is what the test suite and
  `src/*` module `__main__` demos do). To run the live path without Yahoo, provide a
  CSV fallback at `data/raw_prices_backup.csv`.
- No linter/formatter is configured (no ruff/flake8/black/pyproject/setup.cfg). Use
  `.venv/bin/python -m compileall src experiments tests` for a quick syntax check.

### Gotchas

- Import modules via the `src` package (e.g. `from src.portfolio_optimizer import optimize`);
  root-level duplicate `.py` files mirror `src/` but `src.*` is canonical.
- `optimize(mu, sigma, ...)` expects `mu` as a `pd.Series` and `sigma` as a
  `pd.DataFrame` (indexed by ticker), not raw numpy arrays.
- Rebalance frequency `"15D"` counts **business days**, not calendar days
  (`get_rebalance_dates` in `src/rolling_engine.py`).
