# AGENTS.md

## Cursor Cloud specific instructions

Autoportfolio is a single-product **Python quant library** (portfolio optimization + rolling
backtesting). There is no web server, database, or long-running service — everything is
invoke-and-exit via Python. See `README.md` for the module map and usage examples.

### Environment
- Runs on Python 3.12 inside a virtualenv at `.venv/` (system Python is PEP 668
  "externally managed", so a venv is required). The update script creates/refreshes it.
- Always use the venv interpreter: `.venv/bin/python` and `.venv/bin/pip` (or
  `source .venv/bin/activate` first). Do NOT `pip install` against system Python.
- System packages `python3-dev` and `python3.12-venv` are preinstalled in the VM image;
  `cvxpy==1.4.2` has no cp312 wheel and is compiled from source on first install (needs the
  headers + gcc/g++, already present).

### Test / run
- Tests: `.venv/bin/python -m pytest tests/ -q` (229 tests, all offline/synthetic data).
- Experiments (`python -m experiments.experiment_*`), the README quick-start, and the notebook
  fetch live prices from Yahoo Finance via `yfinance`. **Yahoo Finance egress is blocked in the
  Cloud VM** (downloads return empty and `data_client` raises "All data sources exhausted").
  Demonstrate/verify core functionality with synthetic data instead (this is what the test
  suite and `src/*` module `__main__` demos do). To run the live path, provide a CSV fallback
  at `data/raw_prices_backup.csv`.
- No linter/formatter is configured (no ruff/flake8/black/pyproject/setup.cfg). Use
  `.venv/bin/python -m compileall src experiments tests` for a quick syntax sanity check.

### Gotchas
- Import modules via the `src` package (e.g. `from src.portfolio_optimizer import optimize`);
  root-level duplicate `.py` files mirror `src/` but `src.*` is canonical.
- `optimize(mu, sigma, ...)` expects `mu` as a `pd.Series` and `sigma` as a `pd.DataFrame`
  (indexed by ticker), not raw numpy arrays.
- The Plotly HTML dashboard (`src/dashboard_results.py`) needs `plotly`, which is intentionally
  **not** in `requirements.txt`; `pip install plotly` on demand if you need it.
