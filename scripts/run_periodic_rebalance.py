#!/usr/bin/env python3
"""
run_periodic_rebalance.py — Rebalanceo periódico cada 15 días hábiles (sin broker).

Qué es esto
-----------
NO es un cron automático en la nube ni un bot conectado a un broker.
Es un **comando** que tú (o un scheduler tipo cron) ejecutas cuando toca
rebalancear. Usa el mismo motor de Fase 6:

  universo → precios (desde 2022) → régimen/señales/HRP → PositionLedger

Estado persistente en ``data/live_state/``:
  - ledger_lots.json / cash / realized_pnl
  - last_rebalance.txt
  - latest_snapshot.csv
  - rebalance_history.csv

Uso
---
  # Desde la raíz del repo, con venv activo:
  python -m scripts.run_periodic_rebalance

  # Forzar aunque no hayan pasado 15 días hábiles:
  python -m scripts.run_periodic_rebalance --force

  # Solo informar si toca rebalancear (exit 0 = toca, 1 = aún no):
  python -m scripts.run_periodic_rebalance --check-only

Cron ejemplo (cada día a las 16:00; el script decide si ya pasaron 15D):
  0 16 * * 1-5 cd /path/to/autoportfolio && .venv/bin/python -m scripts.run_periodic_rebalance
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.data_client import load_universe, fetch_prices, fetch_volumes, clean_prices
from src.position_ledger import PositionLedger, Lot
from src.regime_detector import classify_regime
from src.signal_stack import compute_signal_scores
from src.conviction_scoring import conviction_score, conviction_to_tilts
from src.hrp_sizing import hrp_weights
from src.rolling_engine import get_window

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("periodic_rebalance")

SEED = 42
DEFAULT_START = "2022-01-01"
REBALANCE_EVERY_N = 15          # trading days
WARMUP = 252
WINDOW = 252
STATE_DIR = Path("data/live_state")
INITIAL_CASH = 1_000_000.0


# ===================================================================
# State I/O
# ===================================================================
def _ensure_state_dir() -> Path:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return STATE_DIR


def save_ledger(ledger: PositionLedger, path: Path) -> None:
    """Persist ledger lots + cash + realized PnL to JSON."""
    lots_payload: List[dict] = []
    for tk, lots in ledger._lots.items():
        for lot in lots:
            lots_payload.append({
                "ticker": lot.ticker,
                "shares": lot.shares,
                "entry_price": lot.entry_price,
                "entry_date": (
                    lot.entry_date.isoformat()
                    if pd.notna(lot.entry_date) else None
                ),
            })
    payload = {
        "cash": ledger.cash,
        "realized_pnl": ledger.realized_pnl,
        "lots": lots_payload,
    }
    path.write_text(json.dumps(payload, indent=2))


def load_ledger(path: Path, initial_cash: float = INITIAL_CASH) -> PositionLedger:
    """Restore PositionLedger from JSON, or create a fresh one."""
    ledger = PositionLedger(initial_cash=initial_cash)
    if not path.exists():
        return ledger
    payload = json.loads(path.read_text())
    ledger.cash = float(payload.get("cash", initial_cash))
    ledger.realized_pnl = float(payload.get("realized_pnl", 0.0))
    ledger._lots = {}
    for row in payload.get("lots", []):
        ed = row.get("entry_date")
        entry_date = pd.Timestamp(ed) if ed else pd.NaT
        lot = Lot(
            ticker=row["ticker"],
            shares=float(row["shares"]),
            entry_price=float(row["entry_price"]),
            entry_date=entry_date,
        )
        ledger._lots.setdefault(lot.ticker, []).append(lot)
    return ledger


def read_last_rebalance(path: Path) -> Optional[pd.Timestamp]:
    if not path.exists():
        return None
    text = path.read_text().strip()
    if not text:
        return None
    return pd.Timestamp(text)


def write_last_rebalance(path: Path, t: pd.Timestamp) -> None:
    path.write_text(str(pd.Timestamp(t).date()))


def trading_days_between(
    prices: pd.DataFrame,
    a: pd.Timestamp,
    b: pd.Timestamp,
) -> int:
    """Count trading days in prices index strictly after a, up to and including b."""
    idx = prices.index
    mask = (idx > a) & (idx <= b)
    return int(mask.sum())


def should_rebalance(
    prices: pd.DataFrame,
    last: Optional[pd.Timestamp],
    every_n: int = REBALANCE_EVERY_N,
) -> bool:
    """True if never rebalanced, or >= every_n trading days since last."""
    if last is None:
        return True
    t = prices.index[-1]
    return trading_days_between(prices, last, t) >= every_n


# ===================================================================
# Target weights (Fase 6 stack)
# ===================================================================
def compute_target_weights(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    t: pd.Timestamp,
    window: int = WINDOW,
) -> tuple[pd.Series, str]:
    """Build HRP + conviction target weights at date t.

    Returns:
        (weights, regime_state)
    """
    wp = get_window(prices, t, window=window)
    # Drop columns with all-NaN in window
    wp = wp.dropna(axis=1, how="any")
    if wp.shape[1] < 3:
        raise RuntimeError(f"Insufficient assets in window at {t.date()}")

    active = wp.columns.tolist()
    rets = wp.pct_change().dropna(how="all")
    base = hrp_weights(rets)

    report = classify_regime(prices[active], t, window=63)
    scores = compute_signal_scores(prices, volumes, t)
    scores = scores.reindex(active).fillna(0.0)
    conv = conviction_score(scores, report.state)
    w = conviction_to_tilts(conv, base_weights=base.reindex(active).fillna(0.0))
    w = w.clip(lower=0.0)
    w = w / w.sum()
    return w, report.state


# ===================================================================
# Main step
# ===================================================================
def run_once(
    force: bool = False,
    start: str = DEFAULT_START,
    every_n: int = REBALANCE_EVERY_N,
    check_only: bool = False,
) -> int:
    """Execute one periodic-rebalance check/step.

    Returns:
        Process exit code (0 = rebalanced or check says due; 1 = not due).
    """
    np.random.seed(SEED)
    state = _ensure_state_dir()
    ledger_path = state / "ledger.json"
    last_path = state / "last_rebalance.txt"
    hist_path = state / "rebalance_history.csv"
    snap_path = state / "latest_snapshot.csv"

    logger.info("Loading universe + prices from %s …", start)
    tickers = load_universe()["ticker"].tolist()
    # end=None → yfinance hasta hoy: pass tomorrow-ish via omit; use today+
    end = (datetime.now(timezone.utc) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    prices_raw = fetch_prices(tickers, start=start, end=end)
    volumes_raw = fetch_volumes(tickers, start=start, end=end)
    prices, _ = clean_prices(prices_raw)
    volumes = volumes_raw.reindex(prices.index).ffill().fillna(1e6)
    volumes = volumes.reindex(columns=prices.columns).fillna(1e6)

    if len(prices) < WARMUP + 5:
        logger.error(
            "Not enough history (%d rows). Need >= %d trading days from %s.",
            len(prices), WARMUP + 5, start,
        )
        return 2

    t = pd.Timestamp(prices.index[-1])
    last = read_last_rebalance(last_path)
    due = should_rebalance(prices, last, every_n=every_n)

    logger.info(
        "As-of %s | last_rebalance=%s | trading_days_since=%s | due=%s",
        t.date(),
        last.date() if last is not None else "never",
        trading_days_between(prices, last, t) if last is not None else "n/a",
        due or force,
    )

    if check_only:
        return 0 if (due or force) else 1

    if not due and not force:
        logger.info(
            "Skip: fewer than %d trading days since last rebalance. "
            "Use --force to override.",
            every_n,
        )
        return 1

    ledger = load_ledger(ledger_path)
    w_target, regime = compute_target_weights(prices, volumes, t)
    prices_t = prices.loc[t]
    nav_pre = ledger.nav(prices_t)

    # Rough cost estimate (~7 bps on L1 turnover)
    w_prev = ledger.weights_from_positions(prices_t)
    if w_prev.empty:
        w_prev = pd.Series(0.0, index=w_target.index)
    aligned_prev = w_prev.reindex(w_target.index).fillna(0.0)
    turnover = float((w_target - aligned_prev).abs().sum())
    cost_abs = turnover * (7.0 / 10_000.0) * max(nav_pre, 1e-6)

    trades = ledger.trades_from_target_weights(w_target, prices_t, nav=nav_pre)
    ledger.apply_trades(trades, prices_t, costs=cost_abs, t=t)
    snap = ledger.snapshot(t, prices_t)

    save_ledger(ledger, ledger_path)
    write_last_rebalance(last_path, t)

    # Snapshot weights for audit
    w_now = ledger.weights_from_positions(prices_t)
    snap_df = pd.DataFrame({
        "shares": snap.positions,
        "avg_entry": snap.avg_entry,
        "weight": w_now,
    })
    snap_df.to_csv(snap_path)

    row = {
        "date": t.date().isoformat(),
        "regime": regime,
        "nav": snap.nav,
        "cash": snap.cash,
        "turnover": turnover,
        "cost": cost_abs,
        "realized_pnl": snap.realized_pnl,
        "unrealized_pnl": snap.unrealized_pnl,
        "n_positions": int((snap.positions > 0).sum()) if len(snap.positions) else 0,
        "forced": bool(force),
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    hist = pd.DataFrame([row])
    if hist_path.exists():
        prev = pd.read_csv(hist_path)
        hist = pd.concat([prev, hist], ignore_index=True)
    hist.to_csv(hist_path, index=False)

    logger.info(
        "REBALANCED %s | regime=%s | NAV=%.2f | turnover=%.3f | positions=%d",
        t.date(), regime, snap.nav, turnover, row["n_positions"],
    )
    print(
        f"\nOK rebalance {t.date()} | regime={regime} | "
        f"NAV={snap.nav:,.2f} | positions={row['n_positions']}\n"
        f"State → {state}/\n"
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Periodic 15-trading-day portfolio rebalance (no broker).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rebalance even if <15 trading days since last run.",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Exit 0 if a rebalance is due, 1 otherwise (no trades).",
    )
    parser.add_argument(
        "--start", default=DEFAULT_START,
        help=f"History start date (default {DEFAULT_START}).",
    )
    parser.add_argument(
        "--every-n", type=int, default=REBALANCE_EVERY_N,
        help=f"Trading days between rebalances (default {REBALANCE_EVERY_N}).",
    )
    args = parser.parse_args(argv)
    return run_once(
        force=args.force,
        start=args.start,
        every_n=args.every_n,
        check_only=args.check_only,
    )


if __name__ == "__main__":
    sys.exit(main())
