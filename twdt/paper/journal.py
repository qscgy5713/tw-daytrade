"""模擬單成交紀錄(CSV),每個交易日一個檔。"""
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

from twdt.backtest.engine import Trade

JOURNAL_DIR = Path(os.environ.get("TWDT_CACHE", Path(__file__).resolve().parents[2] / "cache")) / "paper"


def save_trades(trades: List[Trade], day: str, directory: Optional[Path] = None) -> Optional[Path]:
    """寫入 trades_<day>.csv;同一天重跑會覆寫(模擬單每天只有一次完整 session)。"""
    if not trades:
        return None
    directory = directory or JOURNAL_DIR
    directory.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{**t.__dict__, "net_pnl": t.net_pnl} for t in trades])
    path = directory / f"trades_{day}.csv"
    df.to_csv(path, index=False)
    return path


def load_trades(directory: Optional[Path] = None) -> pd.DataFrame:
    files = sorted((directory or JOURNAL_DIR).glob("trades_*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.concat(pd.read_csv(f, parse_dates=["entry_time", "exit_time"]) for f in files)
