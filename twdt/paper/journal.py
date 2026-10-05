"""模擬單成交紀錄(CSV),每個交易日一個檔。"""
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

from twdt.backtest.engine import Trade

JOURNAL_DIR = Path(os.environ.get("TWDT_CACHE", Path(__file__).resolve().parents[2] / "cache")) / "paper"


def save_trades(trades: List[Trade], day: str, directory: Optional[Path] = None) -> Optional[Path]:
    """寫入 trades_<day>.csv;若檔案已存在(例如兩個行程各跑不同標的)則合併去重,不覆寫。"""
    if not trades:
        return None
    directory = directory or JOURNAL_DIR
    directory.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{**t.__dict__, "net_pnl": t.net_pnl} for t in trades])
    path = directory / f"trades_{day}.csv"
    if path.exists():
        old = pd.read_csv(path, parse_dates=["entry_time", "exit_time"], dtype={"symbol": str})
        df = pd.concat([old, df], ignore_index=True)
        df = df.drop_duplicates(subset=["symbol", "entry_time", "exit_time"], keep="last")
        df = df.sort_values("exit_time", kind="stable")
    df.to_csv(path, index=False)
    return path


def load_trades(directory: Optional[Path] = None) -> pd.DataFrame:
    files = sorted((directory or JOURNAL_DIR).glob("trades_*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.concat(pd.read_csv(f, parse_dates=["entry_time", "exit_time"], dtype={"symbol": str})
                     for f in files)
