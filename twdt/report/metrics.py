"""績效報表。"""
from typing import Iterable

import pandas as pd

from twdt.backtest.engine import Trade


def summarize(trades: Iterable[Trade]) -> dict:
    rows = [{"date": t.entry_time.date(), "exit_time": t.exit_time, "gross": t.gross_pnl,
             "cost": t.cost, "net": t.net_pnl} for t in trades]
    if not rows:
        return {"trades": 0}
    # 多檔時成交是逐檔串起來的,必須依出場時間排序,權益曲線與回撤才有意義
    df = pd.DataFrame(rows).sort_values("exit_time", kind="stable").reset_index(drop=True)
    # 以 0 為起點,否則第一筆就虧損時回撤會被漏算
    equity = pd.concat([pd.Series([0.0]), df["net"].cumsum()], ignore_index=True)
    drawdown = (equity - equity.cummax()).min()
    wins = df[df["net"] > 0]["net"]
    losses = df[df["net"] <= 0]["net"]
    gross_total = df["gross"].sum()
    return {
        "trades": len(df),
        "net_pnl": df["net"].sum(),
        "gross_pnl": gross_total,
        "total_cost": df["cost"].sum(),
        # 成本吃掉毛利的比例;毛利 <= 0 時無意義
        "cost_share_of_gross": df["cost"].sum() / gross_total if gross_total > 0 else None,
        "win_rate": len(wins) / len(df),
        "expectancy": df["net"].mean(),
        "profit_factor": wins.sum() / -losses.sum() if losses.sum() < 0 else None,
        "max_drawdown": drawdown,
        "days": df["date"].nunique(),
    }
