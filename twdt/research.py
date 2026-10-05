"""選股與參數的樣本外驗證工具:用前段選、後段驗證,避免用同一段資料挑出「看起來最好」的結果。"""
from itertools import product
from typing import Dict, List, Tuple

import pandas as pd

from twdt.backtest.costs import CostModel
from twdt.backtest.engine import run_day
from twdt.risk.rules import RiskConfig
from twdt.signals.rules import opening_range_breakout


def split_days(data: Dict[str, pd.DataFrame], n_dev: int) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], List]:
    """依「所有標的共同的交易日」排序後切成前 n_dev 天(開發段)與其餘(驗證段)。

    只用各標的都有資料的日子,避免某檔缺資料時被誤判成開發段/驗證段不一致。
    """
    day_sets = [set(df.index.date) for df in data.values() if len(df)]
    common = sorted(set.intersection(*day_sets)) if day_sets else []
    if not 0 < n_dev < len(common):
        raise ValueError(f"n_dev 必須介於 1 與 {len(common) - 1} 之間(共同交易日 {len(common)} 天)")
    dev_days, test_days = set(common[:n_dev]), set(common[n_dev:])
    pick = lambda days: {s: df[[d in days for d in df.index.date]] for s, df in data.items()}
    return pick(dev_days), pick(test_days), common


def run_symbol(df: pd.DataFrame, symbol: str, cost: CostModel, risk: RiskConfig,
               or_minutes: int = 15, signal=None) -> pd.DataFrame:
    """單檔回測,回傳逐筆交易(含以進場金額為分母的報酬率,方便不同價位的股票互相比較)。

    signal 為 None 時用開盤區間突破(or_minutes);也可傳入任何 SignalFn。
    """
    signal = signal or opening_range_breakout(or_minutes)
    rows = []
    for _, day_bars in df.groupby(df.index.date):
        for t in run_day(symbol, day_bars, signal, cost, risk):
            notional = t.entry_price * t.shares
            rows.append({"symbol": symbol, "dir": t.direction, "day": t.entry_time.date(), "reason": t.exit_reason,
                         "entry_time": t.entry_time, "net": t.net_pnl, "gross": t.gross_pnl,
                         "cost": t.cost, "ret": t.net_pnl / notional,
                         "gross_ret": t.gross_pnl / notional, "cost_ret": t.cost / notional})
    return pd.DataFrame(rows)


def evaluate_grid(data: Dict[str, pd.DataFrame], cost: CostModel, stops: List[float],
                  targets: List[float], or_minutes: int = 15, min_trades: int = 20) -> pd.DataFrame:
    """每個 (標的, 停損, 停利) 組合一列;交易筆數不足 min_trades 的組合不納入排名。"""
    out = []
    for sym, df in data.items():
        for stop, target in product(stops, targets):
            tr = run_symbol(df, sym, cost, RiskConfig(stop_loss_pct=stop, take_profit_pct=target),
                            or_minutes)
            n = len(tr)
            out.append({"symbol": sym, "stop": stop, "target": target, "trades": n,
                        "net_ret_sum": tr["ret"].sum() if n else 0.0,
                        "net_pnl": tr["net"].sum() if n else 0.0,
                        "win_rate": (tr["net"] > 0).mean() if n else 0.0,
                        "eligible": n >= min_trades})
    return pd.DataFrame(out)


def pick_best(grid: pd.DataFrame) -> pd.Series:
    """以「淨報酬率總和」選最好的合格組合;沒有任何合格組合時回報錯誤。"""
    ok = grid[grid["eligible"]]
    if ok.empty:
        raise ValueError("沒有任何組合達到最低交易筆數")
    return ok.sort_values("net_ret_sum", ascending=False).iloc[0]
