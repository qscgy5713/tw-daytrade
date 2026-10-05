"""單檔股票的日內回測引擎。

輸入為單日分鐘線(DatetimeIndex,欄位 open/high/low/close/volume)。
訊號在 bar i 收盤後產生,於 bar i+1 開盤成交,避免未來函數。
同一根 bar 同時碰到停損與停利時,保守地視為先停損。
"""
from dataclasses import dataclass
from typing import Callable, List, Optional

import pandas as pd

from twdt.backtest.costs import CostModel
from twdt.risk.rules import RiskConfig, stop_price, target_price

# 訊號函式:給定到 bar i 為止的資料,回傳 1(做多)、-1(放空)、0(無)
SignalFn = Callable[[pd.DataFrame], int]


@dataclass
class Trade:
    symbol: str
    direction: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: int
    gross_pnl: float
    cost: float
    exit_reason: str

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.cost


def _exit_check(bar: pd.Series, direction: int, stop: float, target: float) -> Optional[tuple]:
    """回傳 (成交價, 原因) 或 None。先檢查停損。"""
    if direction == 1:
        if bar["low"] <= stop:
            return min(stop, bar["open"]), "stop"
        if bar["high"] >= target:
            return max(target, bar["open"]), "target"
    else:
        if bar["high"] >= stop:
            return max(stop, bar["open"]), "stop"
        if bar["low"] <= target:
            return min(target, bar["open"]), "target"
    return None


def run_day(
    symbol: str,
    bars: pd.DataFrame,
    signal_fn: SignalFn,
    cost: CostModel,
    risk: RiskConfig,
    daily_pnl_so_far: float = 0.0,
) -> List[Trade]:
    """跑一檔股票一天,同一時間最多一筆部位。"""
    trades: List[Trade] = []
    pnl_today = daily_pnl_so_far
    n = len(bars)
    i = 0
    while i < n - 1:
        if pnl_today <= -risk.max_daily_loss:
            break
        direction = signal_fn(bars.iloc[: i + 1])
        if direction == 0:
            i += 1
            continue

        entry_i = i + 1
        entry_bar = bars.iloc[entry_i]
        # 以實際進場那根 bar 的時間判斷截止,而非訊號 bar
        if entry_bar.name.time() > risk.no_entry_after or entry_bar.name.time() >= risk.force_close_at:
            break
        side = "buy" if direction == 1 else "sell"
        entry = cost.fill_price(entry_bar["open"], side)
        stop = stop_price(entry, direction, risk)
        target = target_price(entry, direction, risk)

        exit_price, reason, exit_i = None, "", entry_i
        for j in range(entry_i, n):
            bar = bars.iloc[j]
            # 進場那根 bar 也要檢查(開盤後同根觸及停損)
            hit = _exit_check(bar, direction, stop, target)
            if hit is not None:
                raw, reason = hit
                # 停利是限價單,不吃滑價;停損才套用不利滑價
                exit_price = raw if reason == "target" else cost.fill_price(
                    raw, "sell" if direction == 1 else "buy")
                exit_i = j
                break
            if bar.name.time() >= risk.force_close_at or j == n - 1:
                exit_price = cost.fill_price(bar["close"], "sell" if direction == 1 else "buy")
                reason, exit_i = "force_close", j
                break

        gross = (exit_price - entry) * direction * risk.shares_per_trade
        c = cost.round_trip_cost(entry, exit_price, risk.shares_per_trade, direction)
        trade = Trade(symbol, direction, entry_bar.name, bars.index[exit_i], entry,
                      exit_price, risk.shares_per_trade, gross, c, reason)
        trades.append(trade)
        pnl_today += trade.net_pnl
        i = exit_i + 1
    return trades
