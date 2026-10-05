"""模擬單引擎:逐筆報價驅動,與回測共用成本模型、風控規則與訊號函式。

與回測 run_day 的語意對齊:
- 訊號只在一根分鐘線「完成」後評估,並於下一筆報價(下一根開盤)成交。
- 出場後不會在同一筆報價立刻反向進場(回測也是出場後下一根才重新評估)。
- 停損吃不利滑價(以當下報價成交);停利是限價單,不吃滑價。

已知與真實的差異:13:25 後是收盤集合競價,實際成交價在 13:30 撮合,
這裡以 13:25 之後第一筆報價強制平倉,會與真實成交價有小誤差。
"""
from datetime import time as dtime
from typing import Callable, List, Optional

import pandas as pd

from twdt.backtest.costs import CostModel
from twdt.backtest.engine import SignalFn, Trade
from twdt.data.realtime import MinuteBarBuilder
from twdt.risk.rules import RiskConfig, stop_price, target_price


class PaperTrader:
    def __init__(self, symbol: str, signal_fn: SignalFn, cost: CostModel, risk: RiskConfig,
                 session_open: dtime = dtime(9, 0),
                 on_event: Callable[[str], None] = lambda msg: None):
        self.symbol = symbol
        self.signal_fn = signal_fn
        self.cost = cost
        self.risk = risk
        self.session_open = session_open
        self.on_event = on_event
        self.builder = MinuteBarBuilder()
        self.trades: List[Trade] = []
        self.position: Optional[dict] = None
        self.realized_pnl = 0.0
        self._date = None
        self._enabled: Optional[bool] = None
        self._last: Optional[tuple] = None  # (ts, price)

    # ---- 對外介面 ----
    def on_quote(self, ts: pd.Timestamp, price: float, cum_volume: int) -> None:
        if self._date is None:
            self._date = ts.date()
            self._enabled = self._started_in_time(ts)
            if not self._enabled:
                self.on_event(f"[{self.symbol}] 開盤區間資料不完整(啟動於 {ts.time()}),今天不交易")
        elif ts.date() != self._date:
            return
        if self._last is not None and ts < self._last[0]:
            return  # 時間倒退的報價(重複或亂序)
        self._last = (ts, price)

        bars_before = len(self.builder.bars)
        self.builder.update(ts, price, cum_volume)
        bar_completed = len(self.builder.bars) > bars_before

        exited_now = False
        if self.position is not None:
            exited_now = self._check_exit(ts, price)
        if bar_completed and self.position is None and not exited_now and self._can_enter(ts):
            direction = self.signal_fn(self.builder.to_frame())
            if direction != 0:
                self._enter(ts, price, direction)

    def finish(self) -> None:
        """收工:若還有部位,以最後一筆報價平倉(例如收盤前沒收到 13:25 後的報價)。"""
        if self.position is not None and self._last is not None:
            ts, price = self._last
            self._exit(ts, self._exit_fill(price), "force_close")

    # ---- 內部 ----
    def _started_in_time(self, ts: pd.Timestamp) -> bool:
        # 開盤區間策略需要完整的 09:00 起算資料,盤中才啟動就不交易
        limit = pd.Timestamp.combine(ts.date(), self.session_open) + pd.Timedelta(minutes=2)
        return ts < limit

    def _can_enter(self, ts: pd.Timestamp) -> bool:
        t = ts.time()
        return bool(self._enabled
                    and t <= self.risk.no_entry_after
                    and t < self.risk.force_close_at
                    and self.realized_pnl > -self.risk.max_daily_loss)

    def _exit_fill(self, price: float) -> float:
        side = "sell" if self.position["direction"] == 1 else "buy"
        return self.cost.fill_price(price, side)

    def _enter(self, ts: pd.Timestamp, price: float, direction: int) -> None:
        entry = self.cost.fill_price(price, "buy" if direction == 1 else "sell")
        self.position = {"direction": direction, "entry": entry, "entry_time": ts,
                         "stop": stop_price(entry, direction, self.risk),
                         "target": target_price(entry, direction, self.risk)}
        self.on_event(f"[{self.symbol}] {ts.time()} 進場 {'多' if direction == 1 else '空'}"
                      f" @ {entry:.2f} 停損 {self.position['stop']:.2f}"
                      f" 停利 {self.position['target']:.2f}")

    def _check_exit(self, ts: pd.Timestamp, price: float) -> bool:
        p = self.position
        d = p["direction"]
        if ts.time() >= self.risk.force_close_at:
            self._exit(ts, self._exit_fill(price), "force_close")
            return True
        if (d == 1 and price <= p["stop"]) or (d == -1 and price >= p["stop"]):
            self._exit(ts, self._exit_fill(price), "stop")
            return True
        if (d == 1 and price >= p["target"]) or (d == -1 and price <= p["target"]):
            # 限價單:價格跳過停利價時以更好的價格成交,不吃滑價
            fill = max(p["target"], price) if d == 1 else min(p["target"], price)
            self._exit(ts, fill, "target")
            return True
        return False

    def _exit(self, ts: pd.Timestamp, exit_price: float, reason: str) -> None:
        p = self.position
        d, shares = p["direction"], self.risk.shares_per_trade
        gross = (exit_price - p["entry"]) * d * shares
        cost = self.cost.round_trip_cost(p["entry"], exit_price, shares, d)
        trade = Trade(self.symbol, d, p["entry_time"], ts, p["entry"], exit_price,
                      shares, gross, cost, reason)
        self.trades.append(trade)
        self.realized_pnl += trade.net_pnl
        self.position = None
        self.on_event(f"[{self.symbol}] {ts.time()} 出場({reason}) @ {exit_price:.2f}"
                      f" 淨損益 {trade.net_pnl:,.0f}")
