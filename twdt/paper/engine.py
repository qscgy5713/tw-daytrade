"""模擬單引擎:逐筆報價驅動,與回測共用成本模型、風控規則與訊號函式。

與回測 run_day 的語意對齊:
- 訊號只在一根分鐘線「完成」後評估,並於下一筆報價(下一根開盤)成交。
  一筆報價跨好幾分鐘時,依序評估每一根新完成的 bar,取第一個非 0 的訊號。
- 出場那根 bar 完成時不評估訊號(回測是出場後從下一根 bar 才重新評估)。
- 停利是限價單,固定以停利價成交、不吃滑價。
- 13:25 起連續撮合結束:以 13:25 前最後一筆報價強制平倉(與回測相同)。

已知與回測/真實的差異:
- 停損:以輪詢看到的報價成交(再加滑價),反映「系統偵測到才送市價單」的延遲;
  回測用 bar 內的停損價,因此模擬單的停損會比回測更差,這是刻意保留的真實性。
- 真實的收盤平倉價在 13:30 集合競價撮合,與 13:25 前最後價會有誤差。
"""
from datetime import time as dtime
from typing import Callable, List, Optional

import pandas as pd

from twdt.backtest.costs import CostModel
from twdt.backtest.engine import SignalFn, Trade
from twdt.data.realtime import MinuteBarBuilder
from twdt.paper.portfolio import CapitalGuard
from twdt.risk.rules import RiskConfig, stop_price, target_price


class PaperTrader:
    def __init__(self, symbol: str, signal_fn: SignalFn, cost: CostModel, risk: RiskConfig,
                 session_open: dtime = dtime(9, 0),
                 started_at: Optional[pd.Timestamp] = None,
                 guard: Optional[CapitalGuard] = None,
                 on_event: Callable[[str], None] = lambda msg: None):
        self.symbol = symbol
        self.signal_fn = signal_fn
        self.cost = cost
        self.risk = risk
        self.session_open = session_open
        # 程式啟動時間(台灣當地、naive);用來判斷開盤區間資料是否完整,
        # 不能用「第一筆報價時間」,因為冷門股可能晚幾分鐘才有第一筆成交
        self.started_at = started_at
        # 單日買賣額度(多個 PaperTrader 共用同一個 guard);None 表示不檢查
        self.guard = guard
        self.on_event = on_event
        self.builder = MinuteBarBuilder()
        self.trades: List[Trade] = []
        self.position: Optional[dict] = None
        self.realized_pnl = 0.0
        self._date = None
        self._enabled: Optional[bool] = None
        self._last: Optional[tuple] = None  # (ts, price)
        self._last_continuous: Optional[float] = None  # 13:25 前最後一筆報價
        self._last_exit_minute: Optional[pd.Timestamp] = None

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
        new_bars = self.builder.bars[bars_before:]
        if ts.time() < self.risk.force_close_at:
            self._last_continuous = price

        exited_now = False
        if self.position is not None:
            exited_now = self._check_exit(ts, price)
        if new_bars and self.position is None and not exited_now and self._can_enter(ts):
            self._evaluate_new_bars(ts, price, new_bars)

    def finish(self, reason: str = "force_close") -> None:
        """收工:若還有部位,以最後一筆報價平倉(例如收盤前沒收到 13:25 後的報價)。

        reason 可傳 "interrupted" 標記人為中斷,事後統計才分得出來。
        """
        if self.position is not None and self._last is not None:
            ts, price = self._last
            self._exit(ts, self._exit_fill(price), reason)

    # ---- 內部 ----
    def _started_in_time(self, ts: pd.Timestamp) -> bool:
        # 開盤區間策略需要完整的 09:00 起算資料,盤中才啟動就不交易
        ref = self.started_at if self.started_at is not None else ts
        limit = pd.Timestamp.combine(ref.date(), self.session_open) + pd.Timedelta(minutes=2)
        return ref < limit

    def _can_enter(self, ts: pd.Timestamp) -> bool:
        # 與回測一致:用「進場那根 bar 的起始分鐘」比較,不是報價的秒數
        t = ts.floor("min").time()
        return bool(self._enabled
                    and t <= self.risk.no_entry_after
                    and t < self.risk.force_close_at
                    and self.realized_pnl > -self.risk.max_daily_loss)

    def _evaluate_new_bars(self, ts: pd.Timestamp, price: float, new_bars: List[dict]) -> None:
        frame = self.builder.to_frame()
        for bar in new_bars:
            # 出場那根 bar(含更早的)不評估:回測出場後從下一根 bar 才重新評估
            if self._last_exit_minute is not None and bar["time"] <= self._last_exit_minute:
                continue
            direction = self.signal_fn(frame[frame.index <= bar["time"]])
            if direction != 0:
                # 訊號只取第一個;額度不足被略過也算用掉(不改拿後面的 bar 的訊號補進場)
                self._enter(ts, price, direction)
                return

    def _exit_fill(self, price: float) -> float:
        side = "sell" if self.position["direction"] == 1 else "buy"
        return self.cost.fill_price(price, side)

    def _enter(self, ts: pd.Timestamp, price: float, direction: int) -> None:
        entry = self.cost.fill_price(price, "buy" if direction == 1 else "sell")
        notional = entry * self.risk.shares_per_trade
        if self.guard is not None and not self.guard.try_reserve(self.symbol, notional, ts):
            need = notional * (2 if self.guard.count_both_legs else 1)
            self.on_event(f"[{self.symbol}] {ts.time()} 訊號略過:單日額度不足"
                          f"(今日已用 {self.guard.used:,.0f} + 本筆需 {need:,.0f} > 上限 {self.guard.limit:,.0f})")
            return
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
            # 連續撮合已結束:以 13:25 前最後一筆報價平倉,與回測一致
            ref = self._last_continuous if self._last_continuous is not None else price
            self._exit(ts, self._exit_fill(ref), "force_close")
            return True
        if (d == 1 and price <= p["stop"]) or (d == -1 and price >= p["stop"]):
            self._exit(ts, self._exit_fill(price), "stop")
            return True
        if (d == 1 and price >= p["target"]) or (d == -1 and price <= p["target"]):
            # 限價單:連續競價中只會成交在自己的掛價,不吃滑價
            self._exit(ts, p["target"], "target")
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
        if self.guard is not None:
            self.guard.release(self.symbol, exit_notional=exit_price * shares)
        self._last_exit_minute = ts.floor("min")
        self.on_event(f"[{self.symbol}] {ts.time()} 出場({reason}) @ {exit_price:.2f}"
                      f" 淨損益 {trade.net_pnl:,.0f}")
