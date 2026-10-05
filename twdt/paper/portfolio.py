"""單日買賣額度:所有標的共用,逐筆累計、當日不循環使用(平倉不退還額度)。

兩種算法(count_both_legs):
- True(預設,保守):買進與賣出兩腿都占額度。進場時一次預留「進場金額 + 預估出場金額」,
  確保進場後額度一定夠賣出平倉,不會出現買了卻賣不掉;平倉時把預留的出場部分換成實際出場金額。
  這對應「額度用完後買賣都不行」的券商規則。
- False:只有新增部位的那一腿占額度(多家券商與金管會說明的摘要:先買後賣的買進、
  先賣後買的賣出;出場那一腿不算)。未逐條核對法規原文。

兩種算法下,額度都是「一天內所有進場的累計」,不是同時持有金額;停損後再進場一樣要再扣一次。
不確定國泰採哪一種,實盤前請向券商確認。這裡只模擬成交,不模擬「委託未成交也占額度」。
多檔同時出訊號時,先處理報價的先佔用額度(先到先得)。
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd


@dataclass(frozen=True)
class Rejection:
    ts: Optional[pd.Timestamp]
    symbol: str
    notional: float
    used: float


class CapitalGuard:
    def __init__(self, limit: float, count_both_legs: bool = True):
        if limit <= 0:
            raise ValueError("額度上限必須大於 0")
        self.limit = float(limit)
        self.count_both_legs = count_both_legs
        self.used = 0.0                      # 今日已累計占用的額度(平倉不退還)
        self.entries = 0                     # 今日已使用額度的進場次數
        self._open: Dict[str, float] = {}    # 持倉中的標的 -> 為出場預留的額度(entry-only 模式為 0)
        self.rejections: List[Rejection] = []

    @property
    def remaining(self) -> float:
        return self.limit - self.used

    def try_reserve(self, symbol: str, notional: float, ts: Optional[pd.Timestamp] = None) -> bool:
        """額度足夠就扣除並回傳 True(剛好等於上限允許);不足則記錄並回傳 False。"""
        if symbol in self._open:
            # 每檔同時最多一筆部位,正常流程不會發生;發生代表呼叫端有 bug,寧可擋下也不重複扣
            raise RuntimeError(f"{symbol} 仍有持倉,不可重複進場")
        exit_reserve = notional if self.count_both_legs else 0.0
        need = notional + exit_reserve
        if self.used + need > self.limit:
            self.rejections.append(Rejection(ts, symbol, need, self.used))
            return False
        self.used += need
        self.entries += 1
        self._open[symbol] = exit_reserve
        return True

    def release(self, symbol: str, exit_notional: Optional[float] = None) -> None:
        """出場:額度**不退還**(當沖額度當日不循環使用)。

        兩腿都算的模式下,把進場時預留的出場額度換成實際出場金額
        (出場價較高時實際金額會略大於預留,差額照實補記,出場本身不會被擋)。
        """
        reserved = self._open.pop(symbol, None)
        if reserved and exit_notional is not None:
            self.used += exit_notional - reserved

    def summary_lines(self) -> List[str]:
        mode = "買賣兩腿都算" if self.count_both_legs else "只算進場"
        lines = [f"單日買賣額度({mode}):上限 {self.limit:,.0f},今日已用 {self.used:,.0f}"
                 f"({self.used / self.limit:.0%}),進場 {self.entries} 次;"
                 f"因額度不足略過 {len(self.rejections)} 次訊號"]
        for r in self.rejections[:10]:
            lines.append(f"  略過 {r.symbol} @ {r.ts.time() if r.ts is not None else '?'}:"
                         f"本筆 {r.notional:,.0f} + 已用 {r.used:,.0f} > {self.limit:,.0f}")
        if len(self.rejections) > 10:
            lines.append(f"  …其餘 {len(self.rejections) - 10} 次省略")
        return lines
