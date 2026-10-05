"""單日買賣額度:所有標的共用,進場金額逐筆累計、當日不循環使用。

依台股券商的單日買賣額度規則(以多家券商與金管會說明的摘要為準,未逐條核對法規原文):
- 只有「新增部位」占用額度:現股買進(含先買後賣的買進)與先賣後買的賣出;出場那一腿不算。
- 當沖的進場金額列入額度後,平倉也不會把額度還回來,額度每天歸零。
- 因此額度是「一天內所有進場金額的總和」,不是同時持有金額;停損後再進場一樣要再扣一次。

這裡只模擬成交,不模擬「委託未成交也占額度」;實際算法各券商可能略有差異,實盤前請向券商確認。
多檔同時出訊號時,先處理報價的先佔用額度(先到先得)。
"""
from dataclasses import dataclass
from typing import List, Optional, Set

import pandas as pd


@dataclass(frozen=True)
class Rejection:
    ts: Optional[pd.Timestamp]
    symbol: str
    notional: float
    used: float


class CapitalGuard:
    def __init__(self, limit: float):
        if limit <= 0:
            raise ValueError("額度上限必須大於 0")
        self.limit = float(limit)
        self.used = 0.0                      # 今日已累計的進場金額(不因出場而減少)
        self.entries = 0                     # 今日已使用額度的進場次數
        self._open: Set[str] = set()         # 目前持倉中的標的,只用來抓重複進場的 bug
        self.rejections: List[Rejection] = []

    @property
    def remaining(self) -> float:
        return self.limit - self.used

    def try_reserve(self, symbol: str, notional: float, ts: Optional[pd.Timestamp] = None) -> bool:
        """額度足夠就扣除並回傳 True(剛好等於上限允許);不足則記錄並回傳 False。"""
        if symbol in self._open:
            # 每檔同時最多一筆部位,正常流程不會發生;發生代表呼叫端有 bug,寧可擋下也不重複扣
            raise RuntimeError(f"{symbol} 仍有持倉,不可重複進場")
        if self.used + notional > self.limit:
            self.rejections.append(Rejection(ts, symbol, notional, self.used))
            return False
        self.used += notional
        self.entries += 1
        self._open.add(symbol)
        return True

    def release(self, symbol: str) -> None:
        """出場:只解除「持倉中」標記,**不退還額度**(當沖額度當日不循環使用)。"""
        self._open.discard(symbol)

    def summary_lines(self) -> List[str]:
        lines = [f"單日買賣額度:上限 {self.limit:,.0f},今日已用 {self.used:,.0f}"
                 f"({self.used / self.limit:.0%}),進場 {self.entries} 次;"
                 f"因額度不足略過 {len(self.rejections)} 次訊號"]
        for r in self.rejections[:10]:
            lines.append(f"  略過 {r.symbol} @ {r.ts.time() if r.ts is not None else '?'}:"
                         f"本筆 {r.notional:,.0f} + 已用 {r.used:,.0f} > {self.limit:,.0f}")
        if len(self.rejections) > 10:
            lines.append(f"  …其餘 {len(self.rejections) - 10} 次省略")
        return lines
