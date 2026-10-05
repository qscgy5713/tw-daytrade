"""其他日內策略訊號(參數都在事前固定,不依回測結果調整)。

訊號函式簽名同 engine.SignalFn:收到「當天到目前為止已完成的 bar」,回傳 1 做多 / -1 放空 / 0 無。
下單一律在下一根 bar 開盤成交,不會用到未來資料。
"""
from datetime import time
from typing import Dict

import pandas as pd


def open_move(threshold: float = 0.01, trigger: time = time(9, 25), direction: int = 1):
    """開盤動能(direction=1 順勢)/反轉(direction=-1 逆勢):

    在 trigger 那根 bar 完成時(5 分鐘線的 09:25 bar 於 09:30 收盤),
    若價格相對當日開盤漲/跌超過 threshold,順勢(或逆勢)做。每天最多觸發一次。
    """
    def signal(bars: pd.DataFrame) -> int:
        if bars.index[-1].time() != trigger:
            return 0
        move = bars["close"].iloc[-1] / bars["open"].iloc[0] - 1
        if move > threshold:
            return direction
        if move < -threshold:
            return -direction
        return 0
    return signal


def gap_fade(prev_close_by_date: Dict, threshold: float = 0.015, trigger: time = time(9, 10)):
    """跳空回補:開盤跳空超過 threshold,且在 trigger 那根 bar 完成時缺口仍未回補,就朝回補方向做。

    prev_close_by_date:{日期: 前一交易日收盤價};沒有前收盤(第一天)就不交易。
    """
    def signal(bars: pd.DataFrame) -> int:
        if bars.index[-1].time() != trigger:
            return 0
        prev = prev_close_by_date.get(bars.index[0].date())
        if not prev:
            return 0
        gap = bars["open"].iloc[0] / prev - 1
        last = bars["close"].iloc[-1]
        if gap > threshold and last > prev:      # 向上跳空且還沒回補 -> 放空
            return -1
        if gap < -threshold and last < prev:     # 向下跳空且還沒回補 -> 做多
            return 1
        return 0
    return signal


def vwap_fade(threshold: float = 0.015, start: time = time(9, 30)):
    """VWAP 回歸:價格「剛剛」偏離當日 VWAP 超過 threshold 時往回做(前一根還在範圍內才觸發)。"""
    def signal(bars: pd.DataFrame) -> int:
        if len(bars) < 2 or bars.index[-1].time() < start:
            return 0
        vol = bars["volume"]
        if vol.sum() <= 0:
            return 0
        tp = (bars["high"] + bars["low"] + bars["close"]) / 3
        vwap = (tp * vol).cumsum() / vol.cumsum()
        dev = bars["close"] / vwap - 1
        d_last, d_prev = dev.iloc[-1], dev.iloc[-2]
        if pd.isna(d_last) or pd.isna(d_prev):
            return 0
        if d_last > threshold and d_prev <= threshold:
            return -1
        if d_last < -threshold and d_prev >= -threshold:
            return 1
        return 0
    return signal


def new_extreme(after: time = time(10, 0)):
    """日內新高/新低突破:after 之後,收盤價突破先前所有 bar 的最高/最低就順勢做。"""
    def signal(bars: pd.DataFrame) -> int:
        if len(bars) < 2 or bars.index[-1].time() < after:
            return 0
        last = bars["close"].iloc[-1]
        if last > bars["high"].iloc[:-1].max():
            return 1
        if last < bars["low"].iloc[:-1].min():
            return -1
        return 0
    return signal


def random_at(trigger: time = time(9, 25), seed: int = 0):
    """對照組:固定時間隨機方向進場。沒有任何預測力,用來衡量成本本身的拖累。"""
    import random
    rng = random.Random(seed)

    def signal(bars: pd.DataFrame) -> int:
        if bars.index[-1].time() != trigger:
            return 0
        return rng.choice([1, -1])
    return signal


def prev_close_map(df: pd.DataFrame) -> Dict:
    """{日期: 前一個交易日最後一根 bar 的收盤價}。"""
    last_close = df.groupby(df.index.date)["close"].last()
    days = list(last_close.index)
    return {days[i]: float(last_close.iloc[i - 1]) for i in range(1, len(days))}
