"""規則式基準策略。"""
from datetime import time

import pandas as pd


def opening_range_breakout(or_minutes: int = 15, session_open: time = time(9, 0)):
    """開盤區間突破:區間結束後,收盤價突破區間高點做多、跌破低點放空。"""

    def signal(bars: pd.DataFrame) -> int:
        # normalize() 保留時區,tz-aware 與 naive 資料都能比較
        start = bars.index[0].normalize() + pd.Timedelta(
            hours=session_open.hour, minutes=session_open.minute)
        or_end = start + pd.Timedelta(minutes=or_minutes)
        window = bars[bars.index < or_end]
        if len(window) == 0 or bars.index[-1] < or_end:
            return 0
        hi, lo = window["high"].max(), window["low"].min()
        last_close = bars["close"].iloc[-1]
        prev_close = bars["close"].iloc[-2] if len(bars) > 1 else last_close
        # 只在「剛突破」那根觸發,前一根仍在區間內,避免停損後立刻重複進場
        if last_close > hi and prev_close <= hi:
            return 1
        if last_close < lo and prev_close >= lo:
            return -1
        return 0

    return signal
