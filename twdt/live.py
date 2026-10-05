"""盤中輪詢迴圈:記錄器與模擬單共用。"""
import time as _time
from datetime import time as dtime
from typing import Callable, List

import pandas as pd

from twdt.data import realtime

TZ = "Asia/Taipei"  # 不依賴本機時區
OPEN = dtime(9, 0)
# 13:30 收盤集合競價撮合,加上報價延遲約 5 秒,多等到 13:32 才收工
CLOSE = dtime(13, 32)


def poll_loop(symbols: List[str], market: str, interval: float,
              handler: Callable[[dict], None], *,
              now_fn: Callable[[], pd.Timestamp] = lambda: pd.Timestamp.now(tz=TZ),
              fetch_fn: Callable = realtime.fetch_quotes,
              sleep_fn: Callable[[float], None] = _time.sleep,
              log: Callable[[str], None] = print,
              stale_warn_after: int = 10) -> None:
    """輪詢到收盤。只把「今天且在清單內」的報價交給 handler。

    抓報價失敗只記警告、不中斷;handler 的例外則直接往外拋(交易邏輯出錯要大聲失敗,
    不能被當成網路問題吞掉)。
    """
    wanted = set(symbols)
    today = now_fn().date()
    stale_polls = 0
    while now_fn().time() < CLOSE:
        if now_fn().time() >= OPEN:
            try:
                quotes = fetch_fn(symbols, market)
            except Exception as e:
                log(f"[warn] {type(e).__name__}: {e}")
            else:
                # 假日或盤前 MIS 可能回前一交易日的舊報價,不可混進今天
                fresh = [q for q in quotes if q["ts"].date() == today and q["symbol"] in wanted]
                stale_polls = 0 if fresh else stale_polls + 1
                if stale_polls == stale_warn_after:
                    log(f"[warn] 連續 {stale_warn_after} 次沒有今天的報價,"
                        "今天可能休市或代號/市場(tse/otc)錯誤")
                for q in fresh:
                    handler(q)
        sleep_fn(interval)
