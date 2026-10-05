"""Yahoo Finance 盤中歷史(免費、非官方端點):用來做粗略回測。

- 1m 只有最近約 7 天,5m/15m 約 60 天;每次抓取都合併進本地快取,持續執行就能累積超過 60 天。
- K 棒時間為「開始時間」(09:00 這根涵蓋 09:00~09:05),與本專案慣例一致。
- 沒有 13:25 之後的收盤集合競價;成交量偏低(約少 10%),只適合不依賴量的策略。
- 非官方端點,可能改版或限流;資料品質與即時性不如券商/付費資料,結果僅供參考。
- 上市用 .TW、上櫃用 .TWO。
"""
import os
import time
from datetime import time as dtime
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import requests

URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
CACHE_DIR = Path(os.environ.get("TWDT_CACHE", Path(__file__).resolve().parents[2] / "cache"))
TZ = "Asia/Taipei"
DEFAULT_RANGE = {"1m": "7d", "5m": "60d", "15m": "60d"}
COLS = ["open", "high", "low", "close", "volume"]
SESSION = (dtime(9, 0), dtime(13, 30))
DAY_DONE = dtime(13, 35)  # 過了這個時間,今天的資料才視為完整


class YahooError(RuntimeError):
    def __init__(self, msg: str, status_code: Optional[int] = None):
        super().__init__(msg)
        self.status_code = status_code

    @property
    def retryable(self) -> bool:
        return self.status_code is None or self.status_code == 429 or self.status_code >= 500


def ticker(symbol: str, market: str = "tse") -> str:
    return f"{symbol}.TW" if market == "tse" else f"{symbol}.TWO"


def _request(tk: str, interval: str, range_: str) -> dict:
    try:
        resp = requests.get(URL.format(ticker=tk),
                            params={"interval": interval, "range": range_, "includePrePost": "false"},
                            headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    except requests.RequestException as e:
        raise YahooError(f"{tk}: {type(e).__name__}") from None
    try:
        body = resp.json()
    except ValueError:
        body = {}
    chart = body.get("chart") or {}
    err = chart.get("error")
    if resp.status_code != 200 or err or not chart.get("result"):
        desc = (err or {}).get("description") if isinstance(err, dict) else None
        raise YahooError(f"{tk} {interval}/{range_}: HTTP {resp.status_code}, {desc or '沒有資料'}",
                         status_code=resp.status_code)
    return chart["result"][0]


def _request_retry(tk, interval, range_, retries=3, backoff=2.0,
                   sleep: Callable[[float], None] = time.sleep) -> dict:
    for attempt in range(retries + 1):
        try:
            return _request(tk, interval, range_)
        except YahooError as e:
            if not e.retryable or attempt == retries:
                raise
            sleep(backoff * (2 ** attempt))


def to_frame(result: dict) -> pd.DataFrame:
    """Yahoo 回應 -> OHLCV(台灣當地時間 naive)。丟掉空值 bar、四捨五入價格、只留 09:00~13:30。"""
    stamps = result.get("timestamp") or []
    if not stamps:
        return pd.DataFrame(columns=COLS)
    q = result["indicators"]["quote"][0]
    idx = pd.to_datetime(stamps, unit="s", utc=True).tz_convert(TZ).tz_localize(None)
    df = pd.DataFrame({c: q[c] for c in COLS}, index=idx).dropna()
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].round(2)
    df = df[(df.index.time >= SESSION[0]) & (df.index.time < SESSION[1])]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.astype(float)


def fetch_intraday(symbol: str, interval: str = "5m", market: str = "tse",
                   range_: Optional[str] = None, complete_days_only: bool = True,
                   now: Optional[pd.Timestamp] = None, sleep: Callable[[float], None] = time.sleep
                   ) -> pd.DataFrame:
    """抓盤中歷史並合併進本地快取,回傳快取中的完整歷史。

    complete_days_only:今天還沒收盤時,回傳結果排除今天(盤中那根還沒走完的 bar 不可拿來回測);
    快取本身仍保存,下次抓取同一時間戳會以新資料覆蓋。
    """
    if interval not in DEFAULT_RANGE:
        raise ValueError(f"interval 只支援 {sorted(DEFAULT_RANGE)}")
    result = _request_retry(ticker(symbol, market), interval, range_ or DEFAULT_RANGE[interval], sleep=sleep)
    new = to_frame(result)

    path = CACHE_DIR / "yahoo" / interval / f"{symbol}_{market}.parquet"
    if path.exists():
        both = pd.concat([pd.read_parquet(path), new])
        merged = both[~both.index.duplicated(keep="last")].sort_index()   # 同一時間戳以新資料為準
    else:
        merged = new
    if not merged.empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        merged.to_parquet(tmp)
        os.replace(tmp, path)

    if complete_days_only and not merged.empty:
        now = now if now is not None else pd.Timestamp.now(tz=TZ).tz_localize(None)
        if now.time() < DAY_DONE:
            merged = merged[merged.index.date != now.date()]
    return merged
