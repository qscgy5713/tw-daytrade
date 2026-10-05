"""證交所 MIS 即時報價(免費、不需 token),自行組成分鐘線並存檔。

- 報價約延遲 5 秒;輪詢間隔請 >= 5 秒,避免被限流。
- 上市用 tse_<代號>.tw,上櫃用 otc_<代號>.tw。
- volume 單位為「張」(累計量 v 的差值),與 FinMind 的股數不同,混用時需換算。
"""
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

MIS_URL = "https://mis.twse.com.tw/stock/api/getStockInfo.jsp"
CACHE_DIR = Path(os.environ.get("TWDT_CACHE", Path(__file__).resolve().parents[2] / "cache"))


def parse_quote(item: dict) -> Optional[dict]:
    """MIS 單檔報價 -> {symbol, ts, price, cum_volume};尚無成交(z 為 '-')回傳 None。"""
    try:
        price = float(item["z"])
        cum = int(item["v"])
        ts = pd.Timestamp(f"{item['d'][:4]}-{item['d'][4:6]}-{item['d'][6:]} {item['t']}")
    except (KeyError, ValueError):
        return None
    return {"symbol": item["c"], "ts": ts, "price": price, "cum_volume": cum}


def fetch_quotes(symbols: List[str], market: str = "tse") -> List[dict]:
    ex_ch = "|".join(f"{market}_{s}.tw" for s in symbols)
    resp = requests.get(MIS_URL, params={"ex_ch": ex_ch, "json": 1, "delay": 0},
                        headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(f"MIS HTTP {resp.status_code}")
    quotes = (parse_quote(i) for i in resp.json().get("msgArray", []))
    return [q for q in quotes if q is not None]


class MinuteBarBuilder:
    """把逐筆最新價組成每分鐘 OHLCV。沒有成交的分鐘以前一根收盤補平盤 K(量 0)。"""

    def __init__(self):
        self.bars: List[dict] = []  # 已完成的 bar
        self._cur: Optional[dict] = None
        self._last_cum: Optional[int] = None

    def update(self, ts: pd.Timestamp, price: float, cum_volume: int) -> None:
        minute = ts.floor("min")
        # 同一分鐘內累計量只會增加;時間倒退或重複的報價直接忽略
        if self._cur is not None and minute < self._cur["time"]:
            return
        delta = 0 if self._last_cum is None else max(0, cum_volume - self._last_cum)
        self._last_cum = cum_volume if self._last_cum is None else max(self._last_cum, cum_volume)
        if self._cur is None:
            # 第一筆的累計量是盤中已累積量,不算進這根 bar
            self._cur = {"time": minute, "open": price, "high": price, "low": price,
                         "close": price, "volume": 0}
            return
        if minute > self._cur["time"]:
            self.bars.append(self._cur)
            prev_close = self._cur["close"]
            missing = int((minute - self._cur["time"]) / pd.Timedelta(minutes=1)) - 1
            for k in range(1, missing + 1):
                self.bars.append({"time": self._cur["time"] + pd.Timedelta(minutes=k),
                                  "open": prev_close, "high": prev_close,
                                  "low": prev_close, "close": prev_close, "volume": 0})
            self._cur = {"time": minute, "open": price, "high": price, "low": price,
                         "close": price, "volume": 0}
        c = self._cur
        c["high"], c["low"], c["close"] = max(c["high"], price), min(c["low"], price), price
        c["volume"] += delta

    def to_frame(self, include_open_bar: bool = False) -> pd.DataFrame:
        rows = self.bars + ([self._cur] if include_open_bar and self._cur else [])
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows).set_index("time")
        df.index.name = None
        return df.astype(float)


def save_day(symbol: str, df: pd.DataFrame) -> Optional[Path]:
    if df.empty:
        return None
    path = CACHE_DIR / "realtime" / f"{symbol}_{df.index[0].date()}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return path


def load_minute(symbol: str) -> pd.DataFrame:
    """讀回已記錄的所有交易日分鐘線(與 finmind.fetch_minute 同欄位)。"""
    files = sorted((CACHE_DIR / "realtime").glob(f"{symbol}_*.parquet"))
    if not files:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return pd.concat(pd.read_parquet(f) for f in files).sort_index()
