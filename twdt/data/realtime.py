"""證交所 MIS 即時報價(免費、不需 token),自行組成分鐘線並存檔。

- 報價約延遲 5 秒;輪詢間隔請 >= 5 秒,避免被限流。
- 上市用 tse_<代號>.tw,上櫃用 otc_<代號>.tw。
- volume 單位為「張」(累計量 v 的差值),與 FinMind 的股數不同,混用時需換算。
"""
import os
from datetime import time as dtime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

MIS_URL = "https://mis.twse.com.tw/stock/api/getStockInfo.jsp"
CACHE_DIR = Path(os.environ.get("TWDT_CACHE", Path(__file__).resolve().parents[2] / "cache"))


def _last_price(item: dict) -> float:
    """最新成交價。頂層 z 常是 '-'(該次更新沒帶成交價,但累計量 v 仍在增加),
    此時改用巢狀的 trade.z(最近一筆成交價)。兩者都沒有才視為尚無成交。"""
    try:
        return float(item["z"])
    except (KeyError, ValueError, TypeError):
        return float((item.get("trade") or {})["z"])


def parse_quote(item: dict) -> Optional[dict]:
    """MIS 單檔報價 -> {symbol, ts, price, cum_volume};尚無任何成交回傳 None。"""
    try:
        price = _last_price(item)
        cum = int(item["v"])
        ts = pd.Timestamp(f"{item['d'][:4]}-{item['d'][4:6]}-{item['d'][6:]} {item['t']}")
    except (KeyError, ValueError, TypeError):
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
        # 時間倒退或跨日的報價直接忽略(跨日會讓缺口補平填出上千根平盤 K)
        if self._cur is not None and (minute < self._cur["time"]
                                      or ts.date() != self._cur["time"].date()):
            return
        if self._last_cum is None:
            # 09:00 就開始記錄時,累計量全是開盤集合競價,歸入第一根 bar;
            # 盤中才啟動則累計量是之前已發生的量,不算進來
            delta = cum_volume if minute.time() == dtime(9, 0) else 0
            self._last_cum = cum_volume
        else:
            delta = max(0, cum_volume - self._last_cum)
        self._last_cum = max(self._last_cum, cum_volume)
        if self._cur is None:
            self._cur = {"time": minute, "open": price, "high": price, "low": price,
                         "close": price, "volume": delta}
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


def _day_path(symbol: str, day) -> Path:
    return CACHE_DIR / "realtime" / f"{symbol}_{day}.parquet"


def _merge_minutes(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """合併舊檔與本次 session 的分鐘線。

    同一分鐘跨越重啟時,新舊各只有半根 bar:open 取舊的、high/low 取極值、
    close 取新的、量相加。
    """
    both = pd.concat([old, new])
    return both.groupby(level=0).agg(open=("open", "first"), high=("high", "max"),
                                     low=("low", "min"), close=("close", "last"),
                                     volume=("volume", "sum")).sort_index()


class DayRecorder:
    """把某檔當天的分鐘線寫進 parquet,可在同一個 session 內反覆呼叫 flush()。

    第一次 flush 時讀一次磁碟上既有的檔案當「基準」(前一個 session 留下的),
    之後每次都以「基準 + 目前整份 builder 資料」重新寫入。基準不變,所以反覆 flush
    不會把量重複相加。用暫存檔加 os.replace 原子寫入,寫到一半被中斷也不會弄壞舊檔。

    限制:不要讓兩個行程同時記錄同一檔(各自的基準看不到對方的資料)。
    """

    def __init__(self, symbol: str):
        self.symbol = symbol
        self._baseline: Optional[pd.DataFrame] = None
        self._baseline_loaded = False

    def flush(self, df: pd.DataFrame) -> Optional[Path]:
        if df.empty:
            return None
        path = _day_path(self.symbol, df.index[0].date())
        if not self._baseline_loaded:
            self._baseline = pd.read_parquet(path) if path.exists() else None
            self._baseline_loaded = True
        out = df if self._baseline is None else _merge_minutes(self._baseline, df)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        out.to_parquet(tmp)
        os.replace(tmp, path)
        return path


def save_day(symbol: str, df: pd.DataFrame) -> Optional[Path]:
    """單次寫入(與既有檔合併)。同一個 session 內要反覆存檔請改用 DayRecorder。"""
    return DayRecorder(symbol).flush(df)


def load_minute(symbol: str) -> pd.DataFrame:
    """讀回已記錄的所有交易日分鐘線(與 finmind.fetch_minute 同欄位)。"""
    files = sorted((CACHE_DIR / "realtime").glob(f"{symbol}_*.parquet"))
    if not files:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return pd.concat(pd.read_parquet(f) for f in files).sort_index()
