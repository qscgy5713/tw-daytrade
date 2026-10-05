"""FinMind 資料抓取,含本地 parquet 快取。

- 日線(TaiwanStockPrice)免費可用。
- 分鐘線(TaiwanStockKBar)需 FinMind 贊助會員 token。
token 從環境變數 FINMIND_TOKEN 讀取,不寫進程式碼。
"""
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

API_URL = "https://api.finmindtrade.com/api/v4/data"
CACHE_DIR = Path(os.environ.get("TWDT_CACHE", Path(__file__).resolve().parents[2] / "cache"))


class FinMindError(RuntimeError):
    pass


def _get(dataset: str, symbol: str, start: str, end: str, token: Optional[str]) -> pd.DataFrame:
    params = {"dataset": dataset, "data_id": symbol, "start_date": start, "end_date": end}
    if token:
        params["token"] = token
    resp = requests.get(API_URL, params=params, timeout=30)
    # 不用 raise_for_status():它會吃掉回應內容,且例外訊息會帶出含 token 的完整 URL
    try:
        body = resp.json()
    except ValueError:
        body = {}
    if resp.status_code != 200 or body.get("status") != 200:
        raise FinMindError(
            f"{dataset} {symbol} {start}: HTTP {resp.status_code}, msg={body.get('msg')!r}")
    return pd.DataFrame(body["data"])


def _cached(kind: str, symbol: str, start: str, end: str, fetch) -> pd.DataFrame:
    path = CACHE_DIR / kind / f"{symbol}_{start}_{end}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def fetch_daily(symbol: str, start: str, end: str, token: Optional[str] = None) -> pd.DataFrame:
    """日線,欄位 open/high/low/close/volume,index 為日期。"""
    token = token or os.environ.get("FINMIND_TOKEN")

    def fetch():
        raw = _get("TaiwanStockPrice", symbol, start, end, token)
        if raw.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = raw.rename(columns={"max": "high", "min": "low", "Trading_Volume": "volume"})
        df.index = pd.to_datetime(df["date"])
        return df[["open", "high", "low", "close", "volume"]].astype(float)

    return _cached("daily", symbol, start, end, fetch)


def fetch_minute(symbol: str, start: str, end: str, token: Optional[str] = None) -> pd.DataFrame:
    """分鐘線,欄位 open/high/low/close/volume,index 為台灣當地時間(naive,不帶時區)。

    FinMind 的 KBar 為「單日單次請求」,這裡逐日抓取;需贊助會員 token。
    """
    token = token or os.environ.get("FINMIND_TOKEN")
    if not token:
        raise FinMindError("分鐘線需要 FINMIND_TOKEN(贊助會員)")

    def fetch():
        frames = []
        for day in pd.bdate_range(start, end):
            d = day.strftime("%Y-%m-%d")
            raw = _get("TaiwanStockKBar", symbol, d, d, token)
            if raw.empty:  # 假日或停牌
                continue
            ts = pd.to_datetime(raw["date"] + " " + raw["minute"])
            frames.append(pd.DataFrame(
                {"open": raw["open"], "high": raw["high"], "low": raw["low"],
                 "close": raw["close"], "volume": raw["volume"]}).set_index(ts))
        if not frames:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        return pd.concat(frames).astype(float).sort_index()

    return _cached("minute", symbol, start, end, fetch)


def split_days(minute_df: pd.DataFrame):
    """把多日分鐘線切成 {date: 單日 DataFrame}。"""
    return {d: g for d, g in minute_df.groupby(minute_df.index.date)}
