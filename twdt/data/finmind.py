"""FinMind 資料抓取,含本地 parquet 快取。

- 日線(TaiwanStockPrice)免費可用。
- 分鐘線(TaiwanStockKBar)需 FinMind 贊助會員 token。
token 從環境變數 FINMIND_TOKEN 讀取,不寫進程式碼。
"""
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

API_URL = "https://api.finmindtrade.com/api/v4/data"
CACHE_DIR = Path(os.environ.get("TWDT_CACHE", Path(__file__).resolve().parents[2] / "cache"))


class FinMindError(RuntimeError):
    def __init__(self, msg: str, status_code: Optional[int] = None):
        super().__init__(msg)
        self.status_code = status_code

    @property
    def retryable(self) -> bool:
        """網路錯誤、限流(429)、伺服器錯誤(5xx)才值得重試;權限/參數錯誤重試無用。"""
        return self.status_code is None or self.status_code == 429 or self.status_code >= 500


def _get(dataset: str, symbol: str, start: str, end: str, token: Optional[str]) -> pd.DataFrame:
    params = {"dataset": dataset, "data_id": symbol, "start_date": start, "end_date": end}
    if token:
        params["token"] = token
    try:
        resp = requests.get(API_URL, params=params, timeout=30)
    except requests.RequestException as e:
        # requests 的例外訊息會帶完整 URL(含 token),只留例外型別
        raise FinMindError(f"{dataset} {symbol} {start}: {type(e).__name__}") from None
    # 不用 raise_for_status():它會吃掉回應內容,且例外訊息會帶出含 token 的完整 URL
    try:
        body = resp.json()
    except ValueError:
        body = {}
    if resp.status_code != 200 or body.get("status") != 200:
        raise FinMindError(
            f"{dataset} {symbol} {start}: HTTP {resp.status_code}, msg={body.get('msg')!r}",
            status_code=resp.status_code)
    return pd.DataFrame(body["data"])


def _get_retry(dataset: str, symbol: str, start: str, end: str, token: Optional[str],
               retries: int = 3, backoff: float = 2.0) -> pd.DataFrame:
    for attempt in range(retries + 1):
        try:
            return _get(dataset, symbol, start, end, token)
        except FinMindError as e:
            if not e.retryable or attempt == retries:
                raise
            time.sleep(backoff * (2 ** attempt))


def _cached(kind: str, symbol: str, start: str, end: str, fetch) -> pd.DataFrame:
    path = CACHE_DIR / kind / f"{symbol}_{start}_{end}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = fetch()
    # 空結果(可能是 API 暫時沒資料)或區間含今天(盤中資料不完整)都不快取
    if df.empty or pd.Timestamp(end).date() >= pd.Timestamp.now().date():
        return df
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

    FinMind 的 KBar 為「單日單次請求」,這裡逐日抓取並逐日快取:
    中途失敗時已抓好的日子不會作廢,下次只補缺的日子。需贊助會員 token。
    """
    token = token or os.environ.get("FINMIND_TOKEN")
    if not token:
        raise FinMindError("分鐘線需要 FINMIND_TOKEN(贊助會員)")

    today = pd.Timestamp.now().date()
    frames = []
    for day in pd.bdate_range(start, end):
        d = day.strftime("%Y-%m-%d")
        path = CACHE_DIR / "minute" / f"{symbol}_{d}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
            continue
        raw = _get_retry("TaiwanStockKBar", symbol, d, d, token)
        if raw.empty:  # 假日或停牌;不快取,避免把 API 暫時沒資料永久記下來
            continue
        ts = pd.to_datetime(raw["date"] + " " + raw["minute"])
        df = pd.DataFrame(
            {"open": raw["open"], "high": raw["high"], "low": raw["low"],
             "close": raw["close"], "volume": raw["volume"]}).set_index(ts).astype(float)
        df = df.sort_index()
        if day.date() < today:  # 今天盤中資料不完整,不快取
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return pd.concat(frames).sort_index()


def split_days(minute_df: pd.DataFrame):
    """把多日分鐘線切成 {date: 單日 DataFrame}。"""
    return {d: g for d, g in minute_df.groupby(minute_df.index.date)}
