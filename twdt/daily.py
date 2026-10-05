"""日線資料與低頻(隔日以上)策略回測。

資料:Yahoo 日線(免費、非官方)。注意 range=max 會被壓成月線,必須用 period1/period2。
Yahoo 的分割不一定有調整(例如 0050 在 2014-01-02 的 4 比 1 分割),所以 clean_daily 會偵測並調整。

回測語意(避免用到未來資料):
- 訊號用「當日收盤(還原價)」算出,隔天開盤才進出場。
- 報酬為還原開盤價到下一個還原開盤價(含股息再投入),進出場成本在開盤那天扣。
"""
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import requests

from twdt.data import yahoo

CACHE_DIR = Path(os.environ.get("TWDT_CACHE", Path(__file__).resolve().parents[1] / "cache"))
TRADING_DAYS = 245
# 常見分割/合併比例(分割後價格 / 分割前價格)
_SPLIT_RATIOS = [1 / k for k in range(2, 11)] + [float(k) for k in range(2, 11)]


# ---------------------------------------------------------------- 資料
def fetch_daily(symbol: str, market: str = "tse", start: str = "2000-01-01", index: bool = False,
                now: Optional[float] = None) -> pd.DataFrame:
    """抓日線(period1/period2,不用 range=max)並清理後存快取;回傳清理後完整歷史。"""
    tk = symbol if index else yahoo.ticker(symbol, market)
    p1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    p2 = int(now if now is not None else time.time())
    try:
        resp = requests.get(yahoo.URL.format(ticker=tk),
                            params={"interval": "1d", "period1": p1, "period2": p2, "events": "div,splits"},
                            headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    except requests.RequestException as e:
        raise yahoo.YahooError(f"{tk}: {type(e).__name__}") from None
    try:
        body = resp.json()
    except ValueError:
        body = {}
    chart = body.get("chart") or {}
    if resp.status_code != 200 or chart.get("error") or not chart.get("result"):
        raise yahoo.YahooError(f"{tk} 日線: HTTP {resp.status_code}", status_code=resp.status_code)
    res = chart["result"][0]
    idx = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(yahoo.TZ).tz_localize(None).normalize()
    q = res["indicators"]["quote"][0]
    adj = (res["indicators"].get("adjclose") or [{}])[0].get("adjclose") or q["close"]
    raw = pd.DataFrame({"open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"],
                        "adjclose": adj, "volume": q["volume"]}, index=idx)
    df = clean_daily(raw)
    path = CACHE_DIR / "daily" / f"{symbol.replace('^', '')}_{market}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp)
    os.replace(tmp, path)
    return df


def clean_daily(df: pd.DataFrame, split_threshold: float = 0.40, tol: float = 0.03,
                max_open_gap: float = 0.15) -> pd.DataFrame:
    """清理日線:
    1) 丟掉空值列、重複日期,以及任何價格 <= 0 的列(Yahoo 偶爾給開盤價 0);
    2) 丟掉「成交量為 0 且收盤價與前一日相同」的列(停止交易期間的重複價格);
    3) 單日漲跌超過 split_threshold 且比例接近 1/k 或 k(±tol)視為分割/合併,
       把「之前」所有價格(含還原價)乘上調整比例;比例對不上的視為資料錯誤,直接丟出例外,不默默放行;
    4) 開盤價與前一日收盤相差超過 max_open_gap(預設 15%,高於台股 7%/10% 的漲跌幅限制)的列視為資料錯誤丟掉。
    丟掉的筆數記在 df.attrs["dropped"],方便確認沒有大量資料被默默拿掉。
    """
    price_cols = ["open", "high", "low", "close", "adjclose"]
    n0 = len(df)
    d = df.dropna(subset=price_cols).copy()
    n_nan = n0 - len(d)
    d = d[~d.index.duplicated(keep="last")].sort_index()
    nonpos = (d[price_cols] <= 0).any(axis=1)
    n_nonpos = int(nonpos.sum())
    d = d[~nonpos]
    stale = (d["volume"].fillna(0) == 0) & (d["close"] == d["close"].shift(1))
    n_stale = int(stale.sum())
    d = d[~stale]
    # 逐次偵測:每調整一次就重算,避免多次分割互相影響
    for _ in range(10):
        ratio = d["close"] / d["close"].shift(1)
        jump = ratio[(ratio - 1).abs() > split_threshold]
        if jump.empty:
            break
        day, r = jump.index[0], float(jump.iloc[0])
        match = min(_SPLIT_RATIOS, key=lambda s: abs(r / s - 1))
        if abs(r / match - 1) > tol:
            raise ValueError(f"{day.date()} 單日變動 {r - 1:+.1%} 對不上任何常見分割比例,疑似資料錯誤")
        d.loc[d.index < day, price_cols] = d.loc[d.index < day, price_cols] * match
    bad_open = (d["open"] / d["close"].shift(1) - 1).abs() > max_open_gap
    n_open = int(bad_open.sum())
    d = d[~bad_open]
    d = d.astype(float)
    d.attrs["dropped"] = {"空值": n_nan, "價格<=0": n_nonpos, "停止交易重複價": n_stale, "開盤價異常": n_open}
    return d


def adj_open(df: pd.DataFrame) -> pd.Series:
    """還原開盤價:open 乘上 adjclose/close(把股息也算進報酬)。"""
    return df["open"] * (df["adjclose"] / df["close"])


# ---------------------------------------------------------------- 指標與訊號(回傳每日 0/1 持有訊號,以當日收盤算出)
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def rsi(s: pd.Series, n: int = 2) -> pd.Series:
    """Wilder RSI。"""
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(dn != 0, 100.0).where(up.notna() & dn.notna())


def sig_buy_hold(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=df.index)


def sig_sma_filter(df: pd.DataFrame, n: int = 200) -> pd.Series:
    c = df["adjclose"]
    return (c > sma(c, n)).astype(float).where(sma(c, n).notna(), 0.0)


def sig_dual_ma(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.Series:
    c = df["adjclose"]
    f, s_ = sma(c, fast), sma(c, slow)
    return (f > s_).astype(float).where(s_.notna(), 0.0)


def sig_momentum(df: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.Series:
    """時序動能:過去 lookback 日(跳過最近 skip 日)的報酬為正就持有。"""
    c = df["adjclose"]
    mom = c.shift(skip) / c.shift(lookback) - 1
    return (mom > 0).astype(float).where(mom.notna(), 0.0)


def sig_rsi2(df: pd.DataFrame, entry: float = 10, exit_sma: int = 5, trend: int = 200) -> pd.Series:
    """短線回檔買進:長期趨勢向上(收盤>SMA trend)且 RSI(2)<entry 時買進,收盤站上 SMA(exit_sma) 賣出。"""
    c = df["adjclose"]
    r, ma_exit, ma_trend = rsi(c, 2), sma(c, exit_sma), sma(c, trend)
    pos, out = 0.0, []
    for i in range(len(c)):
        if pos == 0.0:
            if pd.notna(r.iloc[i]) and pd.notna(ma_trend.iloc[i]) and r.iloc[i] < entry and c.iloc[i] > ma_trend.iloc[i]:
                pos = 1.0
        elif pd.notna(ma_exit.iloc[i]) and c.iloc[i] > ma_exit.iloc[i]:
            pos = 0.0
        out.append(pos)
    return pd.Series(out, index=c.index)


# ---------------------------------------------------------------- 回測
@dataclass(frozen=True)
class DailyCost:
    fee_rate: float = 0.001425 * 0.6      # 單邊手續費(6 折)
    tax_rate: float = 0.001               # ETF 證交稅 0.1%(個股為 0.3%)
    slippage: float = 0.0005              # 單邊 0.05%

    def buy(self) -> float:
        return self.fee_rate + self.slippage

    def sell(self) -> float:
        return self.fee_rate + self.tax_rate + self.slippage


def backtest_daily(df: pd.DataFrame, signal: pd.Series, cost: DailyCost,
                   start: Optional[str] = None, end: Optional[str] = None) -> dict:
    """訊號(當日收盤)-> 隔天開盤成交 -> 持有到下一個開盤。回傳績效指標與每日資料。

    每個區段都從空手開始:區段第一天若訊號為持有,視為當天開盤買進並扣成本,所有策略一致。
    """
    o = adj_open(df)
    r_oo = o.shift(-1) / o - 1                       # 第 t 天開盤到第 t+1 天開盤
    pos = signal.shift(1).fillna(0.0)                # 第 t 天持有的部位 = 第 t-1 天收盤的訊號
    frame = pd.DataFrame({"pos": pos, "r": r_oo}).dropna()
    if start:
        frame = frame[frame.index >= pd.Timestamp(start)]
    if end:
        frame = frame[frame.index <= pd.Timestamp(end)]
    if frame.empty:
        raise ValueError("區段內沒有資料")
    prev = frame["pos"].shift(1).fillna(0.0)         # 區段開始前視為空手
    entries = (frame["pos"] == 1) & (prev == 0)
    exits = (frame["pos"] == 0) & (prev == 1)
    trade_cost = entries * cost.buy() + exits * cost.sell()
    daily = (1 - trade_cost) * (1 + frame["pos"] * frame["r"]) - 1
    equity = (1 + daily).cumprod()
    years = max((frame.index[-1] - frame.index[0]).days / 365.25, 1e-9)
    vol = daily.std(ddof=1) * np.sqrt(TRADING_DAYS)
    return {
        "cagr": equity.iloc[-1] ** (1 / years) - 1,
        "vol": vol,
        "sharpe": daily.mean() * TRADING_DAYS / vol if vol > 0 else float("nan"),
        "max_dd": float((equity / equity.cummax() - 1).min()),
        "in_market": float(frame["pos"].mean()),
        "entries": int(entries.sum()),
        "final": float(equity.iloc[-1]),
        "years": years,
        "daily": daily,
        "equity": equity,
    }
