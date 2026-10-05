"""月配息 ETF 配置研究:每日再平衡的含息報酬/回檔,以及定期定額的配息現金流模擬。

假設與限制(請一併閱讀結果):
- 報酬用 Yahoo 還原價(含息再投入)算;配息現金流用 Yahoo 的除息日與每股金額,配息於除息日入帳
  (實際約晚 3 週才入帳),不扣稅、不扣手續費;定期定額可買零股(小數股)。
- 期間很短(幾年)且是特定行情,「歷史最佳」不代表未來。
"""
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import requests

URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
CACHE_DIR = Path(os.environ.get("TWDT_CACHE", Path(__file__).resolve().parents[1] / "cache")) / "etf"
TZ = "Asia/Taipei"


# ---------------------------------------------------------------- 資料
def fetch_etf(code: str, suffix: str = "TW", start: str = "2022-01-01", now: float = None
              ) -> Tuple[pd.DataFrame, pd.Series]:
    """回傳 (價格 DataFrame[close, adjclose], 除息日每股配息 Series)。"""
    p1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    p2 = int(now if now is not None else time.time())
    resp = requests.get(URL.format(ticker=f"{code}.{suffix}"),
                        params={"interval": "1d", "period1": p1, "period2": p2, "events": "div"},
                        headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    body = resp.json() if resp.status_code == 200 else {}
    result = (body.get("chart") or {}).get("result")
    if not result:
        raise RuntimeError(f"{code}.{suffix}: HTTP {resp.status_code},沒有資料")
    res = result[0]
    idx = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(TZ).tz_localize(None).normalize()
    q = res["indicators"]["quote"][0]
    adj = res["indicators"]["adjclose"][0]["adjclose"]
    px = pd.DataFrame({"close": q["close"], "adjclose": adj}, index=idx).dropna()
    px = px[~px.index.duplicated(keep="last")].sort_index()
    ev = (res.get("events") or {}).get("dividends", {})
    # 先轉成台灣時區再取日期:時間戳若剛好是台灣時間 00:00,直接用 UTC 取整會少一天
    divs = pd.Series({pd.Timestamp(v["date"], unit="s", tz="UTC").tz_convert(TZ).tz_localize(None).normalize():
                      float(v["amount"]) for v in ev.values()}, dtype=float).sort_index()
    return px, divs


def align(frames: Dict[str, Tuple[pd.DataFrame, pd.Series]], start: str = None):
    """把多檔對齊到共同交易日;非交易日的除息日順延到下一個共同交易日(同日多筆加總)。"""
    idx = None
    for px, _ in frames.values():
        idx = px.index if idx is None else idx.intersection(px.index)
    if start:
        idx = idx[idx >= pd.Timestamp(start)]
    idx = idx.sort_values()
    close = pd.DataFrame({k: v[0]["close"].reindex(idx) for k, v in frames.items()})
    adj = pd.DataFrame({k: v[0]["adjclose"].reindex(idx) for k, v in frames.items()})
    div = pd.DataFrame(0.0, index=idx, columns=list(frames))
    for k, (_, d) in frames.items():
        for day, amt in d.items():
            pos = idx.searchsorted(day)              # 除息日不在共同交易日時,順延到下一個
            if day >= idx[0] and pos < len(idx):
                div.iloc[pos, div.columns.get_loc(k)] += amt
    return close, adj, div


# ---------------------------------------------------------------- 權重
def weights_grid(n: int, step_pct: int = 10) -> np.ndarray:
    """所有「n 檔、每檔為 step_pct 的倍數、合計 100%」的權重組合。"""
    total = 100 // step_pct
    out = []

    def rec(prefix, remaining, slots):
        if slots == 1:
            out.append(prefix + [remaining]); return
        for k in range(remaining + 1):
            rec(prefix + [k], remaining - k, slots - 1)

    rec([], total, n)
    return np.array(out, dtype=float) / total


# ---------------------------------------------------------------- 每日再平衡的含息報酬
def tr_metrics(adj: pd.DataFrame, W: np.ndarray) -> Dict[str, np.ndarray]:
    """每日再平衡到固定權重的含息報酬:年化報酬、年化波動、最大回檔。"""
    R = adj.pct_change().dropna().values                      # (T, n)
    port = R @ W.T                                            # (T, K)
    equity = np.cumprod(1 + port, axis=0)
    years = (adj.index[-1] - adj.index[0]).days / 365.25
    ann = equity[-1] ** (1 / years) - 1
    vol = port.std(axis=0, ddof=1) * np.sqrt(245)
    peak = np.maximum.accumulate(np.vstack([np.ones(W.shape[0]), equity]), axis=0)[1:]
    dd = (equity / peak - 1).min(axis=0)
    return {"ann": ann, "vol": vol, "max_dd": dd}


# ---------------------------------------------------------------- 定期定額配息現金流
def dca_simulate(close: pd.DataFrame, div: pd.DataFrame, W: np.ndarray, contrib: float = 10000.0,
                 reinvest: bool = False) -> dict:
    """每月第一個共同交易日以收盤價買進 contrib 元(依權重分配);配息於除息日、以「當日買進前」的持股入帳。

    reinvest=True:配息現金累積起來,在下一次買進日(當月買進日若除息在買進日當天,則同一天)
    以收盤價買回「配息來源的那一檔」;不計入「投入」,所以累計報酬仍以自己掏的錢為分母。
    """
    idx = close.index
    months = pd.Series(idx.to_period("M"), index=idx)
    first_days = set(months.groupby(months).head(1).index)
    K, n = W.shape
    shares = np.zeros((K, n))
    cash = np.zeros((K, n))                                    # 尚未再投入的配息(僅 reinvest 時使用)
    month_keys = sorted(set(months))
    pos = {m: i for i, m in enumerate(month_keys)}
    monthly_div = np.zeros((len(month_keys), K))
    invested_m = np.zeros(len(month_keys))
    invested = 0.0
    px_arr, div_arr = close.values, div.values
    for t, day in enumerate(idx):
        m = pos[months.iloc[t]]
        d = div_arr[t]
        if d.any():
            got = shares * d                                   # 先入帳(用買進前的持股),(K, n)
            monthly_div[m] += got.sum(axis=1)
            if reinvest:
                cash += got
        if day in first_days:
            shares += contrib * W / px_arr[t]                  # 再買進
            invested += contrib
            if reinvest:
                shares += cash / px_arr[t]                     # 配息再投入同一檔
                cash[:] = 0.0
        invested_m[m] = invested
    last_px = px_arr[-1]
    market_value = shares @ last_px
    kept = cash.sum(axis=1) if reinvest else monthly_div.sum(axis=0)   # 手上還留著的配息現金
    # 最後一個月若尚未走完(最後一天不是該月最後一個營業日),它的配息還沒發生完,不可納入配息指標
    complete = np.ones(len(month_keys), dtype=bool)
    if idx[-1] != pd.offsets.BMonthEnd().rollforward(idx[-1]):
        complete[-1] = False
    return {"monthly_div": monthly_div, "invested_m": invested_m, "market_value": market_value,
            "total_div": monthly_div.sum(axis=0), "dividends_kept": kept, "invested": invested,
            "months": month_keys, "complete": complete, "reinvest": reinvest}


def income_metrics(sim: dict, last_n: int = 12) -> Dict[str, np.ndarray]:
    """配息指標(以投入成本為分母,排除持股逐月累積造成的假性成長;只用「完整的月份」)。

    - yoc_annual:最近 last_n 個完整月份的配息合計 ÷ 該段期間平均投入成本(年化)
    - yoc_cv:滾動 3 個月配息率(3 個月配息 ÷ 平均投入)的變異係數,越小越穩。
      用 3 個月而不是單月,是因為除息日會在月底月初之間飄移,單月會出現「斷一個月、下個月兩次」的假象
    - yoc_min_month:滾動 3 個月配息率的最低值(以 3 個月的配息占投入比例衡量,這是『最差一季』)
    - total_return:累計報酬 = (期末市值 + 手上留著的配息現金 - 總投入) ÷ 總投入;不再投入時留著的是全部配息,
      再投入時配息已變成股票,只剩尚未買回的現金(含尚未走完的最後一個月的價格變動)
    """
    keep = np.flatnonzero(sim["complete"])[-last_n:]
    md, inv = sim["monthly_div"][keep], sim["invested_m"][keep]
    if len(keep) < 6:
        raise ValueError("完整月份不足 6 個月,無法評估配息穩定度")
    c = np.cumsum(np.vstack([np.zeros((1, md.shape[1])), md]), axis=0)
    div3 = c[3:] - c[:-3]                                              # 滾動 3 個月配息合計 (len-2, K)
    inv3 = np.convolve(inv, np.ones(3) / 3, mode="valid")              # 對應的 3 個月平均投入
    yoc3 = div3 / np.maximum(inv3[:, None], 1e-9)
    mean = yoc3.mean(axis=0)
    cv = np.where(mean > 0, yoc3.std(axis=0, ddof=1) / np.where(mean > 0, mean, 1), np.nan)
    return {
        "yoc_annual": md.sum(axis=0) / inv.mean() * (12 / len(inv)),
        "yoc_cv": cv,
        "yoc_min_month": yoc3.min(axis=0),
        "total_return": (sim["market_value"] + sim.get("dividends_kept", sim["total_div"]) - sim["invested"])
        / sim["invested"],
    }
