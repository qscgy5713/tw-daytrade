"""低頻策略在多檔標的上的樣本外比較(策略/參數/切分日與 compare_daily.py 完全相同,事前固定)。

選股規則(不看績效):個股=證交所最近交易日成交值前 N 名的一般股票;ETF=手動列出。
歷史不足(2009-06-01 之後才上市)的標的排除,因為開發段 2010 年起需要完整與暖機資料。
注意:以「現在成交熱門」選股有存活者偏誤,會偏向近年的贏家,個股買進持有因而被高估。
"""
import argparse
import re
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import requests

from compare_daily import SPLIT, STRATEGIES
from twdt import daily

DEV_START = "2010-01-01"
FETCH_START = "2008-01-01"   # 開發段 2010 起,留約兩年暖機
HISTORY_NEEDED = pd.Timestamp("2009-06-01")
BASELINE = "買進持有(基準)"


def top_turnover_stocks(n: int) -> List[str]:
    rows = requests.get("https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL",
                        headers={"User-Agent": "Mozilla/5.0"}, timeout=30).json()
    stocks = [r for r in rows if re.fullmatch(r"[1-9]\d{3}", r["Code"]) and r["TradeValue"].replace(",", "").isdigit()]
    stocks.sort(key=lambda r: int(r["TradeValue"].replace(",", "")), reverse=True)
    return [r["Code"] for r in stocks[:n]]


def evaluate_asset(df: pd.DataFrame, tax: float) -> pd.DataFrame:
    cost = daily.DailyCost(tax_rate=tax)
    end_dev = (pd.Timestamp(SPLIT) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    rows = []
    for name, fn in STRATEGIES.items():
        sig = fn(df)
        for seg, (a, b) in (("開發段", (DEV_START, end_dev)), ("驗證段", (SPLIT, None))):
            r = daily.backtest_daily(df, sig, cost, start=a, end=b)
            rows.append({"strategy": name, "seg": seg, "cagr": r["cagr"], "sharpe": r["sharpe"],
                         "max_dd": r["max_dd"], "entries_per_year": r["entries"] / r["years"],
                         "in_market": r["in_market"]})
    return pd.DataFrame(rows)


def aggregate(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """results: {標的: evaluate_asset 的結果}。每個策略每區段回傳「相對自己買進持有」的整體統計。"""
    rows = []
    for name in STRATEGIES:
        if name == BASELINE:
            continue
        for seg in ("開發段", "驗證段"):
            d = []
            for sym, res in results.items():
                s = res[(res.strategy == name) & (res.seg == seg)].iloc[0]
                b = res[(res.strategy == BASELINE) & (res.seg == seg)].iloc[0]
                d.append({"d_cagr": s.cagr - b.cagr, "d_sharpe": s.sharpe - b.sharpe,
                          "d_dd": s.max_dd - b.max_dd,          # 正值 = 最大回撤比買進持有淺
                          "entries": s.entries_per_year, "in_market": s.in_market,
                          "bh_cagr": b.cagr})
            x = pd.DataFrame(d)
            rows.append({"策略": name, "區段": seg, "標的數": len(x),
                         "買進持有平均年化": x.bh_cagr.mean(),
                         "年化差(平均)": x.d_cagr.mean(), "年化差(中位)": x.d_cagr.median(),
                         "夏普差(平均)": x.d_sharpe.mean(),
                         "夏普勝過買進持有": (x.d_sharpe > 0).mean(), "年化勝過買進持有": (x.d_cagr > 0).mean(),
                         "回撤改善(平均)": x.d_dd.mean(), "每年進場": x.entries.mean(), "持有時間比": x.in_market.mean()})
    return pd.DataFrame(rows)


def show(title: str, agg: pd.DataFrame) -> None:
    pd.set_option("display.width", 220)
    f = agg.copy()
    for c in ("買進持有平均年化", "年化差(平均)", "年化差(中位)", "回撤改善(平均)", "持有時間比", "夏普勝過買進持有", "年化勝過買進持有"):
        f[c] = (f[c] * 100).round(1)
    f["夏普差(平均)"] = f["夏普差(平均)"].round(2)
    f["每年進場"] = f["每年進場"].round(1)
    print(f"\n{title}(年化差/回撤改善單位:百分點;勝過比例單位:%)\n{f.to_string(index=False)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--etfs", nargs="*", default=["0050", "0052", "0055", "0056", "0057", "006203", "006204", "006201"])
    args = ap.parse_args()

    universe = [(s, 0.001, "ETF") for s in args.etfs] + [(s, 0.003, "個股") for s in top_turnover_stocks(args.top)
                                                         if s not in args.etfs]
    results, groups, skipped = {}, {}, []
    for sym, tax, kind in universe:
        try:
            df = daily.fetch_daily(sym, start=FETCH_START)   # 只抓 2008 年以後,避開 2000~2002 年的舊資料雜訊
            if df.index[0] > HISTORY_NEEDED:
                skipped.append((sym, f"歷史太短(從 {df.index[0].date()} 起)")); continue
            results[sym] = evaluate_asset(df, tax)
            groups[sym] = kind
        except Exception as e:                       # 單檔失敗不影響其他檔,但要說清楚為什麼
            skipped.append((sym, f"{type(e).__name__}: {e}"))
        time.sleep(0.5)
    print(f"納入 {len(results)} 檔(ETF {sum(k == 'ETF' for k in groups.values())}、個股 {sum(k == '個股' for k in groups.values())});"
          f"排除 {len(skipped)} 檔:")
    for s, why in skipped:
        print(f"  - {s}: {why}")
    for kind in ("ETF", "個股"):
        sub = {s: r for s, r in results.items() if groups[s] == kind}
        if sub:
            show(f"【{kind} {len(sub)} 檔】", aggregate(sub))
    show(f"【全部 {len(results)} 檔】", aggregate(results))
    print("\n標的:", ", ".join(f"{s}({groups[s]})" for s in results))


if __name__ == "__main__":
    main()
