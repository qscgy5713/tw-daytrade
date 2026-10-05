"""低頻策略的樣本外比較:策略與參數事前固定,以 2019-01-01 切成開發段/驗證段,和買進持有比較。"""
import argparse

import pandas as pd

from twdt import daily

SPLIT = "2019-01-01"
STRATEGIES = {
    "買進持有(基準)": daily.sig_buy_hold,
    "站上200日線才持有": daily.sig_sma_filter,
    "50/200日均線": daily.sig_dual_ma,
    "12-1月時序動能": daily.sig_momentum,
    "RSI(2)短線回檔": daily.sig_rsi2,
}


def run(df: pd.DataFrame, dev_start: str, cost: daily.DailyCost) -> pd.DataFrame:
    end_dev = (pd.Timestamp(SPLIT) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    rows = []
    for name, fn in STRATEGIES.items():
        sig = fn(df)                                   # 訊號用全部歷史算(暖機),但只在各區段內計績
        for seg, (a, b) in (("開發段", (dev_start, end_dev)), ("驗證段", (SPLIT, None))):
            r = daily.backtest_daily(df, sig, cost, start=a, end=b)
            rows.append({"策略": name, "區段": seg, "年數": round(r["years"], 1), "年化報酬": r["cagr"],
                         "年化波動": r["vol"], "夏普": r["sharpe"], "最大回撤": r["max_dd"],
                         "持有時間比": r["in_market"], "進場次數": r["entries"]})
    return pd.DataFrame(rows)


def show(title: str, out: pd.DataFrame) -> None:
    pd.set_option("display.width", 200)
    fmt = out.copy()
    for c in ("年化報酬", "年化波動", "最大回撤", "持有時間比"):
        fmt[c] = (fmt[c] * 100).round(1).astype(str) + "%"
    fmt["夏普"] = fmt["夏普"].round(2)
    print(f"\n{title}\n{fmt.to_string(index=False)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tax", type=float, default=0.001, help="證交稅率:ETF/指數 0.001、個股 0.003")
    args = ap.parse_args()
    cost = daily.DailyCost(tax_rate=args.tax)
    etf = daily.fetch_daily("0050")
    twii = daily.fetch_daily("^TWII", index=True)
    print(f"0050: {len(etf)} 筆 {etf.index[0].date()}~{etf.index[-1].date()},清理丟掉 {etf.attrs.get('dropped')}")
    print(f"加權指數: {len(twii)} 筆 {twii.index[0].date()}~{twii.index[-1].date()},清理丟掉 {twii.attrs.get('dropped')}")
    show("【0050(含股息還原價)】開發段 2010~2018,驗證段 2019~", run(etf, "2010-01-01", cost))
    show("【加權指數(僅價格,不含股息;以 ETF 成本交易)】開發段 2001~2018,驗證段 2019~", run(twii, "2001-01-01", cost))


if __name__ == "__main__":
    main()
