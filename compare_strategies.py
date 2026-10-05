"""多個事前固定參數的日內策略,以前段/後段分開比較。

每個策略只有一組參數(不調參);停損停利統一用 RiskConfig 預設。
另含「隨機方向」對照組,衡量成本本身的拖累。
"""
import argparse
import math

import pandas as pd

from twdt import research
from twdt.backtest.costs import CostModel
from twdt.data import yahoo
from twdt.risk.rules import RiskConfig
from twdt.signals import intraday as sg
from twdt.signals.rules import opening_range_breakout


def strategies(df):
    pc = sg.prev_close_map(df)
    return {
        "ORB 開盤區間突破(基準)": opening_range_breakout(15),
        "VWAP 回歸": sg.vwap_fade(),
        "開盤動能 @09:30": sg.open_move(direction=1),
        "開盤反轉 @09:30(鏡像)": sg.open_move(direction=-1),
        "跳空回補": sg.gap_fade(pc),
        "日內新高低突破": sg.new_extreme(),
        "對照:隨機方向 @09:30": sg.random_at(),
    }


def summarize(trades: pd.DataFrame, n_syms: int) -> dict:
    n = len(trades)
    if n == 0:
        return {"筆數": 0}
    r = trades["ret"]
    t_stat = r.mean() / (r.std(ddof=1) / math.sqrt(n)) if n > 2 and r.std(ddof=1) > 0 else float("nan")
    per_sym = trades.groupby("symbol")["ret"].sum()
    return {"筆數": n, "每筆淨%": r.mean() * 100, "每筆毛%": trades["gross_ret"].mean() * 100,
            "每筆成本%": trades["cost_ret"].mean() * 100, "勝率": (trades["net"] > 0).mean(),
            "t值": t_stat, "賺錢股票": f"{int((per_sym > 0).sum())}/{n_syms}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="+")
    ap.add_argument("--dev-days", type=int, default=35)
    ap.add_argument("--fee-discount", type=float, default=0.6)
    args = ap.parse_args()
    data = {s: yahoo.fetch_intraday(s, "5m") for s in args.symbols}
    dev, test, common = research.split_days(data, args.dev_days)
    print(f"開發段 {common[0]}~{common[args.dev_days - 1]},驗證段 {common[args.dev_days]}~{common[-1]};"
          f"{len(args.symbols)} 檔;停損停利預設 {RiskConfig().stop_loss_pct:.0%}/{RiskConfig().take_profit_pct:.0%}\n")
    cost, risk = CostModel(fee_discount=args.fee_discount), RiskConfig()
    rows = []
    names = list(strategies(next(iter(data.values()))))
    for name in names:
        for seg_name, seg in (("開發段", dev), ("驗證段", test)):
            parts = []
            for s in args.symbols:
                sig = strategies(data[s])[name]          # 每檔各自建立(跳空策略要用該檔前收盤)
                parts.append(research.run_symbol(seg[s], s, cost, risk, signal=sig))
            trades = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) else pd.DataFrame()
            rows.append({"策略": name, "區段": seg_name, **summarize(trades, len(args.symbols))})
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(out.round({"每筆淨%": 3, "每筆毛%": 3, "每筆成本%": 3, "勝率": 2, "t值": 2}).to_string(index=False))


if __name__ == "__main__":
    main()
