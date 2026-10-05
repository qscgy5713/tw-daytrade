"""以 ORB 基準策略跑回測。

用法:
  python run_backtest.py --source synthetic --start 2026-01-01 --end 2026-03-31
  FINMIND_TOKEN=... python run_backtest.py --source finmind --symbols 2330 2317 \
      --start 2026-09-01 --end 2026-09-30
"""
import argparse

from twdt.backtest.costs import CostModel
from twdt.backtest.engine import run_day
from twdt.data import finmind
from twdt.data.synthetic import synthetic_minute
from twdt.report.metrics import summarize
from twdt.risk.rules import RiskConfig
from twdt.signals.rules import opening_range_breakout


def backtest(minute_by_symbol: dict, cost: CostModel, risk: RiskConfig, or_minutes: int):
    """每檔每天獨立計算(單日虧損上限以單檔為單位),回傳全部 Trade。"""
    signal = opening_range_breakout(or_minutes)
    trades = []
    for symbol, minute_df in minute_by_symbol.items():
        for _, day_bars in finmind.split_days(minute_df).items():
            trades.extend(run_day(symbol, day_bars, signal, cost, risk))
    return trades


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "finmind"], default="synthetic")
    ap.add_argument("--symbols", nargs="+", default=["2330"])
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--or-minutes", type=int, default=15)
    ap.add_argument("--fee-discount", type=float, default=0.6)
    args = ap.parse_args()

    if args.source == "synthetic":
        data = {s: synthetic_minute(args.start, args.end, seed=i)
                for i, s in enumerate(args.symbols)}
    else:
        data = {s: finmind.fetch_minute(s, args.start, args.end) for s in args.symbols}

    trades = backtest(data, CostModel(fee_discount=args.fee_discount), RiskConfig(),
                      args.or_minutes)
    for k, v in summarize(trades).items():
        print(f"{k:>20}: {v:,.4f}" if isinstance(v, float) else f"{k:>20}: {v}")


if __name__ == "__main__":
    main()
