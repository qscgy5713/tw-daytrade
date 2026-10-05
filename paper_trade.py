"""盤中模擬單:輪詢即時報價,以 ORB 基準策略虛擬下單,同時記錄分鐘線與成交。

用法: python paper_trade.py 2330 2317 --interval 6
請在 09:00 前啟動;盤中才啟動的標的因開盤區間不完整,當天不會交易。
"""
import argparse

import pandas as pd

from twdt import live
from twdt.backtest.costs import CostModel
from twdt.data import realtime
from twdt.paper.engine import PaperTrader
from twdt.paper.journal import save_trades
from twdt.report.metrics import summarize
from twdt.risk.rules import RiskConfig
from twdt.signals.rules import opening_range_breakout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="+")
    ap.add_argument("--interval", type=float, default=6.0, help="輪詢秒數,請 >= 5")
    ap.add_argument("--market", choices=["tse", "otc"], default="tse")
    ap.add_argument("--or-minutes", type=int, default=15)
    ap.add_argument("--fee-discount", type=float, default=0.6)
    args = ap.parse_args()
    if args.interval < 5:
        raise SystemExit("--interval 不可小於 5 秒(報價本身延遲 5 秒,過密易被限流)")

    cost, risk = CostModel(fee_discount=args.fee_discount), RiskConfig()
    traders = {s: PaperTrader(s, opening_range_breakout(args.or_minutes), cost, risk,
                              on_event=print) for s in args.symbols}
    day = pd.Timestamp.now(tz=live.TZ).strftime("%Y-%m-%d")
    try:
        live.poll_loop(args.symbols, args.market, args.interval,
                       lambda q: traders[q["symbol"]].on_quote(q["ts"], q["price"], q["cum_volume"]))
    except KeyboardInterrupt:
        print("中斷,結算中…")
    finally:
        all_trades = []
        for s, t in traders.items():
            t.finish()  # 還有部位就以最後報價平倉
            realtime.save_day(s, t.builder.to_frame(include_open_bar=True))
            all_trades += t.trades
        path = save_trades(all_trades, day)
        print(f"成交紀錄: {path}")
        for k, v in summarize(all_trades).items():
            print(f"{k:>20}: {v:,.2f}" if isinstance(v, float) else f"{k:>20}: {v}")


if __name__ == "__main__":
    main()
