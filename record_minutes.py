"""盤中記錄分鐘線:每 N 秒輪詢證交所即時報價,自行組成分鐘線,收盤或 Ctrl+C 時存檔。

用法: python record_minutes.py 2330 2317 --interval 6
"""
import argparse
import time
from datetime import time as dtime

import pandas as pd

from twdt.data import realtime

OPEN, CLOSE = dtime(9, 0), dtime(13, 30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="+")
    ap.add_argument("--interval", type=float, default=6.0, help="輪詢秒數,請 >= 5")
    ap.add_argument("--market", choices=["tse", "otc"], default="tse")
    args = ap.parse_args()
    if args.interval < 5:
        raise SystemExit("--interval 不可小於 5 秒(報價本身延遲 5 秒,過密易被限流)")

    builders = {s: realtime.MinuteBarBuilder() for s in args.symbols}
    try:
        while pd.Timestamp.now().time() < CLOSE:
            if pd.Timestamp.now().time() >= OPEN:
                try:
                    for q in realtime.fetch_quotes(args.symbols, args.market):
                        builders[q["symbol"]].update(q["ts"], q["price"], q["cum_volume"])
                except Exception as e:  # 單次失敗不中斷整天記錄
                    print(f"[warn] {type(e).__name__}: {e}")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("中斷,存檔中…")
    for s, b in builders.items():
        path = realtime.save_day(s, b.to_frame(include_open_bar=True))
        print(f"{s}: {len(b.bars)} 根 bar -> {path}")


if __name__ == "__main__":
    main()
