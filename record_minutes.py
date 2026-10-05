"""盤中記錄分鐘線:每 N 秒輪詢證交所即時報價,自行組成分鐘線,收盤或 Ctrl+C 時存檔。

用法: python record_minutes.py 2330 2317 --interval 6
"""
import argparse

from twdt import live
from twdt.data import realtime


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
        live.poll_loop(args.symbols, args.market, args.interval,
                       lambda q: builders[q["symbol"]].update(q["ts"], q["price"], q["cum_volume"]))
    except KeyboardInterrupt:
        print("中斷,存檔中…")
    for s, b in builders.items():
        path = realtime.save_day(s, b.to_frame(include_open_bar=True))
        print(f"{s}: {len(b.bars)} 根 bar -> {path}")


if __name__ == "__main__":
    main()
