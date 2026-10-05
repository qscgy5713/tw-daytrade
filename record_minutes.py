"""盤中記錄分鐘線:每 N 秒輪詢證交所即時報價,自行組成分鐘線。

每 --flush-every 秒自動存檔一次(避免被強制結束時整天資料全丟),收盤或 Ctrl+C 時再存一次。
事件同時寫進 cache/logs/record_<日期>.log。

用法: python record_minutes.py 2330 2317 --interval 6
"""
import argparse

from twdt import live
from twdt.data import realtime
from twdt.logutil import file_logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="+")
    ap.add_argument("--interval", type=float, default=6.0, help="輪詢秒數,請 >= 5")
    ap.add_argument("--market", choices=["tse", "otc"], default="tse")
    ap.add_argument("--flush-every", type=float, default=300.0, help="定期存檔間隔(秒)")
    args = ap.parse_args()
    if args.interval < 5:
        raise SystemExit("--interval 不可小於 5 秒(報價本身延遲 5 秒,過密易被限流)")

    log = file_logger("record")
    log(f"開始記錄 {args.symbols} market={args.market} interval={args.interval}s log={log.path}")
    builders = {s: realtime.MinuteBarBuilder() for s in args.symbols}
    recorders = {s: realtime.DayRecorder(s) for s in args.symbols}

    def flush():
        for s, b in builders.items():
            recorders[s].flush(b.to_frame(include_open_bar=True))

    try:
        live.poll_loop(args.symbols, args.market, args.interval,
                       lambda q: builders[q["symbol"]].update(q["ts"], q["price"], q["cum_volume"]),
                       log=log, flush_fn=flush, flush_every=args.flush_every)
    except KeyboardInterrupt:
        log("中斷,存檔中…")
    except Exception as e:
        log(f"[error] {type(e).__name__}: {e}")
        for s, b in builders.items():
            recorders[s].flush(b.to_frame(include_open_bar=True))
        raise
    for s, b in builders.items():
        path = recorders[s].flush(b.to_frame(include_open_bar=True))
        log(f"{s}: {len(b.bars)} 根 bar -> {path}")


if __name__ == "__main__":
    main()
