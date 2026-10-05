"""盤中記錄分鐘線:每 N 秒輪詢證交所即時報價,自行組成分鐘線,收盤或 Ctrl+C 時存檔。

用法: python record_minutes.py 2330 2317 --interval 6
"""
import argparse
import time
from datetime import time as dtime

import pandas as pd

from twdt.data import realtime

# 13:30 收盤集合競價撮合,加上報價延遲約 5 秒,多等到 13:32 才收工
OPEN, CLOSE = dtime(9, 0), dtime(13, 32)
TZ = "Asia/Taipei"  # 不依賴本機時區


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="+")
    ap.add_argument("--interval", type=float, default=6.0, help="輪詢秒數,請 >= 5")
    ap.add_argument("--market", choices=["tse", "otc"], default="tse")
    args = ap.parse_args()
    if args.interval < 5:
        raise SystemExit("--interval 不可小於 5 秒(報價本身延遲 5 秒,過密易被限流)")

    builders = {s: realtime.MinuteBarBuilder() for s in args.symbols}
    today = pd.Timestamp.now(tz=TZ).date()
    polls_without_today = 0
    try:
        while pd.Timestamp.now(tz=TZ).time() < CLOSE:
            if pd.Timestamp.now(tz=TZ).time() >= OPEN:
                try:
                    quotes = realtime.fetch_quotes(args.symbols, args.market)
                    fresh = [q for q in quotes
                             if q["ts"].date() == today and q["symbol"] in builders]
                    # 假日或盤前 MIS 可能回前一交易日的舊報價,不可混進今天的檔
                    for q in fresh:
                        builders[q["symbol"]].update(q["ts"], q["price"], q["cum_volume"])
                    polls_without_today = 0 if fresh else polls_without_today + 1
                    if polls_without_today == 10:  # 約 1 分鐘都沒有今天的報價
                        print("[warn] 連續 10 次沒有今天的報價,今天可能休市或代號/市場(tse/otc)錯誤")
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
