"""盤中模擬單:輪詢即時報價,以 ORB 基準策略虛擬下單,同時記錄分鐘線與成交。

用法: python paper_trade.py 2330 2317 --interval 6
請在 09:00 前啟動;盤中才啟動的標的因開盤區間不完整,當天不會交易。
每 --flush-every 秒自動存分鐘線與成交紀錄,所有事件同時寫進 cache/logs/paper_<日期>.log。
"""
import argparse

import pandas as pd

from twdt import live
from twdt.backtest.costs import CostModel
from twdt.data import realtime
from twdt.data.stats import QuoteStats
from twdt.logutil import file_logger
from twdt.paper.engine import PaperTrader
from twdt.paper.portfolio import CapitalGuard
from twdt.paper.journal import save_trades
from twdt.report.metrics import summarize
from twdt.risk.rules import RiskConfig
from twdt.signals.rules import opening_range_breakout


def report_stats(stats, session_started, log):
    """統計只是輔助資訊:任何失敗只記警告,不可影響收尾存檔或掩蓋原本的錯誤。"""
    try:
        for line in stats.format_lines():
            log(line)
        log(f"密度明細: {stats.save(session_started)}")
    except Exception as e:
        log(f"[warn] 報價密度統計失敗 {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="+")
    ap.add_argument("--interval", type=float, default=6.0, help="輪詢秒數,請 >= 5")
    ap.add_argument("--market", choices=["tse", "otc"], default="tse")
    ap.add_argument("--or-minutes", type=int, default=15)
    ap.add_argument("--fee-discount", type=float, default=0.6)
    ap.add_argument("--capital", type=float, default=300000.0,
                    help="單日買賣額度(元):一天內所有進場金額累計不可超過此值,平倉不退還額度")
    ap.add_argument("--quota-legs", choices=["both", "entry"], default="both",
                    help="額度算法:both=買賣兩腿都占額度(保守,預設);entry=只算進場那一腿")
    ap.add_argument("--flush-every", type=float, default=300.0, help="定期存檔間隔(秒)")
    args = ap.parse_args()
    if args.interval < 5:
        raise SystemExit("--interval 不可小於 5 秒(報價本身延遲 5 秒,過密易被限流)")

    log = file_logger("paper")
    log(f"開始模擬單 {args.symbols} market={args.market} interval={args.interval}s log={log.path}")
    cost, risk = CostModel(fee_discount=args.fee_discount), RiskConfig()
    started_at = pd.Timestamp.now(tz=live.TZ).tz_localize(None)
    guard = CapitalGuard(args.capital, count_both_legs=(args.quota_legs == "both"))
    log(f"單日買賣額度 {guard.limit:,.0f}({'買賣兩腿都算' if guard.count_both_legs else '只算進場'}、逐筆累計、平倉不退還);"
        f"每筆 {risk.shares_per_trade} 股")
    traders = {s: PaperTrader(s, opening_range_breakout(args.or_minutes), cost, risk,
                              started_at=started_at, guard=guard, on_event=log)
               for s in args.symbols}
    day = pd.Timestamp.now(tz=live.TZ).strftime("%Y-%m-%d")
    recorders = {s: realtime.DayRecorder(s) for s in args.symbols}
    stats = QuoteStats()
    session_started = started_at

    def on_quote(q):
        stats.observe(q)
        traders[q["symbol"]].on_quote(q["ts"], q["price"], q["cum_volume"])

    def flush():
        # 分鐘線與「已結束的成交」定期落地;成交紀錄以 (代號, 進出場時間) 去重,可重複寫入
        for s, t in traders.items():
            recorders[s].flush(t.builder.to_frame(include_open_bar=True))
        save_trades([tr for t in traders.values() for tr in t.trades], day)
        try:
            stats.save(session_started)
        except Exception as e:
            log(f"[warn] 統計存檔失敗 {type(e).__name__}: {e}")

    interrupted = False
    try:
        live.poll_loop(args.symbols, args.market, args.interval,
                       on_quote,
                       log=log, flush_fn=flush, flush_every=args.flush_every)
    except KeyboardInterrupt:
        interrupted = True
        log("中斷,結算中…")
    except Exception as e:
        log(f"[error] 交易邏輯例外 {type(e).__name__}: {e}")  # 進 log 檔,之後仍會結算存檔再往外拋
        raise
    finally:
        all_trades = []
        for s, t in traders.items():
            t.finish("interrupted" if interrupted else "force_close")  # 還有部位就以最後報價平倉
            recorders[s].flush(t.builder.to_frame(include_open_bar=True))
            all_trades += t.trades
        path = save_trades(all_trades, day)
        log(f"成交紀錄: {path}")
        for line in guard.summary_lines():
            log(line)
        report_stats(stats, session_started, log)
        for k, v in summarize(all_trades).items():
            log(f"{k:>20}: {v:,.2f}" if isinstance(v, float) else f"{k:>20}: {v}")


if __name__ == "__main__":
    main()
