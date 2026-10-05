"""樣本外選股:前段資料選(標的 x 停損停利),後段完全沒看過的資料只驗證選出來的那一個。

用法: python select_stock.py 2409 2340 6226 ... --dev-days 35
"""
import argparse
import time

from twdt import research
from twdt.backtest.costs import CostModel
from twdt.data import yahoo

STOPS, TARGETS = [0.005, 0.01], [0.01, 0.02]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="+")
    ap.add_argument("--dev-days", type=int, default=35)
    ap.add_argument("--interval", choices=["5m", "15m"], default="5m")
    ap.add_argument("--fee-discount", type=float, default=0.6)
    ap.add_argument("--min-trades", type=int, default=20)
    args = ap.parse_args()

    data = {}
    for s in args.symbols:
        try:
            data[s] = yahoo.fetch_intraday(s, args.interval)
        except yahoo.YahooError as e:
            print(f"[skip] {s}: {e}")
        time.sleep(1.0)  # 對非官方端點客氣一點
    dev, test, common = research.split_days(data, args.dev_days)
    print(f"共同交易日 {len(common)} 天:開發段 {common[0]}~{common[args.dev_days - 1]}({args.dev_days} 天),"
          f"驗證段 {common[args.dev_days]}~{common[-1]}({len(common) - args.dev_days} 天)\n")

    cost = CostModel(fee_discount=args.fee_discount)
    gd = research.evaluate_grid(dev, cost, STOPS, TARGETS, min_trades=args.min_trades)
    best = research.pick_best(gd)
    print(f"組合數 {len(gd)}(合格 {int(gd.eligible.sum())});開發段選出:{best.symbol} "
          f"停損 {best.stop:.1%} 停利 {best.target:.1%} 筆數 {int(best.trades)} "
          f"淨報酬合計 {best.net_ret_sum:.1%} 勝率 {best.win_rate:.0%}")

    # 驗證段:只評估「開發段選出的那一個」,這才是樣本外的正式結果
    gt = research.evaluate_grid({best.symbol: test[best.symbol]}, cost, [best.stop], [best.target], min_trades=1)
    t = gt.iloc[0]
    print(f"驗證段(樣本外):{best.symbol} 筆數 {int(t.trades)} 淨報酬合計 {t.net_ret_sum:.1%} "
          f"淨損益 {t.net_pnl:,.0f} 元 勝率 {t.win_rate:.0%}")

    # 診斷:開發段的好壞,在驗證段有沒有延續?(全部組合的排名相關;接近 0 代表挑選只是雜訊)
    gt_all = research.evaluate_grid(test, cost, STOPS, TARGETS, min_trades=1)
    m = gd[gd.eligible].merge(gt_all, on=["symbol", "stop", "target"], suffixes=("_dev", "_test"))
    if len(m) >= 5:
        rho = m["net_ret_sum_dev"].rank().corr(m["net_ret_sum_test"].rank())
        pos = int((m["net_ret_sum_dev"] > 0).sum())
        pos_t = int(((m["net_ret_sum_dev"] > 0) & (m["net_ret_sum_test"] > 0)).sum())
        print(f"\n診斷:開發段與驗證段的績效排名相關係數 {rho:+.2f}(接近 0 = 挑選只是雜訊)")
        print(f"      開發段為正的 {pos} 個組合中,驗證段仍為正的有 {pos_t} 個;"
              f"驗證段全部組合中為正的 {int((m['net_ret_sum_test'] > 0).sum())}/{len(m)} 個")


if __name__ == "__main__":
    main()
