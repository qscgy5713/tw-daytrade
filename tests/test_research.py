import pandas as pd
import pytest

from twdt import research
from twdt.backtest.costs import CostModel


def frame(days, price=100.0):
    parts = []
    for d in days:
        idx = pd.date_range(f"{d} 09:00", f"{d} 13:20", freq="5min")
        parts.append(pd.DataFrame({"open": price, "high": price, "low": price, "close": price,
                                   "volume": 1.0}, index=idx))
    return pd.concat(parts)


def test_split_is_chronological_and_disjoint_using_common_days_only():
    days = [f"2026-09-{d:02d}" for d in range(1, 11)]
    a = frame(days)
    b = frame(days[2:])                                   # B 缺前兩天
    dev, test, common = research.split_days({"A": a, "B": b}, n_dev=4)
    assert [str(d) for d in common] == days[2:]           # 只用共同交易日
    dev_days = {d for df in dev.values() for d in df.index.date}
    test_days = {d for df in test.values() for d in df.index.date}
    assert dev_days.isdisjoint(test_days)
    assert max(dev_days) < min(test_days)                 # 開發段全部早於驗證段(沒有偷看未來)
    assert len(dev_days) == 4 and len(test_days) == 4


def test_split_rejects_bad_sizes():
    with pytest.raises(ValueError):
        research.split_days({"A": frame(["2026-09-01", "2026-09-02"])}, n_dev=2)
    with pytest.raises(ValueError):
        research.split_days({"A": frame(["2026-09-01", "2026-09-02"])}, n_dev=0)


def test_evaluate_grid_and_pick_best_respect_min_trades():
    rising = frame(["2026-09-01"])
    closes = [100.0] * 4 + [101.0] * 8 + [103.0] * (len(rising) - 12)
    rising["close"] = closes
    rising["open"] = closes
    rising["high"] = [c + 0.1 for c in closes]
    rising["low"] = [c - 0.1 for c in closes]
    grid = research.evaluate_grid({"UP": rising, "FLAT": frame(["2026-09-01"])}, CostModel(),
                                  stops=[0.01], targets=[0.02], min_trades=1)
    assert set(grid["symbol"]) == {"UP", "FLAT"}
    flat = grid[grid.symbol == "FLAT"].iloc[0]
    assert flat["trades"] == 0 and not flat["eligible"]   # 沒有交易的組合不可入選
    assert research.pick_best(grid)["symbol"] == "UP"
    none_ok = research.evaluate_grid({"FLAT": frame(["2026-09-01"])}, CostModel(), [0.01], [0.02], min_trades=5)
    with pytest.raises(ValueError):
        research.pick_best(none_ok)


def test_ret_is_normalised_by_entry_notional():
    df = frame(["2026-09-01"])
    closes = [100.0] * 4 + [101.5] * (len(df) - 4)
    df["close"] = closes
    df["open"] = closes
    df["high"] = [c if i < 4 else c + 3 for i, c in enumerate(closes)]   # 開盤區間維持平,突破後才放大高點
    df["low"] = closes
    tr = research.run_symbol(df, "X", CostModel(fee_discount=1.0, slippage_ticks=0),
                             research.RiskConfig(stop_loss_pct=0.05, take_profit_pct=0.02), 15)
    assert len(tr) >= 1
    row = tr.iloc[0]
    assert row["ret"] == pytest.approx(row["net"] / (101.5 * 1000), rel=0.05)
