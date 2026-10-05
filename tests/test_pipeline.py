from run_backtest import backtest
from twdt.backtest.costs import CostModel
from twdt.data.synthetic import synthetic_minute
from twdt.report.metrics import summarize
from twdt.risk.rules import RiskConfig


def test_synthetic_shape_and_ohlc_valid():
    df = synthetic_minute("2026-10-05", "2026-10-06", seed=1)
    assert len(df) == 2 * 271
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()


def test_synthetic_is_deterministic():
    a = synthetic_minute("2026-10-05", "2026-10-05", seed=3)
    b = synthetic_minute("2026-10-05", "2026-10-05", seed=3)
    assert a.equals(b)


def test_pipeline_end_to_end():
    data = {"A": synthetic_minute("2026-09-01", "2026-09-30", seed=0),
            "B": synthetic_minute("2026-09-01", "2026-09-30", seed=1)}
    trades = backtest(data, CostModel(), RiskConfig(), 15)
    m = summarize(trades)
    assert m["trades"] > 0
    assert all(t.exit_time >= t.entry_time for t in trades)
    assert all(t.exit_time.time() <= RiskConfig().force_close_at for t in trades)
    # 成本是確定的拖累,必定大於 0(隨機漫步的損益本身沒有意義,不斷言)
    assert m["total_cost"] > 0
