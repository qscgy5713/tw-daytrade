from datetime import time

import pandas as pd
import pytest

from twdt.backtest.costs import CostModel, tick_size
from twdt.backtest.engine import run_day
from twdt.risk.rules import RiskConfig


def make_bars(closes, start="2026-10-05 09:00", opens=None, highs=None, lows=None):
    idx = pd.date_range(start, periods=len(closes), freq="1min")
    opens = opens or closes
    highs = highs or [max(o, c) for o, c in zip(opens, closes)]
    lows = lows or [min(o, c) for o, c in zip(opens, closes)]
    return pd.DataFrame({"open": opens, "high": highs, "low": lows,
                         "close": closes, "volume": 1000}, index=idx)


NO_SLIP = CostModel(fee_discount=1.0, slippage_ticks=0)


def test_tick_size():
    assert tick_size(9.9) == 0.01
    assert tick_size(49.9) == 0.05
    assert tick_size(100) == 0.5
    assert tick_size(1000) == 5.0


def test_round_trip_cost_long():
    # 買 100 賣 101,1000 股:手續費 142.5 + 143.9 稅 151.5
    c = NO_SLIP.round_trip_cost(100, 101, 1000, 1)
    assert c == pytest.approx(100000 * 0.001425 + 101000 * 0.001425 + 101000 * 0.0015)


def test_min_fee():
    assert NO_SLIP.fee(10, 1) == 20.0


def test_slippage_direction():
    m = CostModel(slippage_ticks=1)
    assert m.fill_price(100, "buy") > 100
    assert m.fill_price(100, "sell") < 100


def test_target_hit_long():
    bars = make_bars([100, 100, 100, 103, 103], highs=[100, 100, 100, 103, 103])
    risk = RiskConfig(stop_loss_pct=0.01, take_profit_pct=0.02)
    fired = iter([0, 1] + [0] * 10)
    trades = run_day("2330", bars, lambda b: next(fired), NO_SLIP, risk)
    assert len(trades) == 1
    t = trades[0]
    assert t.exit_reason == "target" and t.direction == 1
    assert t.entry_time == bars.index[2]  # 訊號在 bar1,下一根開盤進場
    assert t.net_pnl < t.gross_pnl


def test_stop_before_target_same_bar():
    bars = make_bars([100, 100, 100, 100], highs=[100, 100, 100, 105], lows=[100, 100, 100, 95])
    risk = RiskConfig()
    fired = iter([1] + [0] * 10)
    trades = run_day("X", bars, lambda b: next(fired), NO_SLIP, risk)
    assert trades[0].exit_reason == "stop"


def test_short_pnl_positive_when_price_falls():
    bars = make_bars([100, 100, 97, 97], lows=[100, 100, 97, 97])
    risk = RiskConfig()
    fired = iter([-1] + [0] * 10)
    t = run_day("X", bars, lambda b: next(fired), NO_SLIP, risk)[0]
    assert t.direction == -1 and t.gross_pnl > 0


def test_force_close():
    idx = pd.date_range("2026-10-05 13:20", periods=10, freq="1min")
    bars = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                         "close": 100.0, "volume": 1}, index=idx)
    risk = RiskConfig(no_entry_after=time(13, 24))
    fired = iter([1] + [0] * 20)
    t = run_day("X", bars, lambda b: next(fired), NO_SLIP, risk)[0]
    assert t.exit_reason == "force_close"
    assert t.exit_time.time() == time(13, 25)


def test_no_entry_after_cutoff():
    idx = pd.date_range("2026-10-05 13:01", periods=10, freq="1min")
    bars = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                         "close": 100.0, "volume": 1}, index=idx)
    assert run_day("X", bars, lambda b: 1, NO_SLIP, RiskConfig()) == []


def test_daily_loss_limit_stops_new_entries():
    closes = [100, 100, 98, 98, 98, 98, 96, 96, 96, 96, 96, 96]
    bars = make_bars(closes, lows=closes)
    risk = RiskConfig(stop_loss_pct=0.01, max_daily_loss=1500)
    trades = run_day("X", bars, lambda b: 1, NO_SLIP, risk)
    assert sum(t.net_pnl for t in trades) <= -1500
    assert len(trades) == 1  # 第一筆虧損已超過上限


def test_orb_signal():
    from twdt.signals.rules import opening_range_breakout
    closes = [100] * 15 + [101, 102]
    bars = make_bars(closes)
    sig = opening_range_breakout(15)
    assert sig(bars.iloc[:10]) == 0
    assert sig(bars.iloc[:16]) == 1


def test_entry_cutoff_uses_entry_bar():
    # 13:00 出訊號,進場會落在 13:01,應被擋下
    idx = pd.date_range("2026-10-05 12:59", periods=10, freq="1min")
    bars = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                         "close": 100.0, "volume": 1}, index=idx)
    fired = iter([0, 1] + [0] * 20)
    assert run_day("X", bars, lambda b: next(fired), NO_SLIP, RiskConfig()) == []


def test_target_exit_has_no_slippage():
    bars = make_bars([100, 100, 100, 103, 103], opens=[100] * 5,
                     highs=[100, 100, 100, 103, 103])
    risk = RiskConfig(stop_loss_pct=0.01, take_profit_pct=0.02)
    fired = iter([0, 1] + [0] * 10)
    t = run_day("X", bars, lambda b: next(fired), CostModel(slippage_ticks=1), risk)[0]
    assert t.exit_reason == "target"
    assert t.exit_price == pytest.approx(t.entry_price * 1.02)


def test_orb_tz_aware_and_single_fire():
    from twdt.signals.rules import opening_range_breakout
    closes = [100] * 15 + [101, 102, 103]
    bars = make_bars(closes)
    bars.index = bars.index.tz_localize("Asia/Taipei")
    sig = opening_range_breakout(15)
    assert sig(bars.iloc[:16]) == 1
    assert sig(bars.iloc[:17]) == 0  # 已突破過,不重複觸發


def test_sell_slippage_uses_tick_below_boundary():
    m = CostModel(slippage_ticks=1)
    assert m.fill_price(100, "sell") == pytest.approx(99.9)   # 100 以下的 tick 是 0.1
    assert m.fill_price(500, "sell") == pytest.approx(499.5)
    assert m.fill_price(1000, "sell") == pytest.approx(999.0)
    assert m.fill_price(100, "buy") == pytest.approx(100.5)   # 買進往上,用 100 以上的 tick


def test_force_close_uses_last_bar_before_1325_and_skips_stops():
    idx = pd.date_range("2026-10-05 13:21", periods=9, freq="1min")  # 13:21 ~ 13:29
    closes = [100.0, 100.0, 100.0, 100.0, 100.5, 90.0, 90.0, 90.0, 90.0]  # 13:25 起暴跌
    bars = pd.DataFrame({"open": closes, "high": closes, "low": closes,
                         "close": closes, "volume": 1}, index=idx)
    fired = iter([1] + [0] * 20)
    t = run_day("X", bars, lambda b: next(fired), NO_SLIP, RiskConfig(no_entry_after=time(13, 24)))[0]
    assert t.exit_reason == "force_close"          # 13:25 起不再檢查停損
    assert t.exit_price == pytest.approx(100.0)    # 13:24 那根收盤,不是 13:25 的 90
