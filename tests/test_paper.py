import pandas as pd
import pytest

from twdt.backtest.costs import CostModel
from twdt.paper.engine import PaperTrader
from twdt.risk.rules import RiskConfig
from twdt.signals.rules import opening_range_breakout

NO_SLIP = CostModel(fee_discount=1.0, slippage_ticks=0)
T = pd.Timestamp


def ts(hhmmss, day="2026-10-05"):
    return T(f"{day} {hhmmss}")


def feed(trader, ticks):
    """ticks: [(hh:mm:ss, price)],累計量每筆 +10。"""
    for i, (t, price) in enumerate(ticks):
        trader.on_quote(ts(t), price, 1000 + i * 10)


def flat_open(price=100.0, minutes=15):
    """09:00~09:14 每分鐘兩筆同價報價,形成開盤區間 [price, price]。"""
    out = []
    for m in range(minutes):
        out += [(f"09:{m:02d}:05", price), (f"09:{m:02d}:35", price)]
    return out


def orb_trader(**kw):
    return PaperTrader("X", opening_range_breakout(15), kw.pop("cost", NO_SLIP),
                       kw.pop("risk", RiskConfig()), **kw)


def breakout_ticks(price=101.0):
    # 09:15 這根收 101(突破);09:16 第一筆報價時該 bar 完成,於此成交
    return flat_open() + [("09:15:05", price), ("09:15:40", price), ("09:16:05", price)]


def test_entry_then_target():
    tr = orb_trader()
    feed(tr, breakout_ticks())
    assert tr.position is not None and tr.position["direction"] == 1
    assert tr.position["entry_time"] == ts("09:16:05")
    feed(tr, [("09:16:30", 103.5)])
    t = tr.trades[0]
    assert t.exit_reason == "target" and t.exit_price == pytest.approx(103.5)
    assert t.net_pnl < t.gross_pnl and tr.position is None


def test_stop_exit_with_slippage():
    tr = orb_trader(cost=CostModel(fee_discount=1.0, slippage_ticks=1))
    feed(tr, breakout_ticks())
    entry = tr.position["entry"]
    feed(tr, [("09:16:30", 99.0)])
    t = tr.trades[0]
    assert t.exit_reason == "stop"
    assert t.exit_price < 99.0  # 停損吃不利滑價
    assert t.entry_price > 101.0 and entry == t.entry_price


def test_force_close_at_1325():
    tr = orb_trader()
    feed(tr, breakout_ticks())
    feed(tr, [("13:24:50", 101.0), ("13:25:03", 101.2)])
    assert tr.trades[0].exit_reason == "force_close"
    assert tr.trades[0].exit_time == ts("13:25:03")


def test_finish_closes_open_position_at_last_price():
    tr = orb_trader()
    feed(tr, breakout_ticks())
    tr.finish()
    assert tr.trades[0].exit_reason == "force_close" and tr.position is None
    tr.finish()  # 重複呼叫不會多出交易
    assert len(tr.trades) == 1


def test_late_start_disables_trading_and_warns():
    events = []
    tr = orb_trader(on_event=events.append)
    feed(tr, [("09:30:05", 100.0), ("09:31:05", 105.0), ("09:32:05", 110.0)])
    assert tr.trades == [] and tr.position is None
    assert any("不交易" in e for e in events)


def test_no_flip_on_same_tick_as_exit():
    tr = PaperTrader("X", lambda b: 1, NO_SLIP, RiskConfig(stop_loss_pct=0.01))
    feed(tr, [("09:00:05", 100.0), ("09:01:05", 100.0)])  # 09:00 bar 完成 -> 進場
    assert tr.position is not None
    # 09:02:05 這一筆同時「完成 09:01 bar」與「觸及停損」:出場,且不可在同一筆立刻再進場
    feed(tr, [("09:02:05", 98.0)])
    assert tr.trades[0].exit_reason == "stop"
    assert tr.position is None
    # 下一根 bar 完成時才重新評估並進場(與回測一致)
    feed(tr, [("09:03:05", 98.0)])
    assert tr.position is not None and tr.position["entry_time"] == ts("09:03:05")


def test_daily_loss_limit_blocks_new_entries():
    tr = PaperTrader("X", lambda b: 1, NO_SLIP, RiskConfig(max_daily_loss=500))
    feed(tr, [("09:00:05", 100.0), ("09:01:05", 100.0), ("09:01:30", 98.0),
              ("09:02:05", 98.0), ("09:03:05", 98.0), ("09:04:05", 98.0)])
    assert len(tr.trades) == 1 and tr.realized_pnl <= -500 and tr.position is None


def test_no_entry_after_cutoff():
    tr = PaperTrader("X", lambda b: 1, NO_SLIP, RiskConfig())
    feed(tr, [("09:00:05", 100.0), ("12:59:05", 100.0)])
    tr.position = None
    tr.trades.clear()
    feed(tr, [("13:01:05", 100.0), ("13:02:05", 100.0)])  # 13:01 後新完成的 bar 不得進場
    assert tr.position is None and tr.trades == []


def test_ignores_other_day_and_out_of_order_quotes():
    tr = orb_trader()
    tr.on_quote(ts("09:00:05"), 100.0, 1000)
    tr.on_quote(ts("09:00:05", day="2026-10-06"), 500.0, 5)  # 隔天
    tr.on_quote(ts("09:00:01"), 1.0, 1)  # 時間倒退
    df = tr.builder.to_frame(include_open_bar=True)
    assert len(df) == 1 and df.iloc[0]["high"] == 100.0


def test_short_direction():
    tr = PaperTrader("X", lambda b: -1, NO_SLIP, RiskConfig())
    feed(tr, [("09:00:05", 100.0), ("09:01:05", 100.0), ("09:01:30", 97.0)])
    t = tr.trades[0]
    assert t.direction == -1 and t.exit_reason == "target" and t.gross_pnl > 0


def test_matches_backtest_on_same_bars():
    """相同分鐘線下,模擬單與回測引擎的進出場結果應一致(用每分鐘一筆報價近似)。"""
    from twdt.backtest.engine import run_day
    idx = pd.date_range("2026-10-05 09:00", periods=40, freq="1min")
    closes = [100.0] * 15 + [101.0, 101.2, 103.5] + [103.5] * 22
    bars = pd.DataFrame({"open": closes, "high": closes, "low": closes,
                         "close": closes, "volume": 1.0}, index=idx)
    expected = run_day("X", bars, opening_range_breakout(15), NO_SLIP, RiskConfig())
    tr = orb_trader()
    for i, (t, c) in enumerate(zip(idx, closes)):
        tr.on_quote(t, c, 1000 + i)
    assert len(tr.trades) == len(expected) == 1
    assert tr.trades[0].entry_time == expected[0].entry_time
    assert tr.trades[0].exit_reason == expected[0].exit_reason == "target"
