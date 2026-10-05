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
    # 限價單只成交在自己的掛價(101 * 1.02),即使報價已跳到 103.5 也不給更好的價
    assert t.exit_reason == "target" and t.exit_price == pytest.approx(101 * 1.02)
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


def test_reentry_waits_until_bar_after_exit_bar_like_backtest():
    """回測在 bar k 出場後,從 bar k+1 完成才重新評估(進場在 k+2 開盤)。"""
    from twdt.backtest.engine import run_day
    idx = pd.date_range("2026-10-05 09:00", periods=8, freq="1min")
    closes = [100.0, 100.0, 98.0, 98.0, 98.0, 98.0, 98.0, 98.0]
    bars = pd.DataFrame({"open": closes, "high": closes, "low": closes,
                         "close": closes, "volume": 1.0}, index=idx)
    risk = RiskConfig(stop_loss_pct=0.01)
    expected = run_day("X", bars, lambda b: 1, NO_SLIP, risk)
    tr = PaperTrader("X", lambda b: 1, NO_SLIP, risk)
    for i, (t, c) in enumerate(zip(idx, closes)):
        tr.on_quote(t, c, 1000 + i)
    got = [t.entry_time for t in tr.trades] + ([tr.position["entry_time"]] if tr.position else [])
    assert got[:2] == [ts("09:01:00"), ts("09:04:00")]
    assert got[:2] == [t.entry_time for t in expected][:2]


def test_whipsaw_does_not_reverse_into_short_immediately():
    """ORB 做多在 09:16 停損,該根同時收在區間下緣之下:回測不反手,模擬單也不可。"""
    from twdt.backtest.engine import run_day
    idx = pd.date_range("2026-10-05 09:00", periods=25, freq="1min")
    closes = [100.0] * 15 + [101.0, 98.0, 98.0, 98.0, 98.0, 98.0, 98.0, 98.0, 98.0, 98.0]
    bars = pd.DataFrame({"open": closes, "high": closes, "low": closes,
                         "close": closes, "volume": 1.0}, index=idx)
    expected = run_day("X", bars, opening_range_breakout(15), NO_SLIP, RiskConfig())
    tr = orb_trader()
    for i, (t, c) in enumerate(zip(idx, closes)):
        tr.on_quote(t, c, 1000 + i)
    tr.finish()
    assert [(t.direction, t.entry_time) for t in tr.trades] == \
        [(t.direction, t.entry_time) for t in expected]


def test_signal_evaluated_on_every_new_bar_when_quote_skips_minutes():
    """09:15 收 101(突破),09:16 無成交,下一筆在 09:17:接著補出的平盤 K 不可蓋掉突破訊號。"""
    tr = orb_trader()
    for i, t in enumerate(pd.date_range("2026-10-05 09:00", periods=15, freq="1min")):
        tr.on_quote(t, 100.0, 1000 + i)
    tr.on_quote(ts("09:15:10"), 101.0, 2000)
    tr.on_quote(ts("09:17:05"), 101.0, 2010)
    assert tr.position is not None and tr.position["direction"] == 1
    assert tr.position["entry_time"] == ts("09:17:05")


def test_entry_cutoff_uses_bar_start_minute_like_backtest():
    """12:59 出訊號、13:00:05 進場:bar 起始 13:00 <= 13:00,回測放行,模擬單也要放行。"""
    from twdt.backtest.engine import run_day
    calls = iter([0, 1] + [0] * 50)
    idx = pd.date_range("2026-10-05 12:58", periods=30, freq="1min")
    bars = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                         "close": 100.0, "volume": 1.0}, index=idx)
    expected = run_day("X", bars, lambda b: next(calls), NO_SLIP, RiskConfig())
    calls2 = iter([0, 1] + [0] * 50)
    tr = PaperTrader("X", lambda b: next(calls2), NO_SLIP, RiskConfig(),
                     started_at=T("2026-10-05 09:00:00"))
    feed(tr, [("12:58:05", 100.0), ("12:59:05", 100.0), ("13:00:05", 100.0), ("13:01:05", 100.0)])
    assert len(expected) == 1 and expected[0].entry_time == ts("13:00:00")
    assert tr.position is not None and tr.position["entry_time"] == ts("13:00:05")


def test_force_close_uses_last_continuous_price_not_auction_price():
    tr = orb_trader()
    feed(tr, breakout_ticks())
    feed(tr, [("13:24:50", 100.5), ("13:30:00", 101.5)])  # 13:25 後第一筆是集合競價價
    t = tr.trades[0]
    assert t.exit_reason == "force_close" and t.exit_price == pytest.approx(100.5)


def test_started_at_decides_late_start_not_first_quote_time():
    # 準時啟動(09:00:30),但冷門股到 09:06 才有第一筆成交:不應被停用
    tr = orb_trader(started_at=T("2026-10-05 09:00:30"))
    tr.on_quote(ts("09:06:10"), 100.0, 100)
    assert tr._enabled is True
    # 10:05 才啟動:停用
    events = []
    tr2 = orb_trader(started_at=T("2026-10-05 10:05:00"), on_event=events.append)
    tr2.on_quote(ts("10:05:10"), 100.0, 100)
    assert tr2._enabled is False and any("不交易" in e for e in events)


def test_finish_with_interrupted_reason():
    tr = orb_trader()
    feed(tr, breakout_ticks())
    tr.finish("interrupted")
    assert tr.trades[0].exit_reason == "interrupted"


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


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_matches_backtest_on_random_flat_bars(seed):
    """每分鐘一筆報價、每根 bar 只有單一價格時,兩邊的成交應完全一致(含滑價、停損、停利、強制平倉)。"""
    import numpy as np
    from twdt.backtest.engine import run_day
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-10-05 09:00", "2026-10-05 13:29", freq="1min")
    closes = np.round(500 * np.exp(np.cumsum(rng.normal(0, 0.0012, len(idx)))) * 2) / 2
    bars = pd.DataFrame({"open": closes, "high": closes, "low": closes,
                         "close": closes, "volume": 1.0}, index=idx)
    cost = CostModel(fee_discount=0.6, slippage_ticks=1)
    risk = RiskConfig(stop_loss_pct=0.004, take_profit_pct=0.006)
    expected = run_day("X", bars, opening_range_breakout(15), cost, risk)

    tr = PaperTrader("X", opening_range_breakout(15), cost, risk)
    for i, (t, c) in enumerate(zip(idx, closes)):
        tr.on_quote(t, float(c), 1000 + i)
    tr.finish()

    def key(t):
        return (t.direction, t.entry_time, t.exit_time, t.exit_reason,
                round(t.entry_price, 4), round(t.net_pnl, 2))

    # 停損:回測用 bar 內價格、模擬單用輪詢價,單一價格 bar 下兩者相同;
    # 停利:模擬單固定成交在掛價,回測在 bar 開盤高於掛價時成交在開盤價,故只比對非跳空的筆
    assert len(tr.trades) == len(expected)
    for got, exp in zip(tr.trades, expected):
        assert (got.direction, got.entry_time, got.exit_time, got.exit_reason) == \
            (exp.direction, exp.entry_time, exp.exit_time, exp.exit_reason)
        assert got.entry_price == pytest.approx(exp.entry_price)
        if exp.exit_reason != "target":
            assert got.exit_price == pytest.approx(exp.exit_price)
        else:
            assert got.exit_price <= exp.exit_price + 1e-9  # 模擬單不會比回測樂觀


def _momentum_signal(bars):
    """密集交易的訊號:最後一根收盤高於前一根做多、低於做空。專門用來壓測兩邊引擎的一致性。"""
    if len(bars) < 2:
        return 0
    d = bars["close"].iloc[-1] - bars["close"].iloc[-2]
    return 1 if d > 0 else (-1 if d < 0 else 0)


@pytest.mark.parametrize("seed", range(8))
def test_matches_backtest_with_dense_trading(seed):
    """多筆、多空雙向、含停損/停利/強制平倉/出場後重新評估,逐筆比對兩邊引擎。"""
    import numpy as np
    from twdt.backtest.engine import run_day
    rng = np.random.default_rng(100 + seed)
    idx = pd.date_range("2026-10-05 09:00", "2026-10-05 13:29", freq="1min")
    closes = np.round(500 * np.exp(np.cumsum(rng.normal(0, 0.0015, len(idx)))) * 2) / 2
    bars = pd.DataFrame({"open": closes, "high": closes, "low": closes,
                         "close": closes, "volume": 1.0}, index=idx)
    cost = CostModel(fee_discount=0.6, slippage_ticks=1)
    risk = RiskConfig(stop_loss_pct=0.003, take_profit_pct=0.004, max_daily_loss=10 ** 9)
    expected = run_day("X", bars, _momentum_signal, cost, risk)

    tr = PaperTrader("X", _momentum_signal, cost, risk)
    for i, (t, c) in enumerate(zip(idx, closes)):
        tr.on_quote(t, float(c), 1000 + i)
    tr.finish()

    assert len(expected) >= 10  # 確認測試真的有在壓力測試,不是空轉
    assert len(tr.trades) == len(expected)
    for got, exp in zip(tr.trades, expected):
        assert (got.direction, got.entry_time, got.exit_time, got.exit_reason) == \
            (exp.direction, exp.entry_time, exp.exit_time, exp.exit_reason)
        assert got.entry_price == pytest.approx(exp.entry_price)
        if exp.exit_reason == "target":
            # 回測在 bar 開盤已越過停利價時成交在開盤價(跳空);模擬單固定成交在掛價
            assert got.exit_price <= exp.exit_price + 1e-9 if exp.direction == 1 \
                else got.exit_price >= exp.exit_price - 1e-9
        else:
            assert got.exit_price == pytest.approx(exp.exit_price)
