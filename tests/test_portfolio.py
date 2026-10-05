import pandas as pd
import pytest

from twdt.backtest.costs import CostModel
from twdt.paper.engine import PaperTrader
from twdt.paper.portfolio import CapitalGuard
from twdt.risk.rules import RiskConfig

NO_SLIP = CostModel(fee_discount=1.0, slippage_ticks=0)
T = pd.Timestamp


def ts(hhmmss, day="2026-10-05"):
    return T(f"{day} {hhmmss}")


def test_guard_is_cumulative_and_release_does_not_refund():
    g = CapitalGuard(100_000)
    assert g.try_reserve("A", 60_000) is True
    assert g.try_reserve("B", 40_000) is True            # 剛好等於上限:允許
    assert g.used == 100_000 and g.entries == 2 and g.remaining == 0
    assert g.try_reserve("C", 1) is False                # 超過一塊就拒絕
    assert [r.symbol for r in g.rejections] == ["C"] and g.rejections[0].used == 100_000
    g.release("A")                                       # 平倉
    assert g.used == 100_000                             # 額度不退還(當沖額度當日不循環)
    assert g.try_reserve("C", 1) is False
    g.release("ZZZ")                                     # 釋放不存在的標的不會出錯


def test_same_symbol_reentry_consumes_quota_again():
    g = CapitalGuard(100_000)
    assert g.try_reserve("A", 40_000)
    g.release("A")
    assert g.try_reserve("A", 40_000)                    # 停損後再進場:要再扣一次
    assert g.used == 80_000 and g.entries == 2
    g.release("A")
    assert g.try_reserve("A", 40_000) is False           # 第三次就超過了


def test_guard_rejects_invalid_limit_and_double_entry_while_open():
    with pytest.raises(ValueError):
        CapitalGuard(0)
    g = CapitalGuard(1_000_000)
    g.try_reserve("A", 10)
    with pytest.raises(RuntimeError):
        g.try_reserve("A", 10)                           # 仍持倉卻再進場 = 呼叫端 bug,不可默默放行


def test_summary_lines_mentions_rejections():
    g = CapitalGuard(50_000)
    g.try_reserve("A", 40_000)
    g.try_reserve("B", 40_000, ts("09:20:05"))
    text = "\n".join(g.summary_lines())
    assert "上限 50,000" in text and "今日已用 40,000" in text
    assert "略過 1 次" in text and "B @ 09:20:05" in text


def _always_long_trader(sym, guard, events=None, **risk_kw):
    return PaperTrader(sym, lambda b: 1, NO_SLIP, RiskConfig(**risk_kw), guard=guard,
                       on_event=(events.append if events is not None else (lambda m: None)),
                       started_at=T("2026-10-05 09:00:00"))


def feed(tr, ticks):
    for i, (t, p) in enumerate(ticks):
        tr.on_quote(ts(t), p, 1000 + i * 10)


def test_second_symbol_is_rejected_when_quota_is_exhausted():
    guard = CapitalGuard(100_000)                        # 價格 100 x 1000 股 = 10 萬,只夠一次進場
    events = []
    a = _always_long_trader("A", guard, events)
    b = _always_long_trader("B", guard, events)
    for tr in (a, b):
        feed(tr, [("09:00:05", 100.0)])
    for tr in (a, b):                                    # 兩檔在同一輪報價都出訊號,A 先處理
        feed(tr, [("09:01:05", 100.0)])
    assert a.position is not None and b.position is None
    assert guard.used == 100_000 and len(guard.rejections) == 1
    assert any("[B]" in e and "單日額度不足" in e for e in events)
    assert b.trades == []


def test_quota_is_not_returned_after_exit_so_nobody_can_enter_again():
    guard = CapitalGuard(150_000)                        # 只夠一次 10 萬的進場
    a = _always_long_trader("A", guard, stop_loss_pct=0.01)
    b = _always_long_trader("B", guard, stop_loss_pct=0.01)
    for tr in (a, b):
        feed(tr, [("09:00:05", 100.0), ("09:01:05", 100.0)])   # A 進場、B 被擋
    feed(a, [("09:02:05", 98.0)])                               # A 停損出場
    assert a.position is None and len(a.trades) == 1
    assert guard.used == pytest.approx(100_000)                  # 平倉後額度仍是用掉的
    # A 之後的訊號想再進場、B 的訊號也想進場:額度都不夠(剩 5 萬 < 約 9.8 萬)
    feed(a, [("09:03:05", 98.0), ("09:04:05", 98.0)])
    feed(b, [("09:02:05", 100.0), ("09:03:05", 100.0), ("09:04:05", 100.0)])
    assert a.position is None and b.position is None
    assert guard.entries == 1 and len(guard.rejections) >= 2


def test_finish_does_not_refund_quota_but_marks_not_open():
    guard = CapitalGuard(1_000_000)
    a = _always_long_trader("A", guard)
    feed(a, [("09:00:05", 100.0), ("09:01:05", 100.0)])
    used = guard.used
    assert used > 0
    a.finish("interrupted")
    assert guard.used == used and a.trades[0].exit_reason == "interrupted"


def test_single_entry_larger_than_limit_never_happens():
    guard = CapitalGuard(50_000)
    events = []
    a = _always_long_trader("A", guard, events)
    feed(a, [("09:00:05", 100.0), ("09:01:05", 100.0), ("09:02:05", 100.0)])
    assert a.position is None and a.trades == []
    assert len(guard.rejections) >= 1 and guard.used == 0


def test_notional_uses_slippage_adjusted_entry_price_and_counts_shorts():
    guard = CapitalGuard(1_000_000)
    cost = CostModel(fee_discount=1.0, slippage_ticks=1)
    short = PaperTrader("S", lambda b: -1, cost, RiskConfig(), guard=guard,
                        started_at=T("2026-10-05 09:00:00"))
    feed(short, [("09:00:05", 100.0), ("09:01:05", 100.0)])
    assert short.position["direction"] == -1
    assert guard.used == pytest.approx(short.position["entry"] * 1000)   # 放空(先賣後買的賣出)也占額度,含滑價


def test_no_guard_keeps_previous_behaviour():
    a = PaperTrader("A", lambda b: 1, NO_SLIP, RiskConfig(), started_at=T("2026-10-05 09:00:00"))
    feed(a, [("09:00:05", 100.0), ("09:01:05", 100.0)])
    assert a.position is not None


def _momentum(bars):
    if len(bars) < 2:
        return 0
    d = bars["close"].iloc[-1] - bars["close"].iloc[-2]
    return 1 if d > 0 else (-1 if d < 0 else 0)


def test_dense_run_total_entries_are_capped_by_daily_quota():
    """密集交易:一天所有進場金額加總不得超過上限,因此全日進場次數被額度硬性封頂。"""
    import numpy as np
    limit = 200_000                                     # 每次約 4.5 萬 -> 全日最多 4 次進場
    guard = CapitalGuard(limit)
    traders = [PaperTrader(f"S{i}", _momentum, CostModel(),
                           RiskConfig(stop_loss_pct=0.003, take_profit_pct=0.004, max_daily_loss=10 ** 9),
                           guard=guard, started_at=T("2026-10-05 09:00:00")) for i in range(5)]
    rng = np.random.default_rng(7)
    idx = pd.date_range("2026-10-05 09:00", "2026-10-05 13:29", freq="1min")
    paths = [np.round(45 * np.exp(np.cumsum(rng.normal(0, 0.0015, len(idx)))) * 20) / 20 for _ in traders]
    for k, t in enumerate(idx):
        for tr, path in zip(traders, paths):
            tr.on_quote(t, float(path[k]), 1000 + k)
        assert guard.used <= limit + 1e-6
    for tr in traders:
        tr.finish()
    n_trades = sum(len(tr.trades) for tr in traders)
    assert n_trades == guard.entries                    # 每筆成交都對應一次額度扣除
    assert 1 <= guard.entries <= 4                      # 被額度封頂(無額度時同情境有幾十筆)
    assert len(guard.rejections) > 50                   # 其餘訊號都被擋下
    assert guard.used <= limit
