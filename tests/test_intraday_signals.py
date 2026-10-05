from datetime import date, time

import pandas as pd
import pytest

from twdt.signals import intraday as sg

T = pd.Timestamp


def bars(closes, start="2026-10-05 09:00", opens=None, highs=None, lows=None, vols=None):
    idx = pd.date_range(start, periods=len(closes), freq="5min")
    opens = opens or closes
    return pd.DataFrame({"open": opens,
                         "high": highs or [max(o, c) for o, c in zip(opens, closes)],
                         "low": lows or [min(o, c) for o, c in zip(opens, closes)],
                         "close": closes, "volume": vols or [100.0] * len(closes)}, index=idx)


def test_open_move_fires_only_on_trigger_bar_and_direction():
    closes = [100, 100.2, 100.5, 100.8, 101.0, 101.5]       # 09:00..09:25,最後一根是 09:25,漲 1.5%
    b = bars(closes, opens=[100] * 6)
    assert sg.open_move(0.01, direction=1)(b) == 1           # 順勢做多
    assert sg.open_move(0.01, direction=-1)(b) == -1         # 鏡像:反轉做空
    assert sg.open_move(0.01)(b.iloc[:5]) == 0               # 不是 trigger 那根 bar
    assert sg.open_move(0.02)(b) == 0                        # 沒達門檻
    down = bars([100, 99.5, 99.0, 98.8, 98.7, 98.5], opens=[100] * 6)
    assert sg.open_move(0.01)(down) == -1 and sg.open_move(0.01, direction=-1)(down) == 1


def test_gap_fade_directions_and_requires_unfilled_gap_and_prev_close():
    up = bars([103.2, 103.0, 103.1], opens=[103.0] * 3)      # 前收 100,開盤 103(+3%)未回補
    s = sg.gap_fade({date(2026, 10, 5): 100.0}, 0.015)
    assert s(up) == -1
    assert sg.gap_fade({}, 0.015)(up) == 0                    # 沒有前收盤(第一天)不交易
    filled = bars([102.0, 101.0, 99.5], opens=[103.0] * 3)    # 已跌破前收 -> 缺口回補,不做
    assert s(filled) == 0
    down = bars([96.8, 97.0, 96.9], opens=[97.0] * 3)         # 前收 100,開盤 97(-3%)未回補 -> 做多
    assert s(down) == 1
    small = bars([100.8, 100.9, 100.7], opens=[101.0] * 3)    # 跳空只有 1%
    assert s(small) == 0
    assert s(up.iloc[:2]) == 0                                # 還沒到 trigger 那根


def test_vwap_fade_fires_once_on_fresh_deviation_only():
    base = [100.0] * 8                                        # 09:00~09:35 均價 100
    below = bars(base + [98.0], vols=[100.0] * 9)             # 09:40 收 98:低於 VWAP 約 -1.8% -> 做多
    assert sg.vwap_fade(0.015)(below) == 1
    again = bars(base + [98.0, 97.9], vols=[100.0] * 10)      # 已經在範圍外,不重複觸發
    assert sg.vwap_fade(0.015)(again) == 0
    above = bars(base + [102.0], vols=[100.0] * 9)
    assert sg.vwap_fade(0.015)(above) == -1
    assert sg.vwap_fade(0.015)(below.iloc[:5]) == 0           # 09:30 之前不交易
    assert sg.vwap_fade(0.015)(bars([100.0] * 9, vols=[0.0] * 9)) == 0   # 沒有成交量不算 VWAP


def test_new_extreme_after_cutoff_only():
    base = [100.0] * 12                                       # 09:00~09:55
    up = bars(base + [101.0])                                 # 10:00 突破新高
    assert sg.new_extreme()(up) == 1
    dn = bars(base + [99.0])
    assert sg.new_extreme()(dn) == -1
    early = bars([100.0] * 5 + [101.0])                       # 09:25 就突破:還沒到 10:00
    assert sg.new_extreme()(early) == 0
    inside = bars(base + [100.0])
    assert sg.new_extreme()(inside) == 0


def test_random_at_is_seeded_and_only_on_trigger():
    b = bars([100] * 6)
    a1 = [sg.random_at(seed=1)(b) for _ in range(3)]
    a2 = [sg.random_at(seed=1)(b) for _ in range(3)]
    assert a1 == a2 and set(a1) <= {1, -1}
    assert sg.random_at()(b.iloc[:4]) == 0


def test_prev_close_map_uses_previous_day_last_bar():
    d1 = bars([100, 101], start="2026-10-02 13:15")
    d2 = bars([105, 106], start="2026-10-05 09:00")
    m = sg.prev_close_map(pd.concat([d1, d2]))
    assert m == {date(2026, 10, 5): 101.0}


def test_run_symbol_accepts_custom_signal():
    from twdt import research
    from twdt.backtest.costs import CostModel
    idx = pd.date_range("2026-10-05 09:00", "2026-10-05 13:20", freq="5min")
    closes = [100.0] * 5 + [101.5] * (len(idx) - 5)   # 09:25 這根(觸發 bar)收在 +1.5%
    df = pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes, "volume": 1.0}, index=idx)
    tr = research.run_symbol(df, "X", CostModel(fee_discount=1.0, slippage_ticks=0),
                             research.RiskConfig(), signal=sg.open_move(0.01, direction=1))
    assert len(tr) == 1 and tr.iloc[0]["dir"] == 1 and "day" in tr.columns
