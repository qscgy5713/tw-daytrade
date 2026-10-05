import pandas as pd
import pytest

from twdt import live
from twdt.paper.journal import load_trades, save_trades
from twdt.backtest.engine import Trade

T = pd.Timestamp


def clock(times):
    it = iter(times)
    last = {"v": None}

    def now():
        try:
            last["v"] = next(it)
        except StopIteration:
            last["v"] = T("2026-10-05 14:00", tz=live.TZ)
        return last["v"]

    return now


def q(sym, t, day="2026-10-05"):
    return {"symbol": sym, "ts": T(f"{day} {t}"), "price": 100.0, "cum_volume": 1}


def test_poll_filters_stale_and_unknown_and_waits_for_open():
    got, logs = [], []
    # now_fn 呼叫順序:取 today -> 迴圈條件 -> 開盤判斷,每輪各一次
    times = [T(f"2026-10-05 {t}", tz=live.TZ) for t in [
        "08:59:00",                    # today
        "08:59:00", "08:59:00",        # 盤前:不抓
        "09:00:10", "09:00:10"]]       # 開盤:抓
    quotes = [q("2330", "09:00:05"), q("2330", "09:00:06", day="2026-10-02"), q("9999", "09:00:05")]
    fetched = []
    live.poll_loop(["2330"], "tse", 6, got.append, now_fn=clock(times),
                   fetch_fn=lambda s, m: fetched.append(1) or quotes,
                   sleep_fn=lambda s: None, log=logs.append)
    assert len(fetched) == 1  # 盤前沒有抓
    assert [x["symbol"] for x in got] == ["2330"] and got[0]["ts"].day == 5


def test_poll_warns_after_consecutive_stale_polls():
    logs = []
    times = [T("2026-10-05 10:00:00", tz=live.TZ)] * 40
    live.poll_loop(["2330"], "tse", 6, lambda q: None, now_fn=clock(times),
                   fetch_fn=lambda s, m: [q("2330", "09:00:00", day="2026-10-02")],
                   sleep_fn=lambda s: None, log=logs.append, stale_warn_after=10)
    assert sum("沒有今天的報價" in l for l in logs) == 1


def test_fetch_error_does_not_stop_but_handler_error_propagates():
    logs = []
    times = [T("2026-10-05 10:00:00", tz=live.TZ)] * 40

    def boom(s, m):
        raise ConnectionError("down")

    live.poll_loop(["2330"], "tse", 6, lambda q: None, now_fn=clock(times), fetch_fn=boom,
                   sleep_fn=lambda s: None, log=logs.append)
    assert any("ConnectionError" in l for l in logs)

    def bad_handler(_):
        raise ValueError("bug in trading logic")

    with pytest.raises(ValueError):
        live.poll_loop(["2330"], "tse", 6, bad_handler, now_fn=clock(times),
                       fetch_fn=lambda s, m: [q("2330", "10:00:00")],
                       sleep_fn=lambda s: None, log=logs.append)


def test_journal_roundtrip(tmp_path):
    ts_ = T("2026-10-05 09:16:05")
    tr = Trade("2330", 1, ts_, ts_ + pd.Timedelta(minutes=5), 100.0, 103.0, 1000, 3000.0, 800.0, "target")
    assert save_trades([], "2026-10-05", tmp_path) is None
    path = save_trades([tr], "2026-10-05", tmp_path)
    df = load_trades(tmp_path)
    assert path.exists() and len(df) == 1
    assert df.iloc[0]["net_pnl"] == pytest.approx(2200.0) and df.iloc[0]["exit_reason"] == "target"
