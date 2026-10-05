import pandas as pd
import pytest

from twdt.data import realtime
from twdt.data.realtime import MinuteBarBuilder, parse_quote

T = pd.Timestamp


def test_parse_quote_ok_and_no_trade():
    item = {"c": "2330", "d": "20261005", "t": "09:56:24", "z": "2570.0000", "v": "11986"}
    q = parse_quote(item)
    assert q == {"symbol": "2330", "ts": T("2026-10-05 09:56:24"), "price": 2570.0,
                 "cum_volume": 11986}
    assert parse_quote({**item, "z": "-"}) is None
    assert parse_quote({"c": "2330"}) is None


def test_builder_ohlcv_and_volume_delta():
    b = MinuteBarBuilder()
    b.update(T("2026-10-05 09:00:10"), 100, 1000)  # 第一筆:既有累計量不計入
    b.update(T("2026-10-05 09:00:30"), 102, 1010)
    b.update(T("2026-10-05 09:00:50"), 99, 1015)
    b.update(T("2026-10-05 09:01:05"), 101, 1020)
    df = b.to_frame()
    assert len(df) == 1
    r = df.iloc[0]
    assert (r["open"], r["high"], r["low"], r["close"]) == (100, 102, 99, 99)
    assert r["volume"] == 15
    assert b.to_frame(include_open_bar=True).iloc[-1]["volume"] == 5


def test_builder_fills_gap_minutes_flat():
    b = MinuteBarBuilder()
    b.update(T("2026-10-05 09:00:10"), 100, 1000)
    b.update(T("2026-10-05 09:03:10"), 105, 1030)
    df = b.to_frame()
    assert list(df.index) == [T("2026-10-05 09:00"), T("2026-10-05 09:01"), T("2026-10-05 09:02")]
    assert (df.loc["2026-10-05 09:01":, "close"] == 100).all()
    assert (df.loc["2026-10-05 09:01":, "volume"] == 0).all()


def test_builder_ignores_out_of_order_and_stale_volume():
    b = MinuteBarBuilder()
    b.update(T("2026-10-05 09:05:10"), 100, 1000)
    b.update(T("2026-10-05 09:04:50"), 90, 990)  # 時間倒退,忽略
    b.update(T("2026-10-05 09:05:20"), 101, 995)  # 累計量倒退,量不為負
    df = b.to_frame(include_open_bar=True)
    assert len(df) == 1 and df.iloc[0]["low"] == 100 and df.iloc[0]["volume"] == 0


def test_save_and_load_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(realtime, "CACHE_DIR", tmp_path)
    b = MinuteBarBuilder()
    b.update(T("2026-10-05 09:00:10"), 100, 1000)
    b.update(T("2026-10-05 09:01:10"), 101, 1010)
    assert realtime.save_day("2330", b.to_frame(include_open_bar=True)) is not None
    df = realtime.load_minute("2330")
    assert len(df) == 2 and list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert realtime.save_day("2330", pd.DataFrame()) is None
    assert realtime.load_minute("9999").empty


def test_fetch_quotes_filters_no_trade(monkeypatch):
    class R:
        status_code = 200

        def json(self):
            return {"msgArray": [
                {"c": "2330", "d": "20261005", "t": "09:00:01", "z": "-", "v": "0"},
                {"c": "2317", "d": "20261005", "t": "09:00:02", "z": "200.0", "v": "5"}]}

    monkeypatch.setattr(realtime.requests, "get", lambda *a, **k: R())
    qs = realtime.fetch_quotes(["2330", "2317"])
    assert [q["symbol"] for q in qs] == ["2317"]
