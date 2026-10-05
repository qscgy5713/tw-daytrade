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
    assert parse_quote({**item, "z": "-"}) is None  # 頂層 z 為 '-' 且沒有 trade
    assert parse_quote({"c": "2330"}) is None


def test_builder_ohlcv_and_volume_delta():
    b = MinuteBarBuilder()
    b.update(T("2026-10-05 09:30:10"), 100, 1000)  # 盤中啟動:既有累計量不計入
    b.update(T("2026-10-05 09:30:30"), 102, 1010)
    b.update(T("2026-10-05 09:30:50"), 99, 1015)
    b.update(T("2026-10-05 09:31:05"), 101, 1020)
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


def test_opening_auction_volume_counted_when_started_at_open():
    b = MinuteBarBuilder()
    b.update(T("2026-10-05 09:00:05"), 100, 800)
    b.update(T("2026-10-05 09:00:30"), 101, 850)
    assert b.to_frame(include_open_bar=True).iloc[0]["volume"] == 850


def test_builder_ignores_cross_day_quotes():
    b = MinuteBarBuilder()
    b.update(T("2026-10-05 13:29:10"), 100, 5000)
    b.update(T("2026-10-06 09:00:10"), 105, 100)  # 隔天報價:忽略,不補上千根平盤
    assert len(b.to_frame(include_open_bar=True)) == 1


def test_save_day_merges_on_restart(monkeypatch, tmp_path):
    monkeypatch.setattr(realtime, "CACHE_DIR", tmp_path)
    first = MinuteBarBuilder()
    first.update(T("2026-10-05 09:00:10"), 100, 10)
    first.update(T("2026-10-05 09:01:10"), 101, 20)
    realtime.save_day("2330", first.to_frame(include_open_bar=True))
    second = MinuteBarBuilder()  # 中途重啟
    second.update(T("2026-10-05 09:05:10"), 103, 50)
    realtime.save_day("2330", second.to_frame(include_open_bar=True))
    df = realtime.load_minute("2330")
    assert list(df.index) == [T("2026-10-05 09:00"), T("2026-10-05 09:01"), T("2026-10-05 09:05")]


def test_parse_quote_falls_back_to_nested_trade_price():
    item = {"c": "2317", "d": "20261005", "t": "10:10:00", "z": "-", "v": "26277",
            "trade": {"t": "10:09:35", "v": 1, "z": "255.0000"}}
    q = parse_quote(item)
    assert q["price"] == 255.0 and q["cum_volume"] == 26277
    assert q["ts"] == T("2026-10-05 10:10:00")  # 時間用快照時間,量與價才對得起來
    # trade 也沒有成交價:視為尚無成交
    assert parse_quote({**item, "trade": {"t": "-", "z": "-"}}) is None
    assert parse_quote({**item, "trade": None}) is None
    # 頂層 z 有值時優先使用
    assert parse_quote({**item, "z": "256.0"})["price"] == 256.0


def test_save_day_merges_same_minute_across_restart(monkeypatch, tmp_path):
    monkeypatch.setattr(realtime, "CACHE_DIR", tmp_path)
    first = MinuteBarBuilder()
    first.update(T("2026-10-05 10:00:05"), 100, 1000)
    first.update(T("2026-10-05 10:00:30"), 103, 1020)   # 前半根: O100 H103 L100 C103 V20
    realtime.save_day("2330", first.to_frame(include_open_bar=True))
    second = MinuteBarBuilder()                          # 10:00:40 重啟
    second.update(T("2026-10-05 10:00:40"), 99, 1030)
    second.update(T("2026-10-05 10:00:55"), 101, 1050)  # 後半根: O99 H101 L99 C101 V0(首筆量不計)
    realtime.save_day("2330", second.to_frame(include_open_bar=True))
    row = realtime.load_minute("2330").iloc[0]
    assert (row["open"], row["high"], row["low"], row["close"]) == (100, 103, 99, 101)
    assert row["volume"] == 20 + 20
