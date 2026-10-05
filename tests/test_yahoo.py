import pandas as pd
import pytest

from twdt.data import yahoo

T = pd.Timestamp


def utc_stamp(local: str) -> int:
    """台灣當地時間字串 -> epoch 秒(Yahoo 回的是 UTC epoch)。"""
    return int(T(local, tz="Asia/Taipei").tz_convert("UTC").timestamp())


def result(rows):
    """rows: [(local_time, o, h, l, c, v)];None 代表空值 bar。"""
    stamps = [utc_stamp(r[0]) for r in rows]
    cols = {k: [] for k in ("open", "high", "low", "close", "volume")}
    for r in rows:
        for k, v in zip(cols, r[1:]):
            cols[k].append(v)
    return {"timestamp": stamps, "indicators": {"quote": [cols]}}


class FakeResp:
    def __init__(self, body, status_code=200):
        self._body, self.status_code = body, status_code

    def json(self):
        return self._body


def ok(rows):
    return FakeResp({"chart": {"result": [result(rows)], "error": None}})


def test_to_frame_converts_tz_drops_nan_rounds_and_filters_session():
    rows = [("2026-10-02 08:55", 1, 1, 1, 1, 1),                      # 盤前:丟掉
            ("2026-10-02 09:00", 38.750000001, 40.0, 38.650002, 39.349998, 0),
            ("2026-10-02 09:05", None, None, None, None, None),        # 空值 bar:丟掉
            ("2026-10-02 13:25", 39.0, 39.1, 38.9, 39.0, 10),
            ("2026-10-02 13:30", 39.0, 39.0, 39.0, 39.0, 5)]           # 13:30 起:丟掉
    df = yahoo.to_frame(result(rows))
    assert list(df.index) == [T("2026-10-02 09:00"), T("2026-10-02 13:25")]
    assert df.index.tz is None
    assert df.iloc[0]["open"] == 38.75 and df.iloc[0]["close"] == 39.35     # 四捨五入去浮點雜訊
    assert list(df.columns) == yahoo.COLS


def test_to_frame_empty_result():
    assert yahoo.to_frame({"timestamp": None}).empty


def test_ticker_suffix():
    assert yahoo.ticker("2409") == "2409.TW" and yahoo.ticker("6488", "otc") == "6488.TWO"


def test_fetch_merges_into_cache_and_new_data_wins(monkeypatch, tmp_path):
    monkeypatch.setattr(yahoo, "CACHE_DIR", tmp_path)
    first = ok([("2026-10-01 09:00", 10, 11, 9, 10.5, 100), ("2026-10-01 09:05", 10.5, 11, 10, 10.8, 50)])
    second = ok([("2026-10-01 09:05", 10.5, 12, 10, 11.9, 77),          # 同一時間戳:以新的為準
                 ("2026-10-02 09:00", 12, 13, 11, 12.5, 40)])
    seq = iter([first, second])
    monkeypatch.setattr(yahoo.requests, "get", lambda *a, **k: next(seq))
    now = T("2026-10-05 14:00")
    yahoo.fetch_intraday("2409", "5m", now=now)
    df = yahoo.fetch_intraday("2409", "5m", now=now)
    assert len(df) == 3                                                  # 舊資料保留 + 新資料,持續累積
    assert df.loc[T("2026-10-01 09:05"), "close"] == 11.9
    assert (tmp_path / "yahoo" / "5m" / "2409_tse.parquet").exists()
    assert list(tmp_path.rglob("*.tmp")) == []


def test_complete_days_only_excludes_today_until_market_done(monkeypatch, tmp_path):
    monkeypatch.setattr(yahoo, "CACHE_DIR", tmp_path)
    rows = [("2026-10-02 09:00", 1, 1, 1, 1, 1), ("2026-10-05 09:00", 2, 2, 2, 2, 1)]
    monkeypatch.setattr(yahoo.requests, "get", lambda *a, **k: ok(rows))
    during = yahoo.fetch_intraday("2409", "5m", now=T("2026-10-05 11:00"))
    assert [d.day for d in sorted(set(during.index.date))] == [2]          # 盤中:不含今天
    after = yahoo.fetch_intraday("2409", "5m", now=T("2026-10-05 13:40"))
    assert [d.day for d in sorted(set(after.index.date))] == [2, 5]        # 收盤後:含今天
    incl = yahoo.fetch_intraday("2409", "5m", complete_days_only=False, now=T("2026-10-05 11:00"))
    assert len(incl) == 2


def test_retry_on_429_then_success_and_no_retry_on_404(monkeypatch, tmp_path):
    monkeypatch.setattr(yahoo, "CACHE_DIR", tmp_path)
    sleeps = []
    seq = iter([FakeResp({}, 429), ok([("2026-10-02 09:00", 1, 1, 1, 1, 1)])])
    monkeypatch.setattr(yahoo.requests, "get", lambda *a, **k: next(seq))
    df = yahoo.fetch_intraday("2409", "5m", now=T("2026-10-06 00:00"), sleep=sleeps.append)
    assert len(df) == 1 and sleeps == [2.0]

    calls = []

    def not_found(*a, **k):
        calls.append(1)
        return FakeResp({"chart": {"result": None, "error": {"description": "No data found"}}}, 404)

    monkeypatch.setattr(yahoo.requests, "get", not_found)
    with pytest.raises(yahoo.YahooError) as e:
        yahoo.fetch_intraday("9999", "5m", sleep=sleeps.append)
    assert "No data found" in str(e.value) and len(calls) == 1


def test_network_error_gives_up_after_retries(monkeypatch, tmp_path):
    monkeypatch.setattr(yahoo, "CACHE_DIR", tmp_path)
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise yahoo.requests.ConnectionError("x")

    monkeypatch.setattr(yahoo.requests, "get", boom)
    with pytest.raises(yahoo.YahooError):
        yahoo.fetch_intraday("2409", "5m", sleep=lambda s: None)
    assert len(calls) == 4


def test_invalid_interval_rejected():
    with pytest.raises(ValueError):
        yahoo.fetch_intraday("2409", "2m")


def test_backtest_runs_on_yahoo_shaped_frame(monkeypatch, tmp_path):
    """5 分鐘 K(含缺 bar)餵給既有回測引擎:不崩潰、時間守規則。"""
    from run_backtest import backtest
    from twdt.backtest.costs import CostModel
    from twdt.risk.rules import RiskConfig
    idx = pd.date_range("2026-10-02 09:00", "2026-10-02 13:20", freq="5min")
    closes = [100.0] * 4 + [101.5] * 8 + [103.5] * (len(idx) - 12)
    df = pd.DataFrame({"open": closes, "high": [c + 0.5 for c in closes],
                       "low": [c - 0.2 for c in closes], "close": closes, "volume": 1.0}, index=idx)
    df = df.drop(df.index[[20, 21, 40]])                                  # 模擬被丟掉的空值 bar
    trades = backtest({"X": df}, CostModel(), RiskConfig(), 15)
    assert len(trades) >= 1
    assert all(t.exit_time.time() <= RiskConfig().force_close_at for t in trades)
