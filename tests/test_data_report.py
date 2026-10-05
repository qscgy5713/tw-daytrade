import pandas as pd
import pytest

from twdt.backtest.engine import Trade
from twdt.data import finmind
from twdt.report.metrics import summarize


class FakeResp:
    def __init__(self, body, status_code=200):
        self._body = body
        self.status_code = status_code

    def json(self):
        return self._body


def test_fetch_daily_parses_and_caches(monkeypatch, tmp_path):
    monkeypatch.setattr(finmind, "CACHE_DIR", tmp_path)
    calls = []

    def fake_get(url, params, timeout):
        calls.append(params)
        return FakeResp({"status": 200, "data": [
            {"date": "2026-09-01", "open": 100, "max": 105, "min": 99, "close": 104,
             "Trading_Volume": 5000}]})

    monkeypatch.setattr(finmind.requests, "get", fake_get)
    df = finmind.fetch_daily("2330", "2026-09-01", "2026-09-01")
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df["high"].iloc[0] == 105
    finmind.fetch_daily("2330", "2026-09-01", "2026-09-01")
    assert len(calls) == 1  # 第二次走快取


def test_fetch_minute_requires_token(monkeypatch):
    monkeypatch.delenv("FINMIND_TOKEN", raising=False)
    with pytest.raises(finmind.FinMindError):
        finmind.fetch_minute("2330", "2026-10-05", "2026-10-05")


def test_fetch_minute_parses(monkeypatch, tmp_path):
    monkeypatch.setattr(finmind, "CACHE_DIR", tmp_path)

    def fake_get(url, params, timeout):
        if params["start_date"] != "2026-10-05":
            return FakeResp({"status": 200, "data": []})
        return FakeResp({"status": 200, "data": [
            {"date": "2026-10-05", "minute": "09:00:00", "open": 100, "high": 101,
             "low": 99, "close": 100.5, "volume": 10},
            {"date": "2026-10-05", "minute": "09:01:00", "open": 100.5, "high": 102,
             "low": 100, "close": 101, "volume": 12}]})

    monkeypatch.setattr(finmind.requests, "get", fake_get)
    df = finmind.fetch_minute("2330", "2026-10-05", "2026-10-06", token="t")
    assert len(df) == 2 and df.index[0] == pd.Timestamp("2026-10-05 09:00")
    assert list(finmind.split_days(df)) == [pd.Timestamp("2026-10-05").date()]


def test_api_error_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(finmind, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(finmind.requests, "get",
                        lambda *a, **k: FakeResp({"status": 402, "msg": "limit"}))
    with pytest.raises(finmind.FinMindError):
        finmind.fetch_daily("2330", "2026-10-05", "2026-10-05")


def _trade(net_gross, cost, day):
    ts = pd.Timestamp(f"2026-10-{day:02d} 09:30")
    return Trade("X", 1, ts, ts, 100, 100, 1000, net_gross, cost, "t")


def test_summarize():
    trades = [_trade(500, 100, 5), _trade(-300, 100, 5), _trade(200, 100, 6)]
    m = summarize(trades)
    assert m["trades"] == 3 and m["days"] == 2
    assert m["net_pnl"] == pytest.approx(100)
    assert m["win_rate"] == pytest.approx(2 / 3)
    assert m["max_drawdown"] == pytest.approx(-400)
    assert m["cost_share_of_gross"] == pytest.approx(300 / 400)
    assert summarize([]) == {"trades": 0}


def test_drawdown_counts_first_loss():
    assert summarize([_trade(-300, 0, 5), _trade(100, 0, 6)])["max_drawdown"] == pytest.approx(-300)


def test_http_400_reports_msg_without_leaking_token(monkeypatch, tmp_path):
    monkeypatch.setattr(finmind, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(finmind.requests, "get", lambda *a, **k: FakeResp(
        {"status": 400, "msg": "token invalid"}, status_code=400))
    with pytest.raises(finmind.FinMindError) as e:
        finmind.fetch_minute("2330", "2026-10-05", "2026-10-05", token="SECRET123")
    assert "token invalid" in str(e.value) and "400" in str(e.value)
    assert "SECRET123" not in str(e.value)


def test_empty_and_today_results_not_cached(monkeypatch, tmp_path):
    monkeypatch.setattr(finmind, "CACHE_DIR", tmp_path)
    calls = []

    def empty_get(url, params, timeout):
        calls.append(1)
        return FakeResp({"status": 200, "data": []})

    monkeypatch.setattr(finmind.requests, "get", empty_get)
    finmind.fetch_daily("2330", "2026-09-01", "2026-09-01")
    finmind.fetch_daily("2330", "2026-09-01", "2026-09-01")
    assert len(calls) == 2  # 空結果不快取,第二次仍會重抓

    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    monkeypatch.setattr(finmind.requests, "get", lambda *a, **k: FakeResp(
        {"status": 200, "data": [{"date": today, "open": 1, "max": 1, "min": 1, "close": 1,
                                  "Trading_Volume": 1}]}))
    assert len(finmind.fetch_daily("2330", today, today)) == 1
    assert list(tmp_path.rglob("*.parquet")) == []  # 區間含今天,不快取


def _kbar_resp(day):
    return FakeResp({"status": 200, "data": [
        {"date": day, "minute": "09:00:00", "open": 1, "high": 1, "low": 1, "close": 1,
         "volume": 1}]})


def test_minute_retries_transient_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(finmind, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(finmind.time, "sleep", lambda s: None)
    seq = iter([FakeResp({"status": 500, "msg": "busy"}, 500), _kbar_resp("2026-09-01")])
    monkeypatch.setattr(finmind.requests, "get", lambda *a, **k: next(seq))
    df = finmind.fetch_minute("2330", "2026-09-01", "2026-09-01", token="t")
    assert len(df) == 1


def test_minute_does_not_retry_permission_error(monkeypatch, tmp_path):
    monkeypatch.setattr(finmind, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(finmind.time, "sleep", lambda s: None)
    calls = []

    def get(*a, **k):
        calls.append(1)
        return FakeResp({"status": 400, "msg": "Your level is register"}, 400)

    monkeypatch.setattr(finmind.requests, "get", get)
    with pytest.raises(finmind.FinMindError):
        finmind.fetch_minute("2330", "2026-09-01", "2026-09-01", token="t")
    assert len(calls) == 1


def test_minute_network_error_hides_token_and_gives_up(monkeypatch, tmp_path):
    monkeypatch.setattr(finmind, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(finmind.time, "sleep", lambda s: None)
    calls = []

    def get(*a, **k):
        calls.append(1)
        raise finmind.requests.ConnectionError("https://x?token=SECRET123")

    monkeypatch.setattr(finmind.requests, "get", get)
    with pytest.raises(finmind.FinMindError) as e:
        finmind.fetch_minute("2330", "2026-09-01", "2026-09-01", token="SECRET123")
    assert "SECRET123" not in str(e.value) and e.value.__cause__ is None
    assert len(calls) == 4  # 首次 + 3 次重試


def test_minute_partial_failure_keeps_finished_days(monkeypatch, tmp_path):
    monkeypatch.setattr(finmind, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(finmind.time, "sleep", lambda s: None)

    def get(url, params, timeout):
        if params["start_date"] == "2026-09-01":
            return _kbar_resp("2026-09-01")
        return FakeResp({"status": 400, "msg": "boom"}, 400)

    monkeypatch.setattr(finmind.requests, "get", get)
    with pytest.raises(finmind.FinMindError):
        finmind.fetch_minute("2330", "2026-09-01", "2026-09-02", token="t")
    assert (tmp_path / "minute" / "2330_2026-09-01.parquet").exists()

    calls = []

    def get2(url, params, timeout):
        calls.append(params["start_date"])
        return _kbar_resp(params["start_date"])

    monkeypatch.setattr(finmind.requests, "get", get2)
    df = finmind.fetch_minute("2330", "2026-09-01", "2026-09-02", token="t")
    assert calls == ["2026-09-02"] and len(df) == 2  # 只補缺的那天
