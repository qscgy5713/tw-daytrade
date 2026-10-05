import pandas as pd
import pytest

from twdt.backtest.engine import Trade
from twdt.data import finmind
from twdt.report.metrics import summarize


class FakeResp:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


def test_fetch_daily_parses_and_caches(monkeypatch, tmp_path):
    monkeypatch.setattr(finmind, "CACHE_DIR", tmp_path)
    calls = []

    def fake_get(url, params, timeout):
        calls.append(params)
        return FakeResp({"status": 200, "data": [
            {"date": "2026-10-05", "open": 100, "max": 105, "min": 99, "close": 104,
             "Trading_Volume": 5000}]})

    monkeypatch.setattr(finmind.requests, "get", fake_get)
    df = finmind.fetch_daily("2330", "2026-10-05", "2026-10-05")
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df["high"].iloc[0] == 105
    finmind.fetch_daily("2330", "2026-10-05", "2026-10-05")
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
