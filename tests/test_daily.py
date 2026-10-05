import numpy as np
import pandas as pd
import pytest
import requests

from twdt import daily
from twdt.data import yahoo

T = pd.Timestamp


def make(closes, start="2020-01-01", volume=1000.0):
    idx = pd.bdate_range(start, periods=len(closes))
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "adjclose": c, "volume": volume}, index=idx)


# ---------- clean_daily
def test_clean_drops_nan_duplicates_and_stale_zero_volume_rows():
    df = make([100, 101, 101, 101, 102])
    df.iloc[1, df.columns.get_loc("volume")] = 500
    df.iloc[2, df.columns.get_loc("volume")] = 0            # 停止交易:價格不變且量為 0 -> 丟掉
    df.iloc[3, df.columns.get_loc("volume")] = 0            # 同上
    df.iloc[4, df.columns.get_loc("close")] = np.nan        # 空值列 -> 丟掉
    out = daily.clean_daily(df)
    assert list(out["close"]) == [100.0, 101.0]
    dup = pd.concat([make([1.0, 2.0]), make([3.0], start="2020-01-02")])
    assert len(daily.clean_daily(dup)) == 2                 # 重複日期只留一筆


def test_clean_drops_nonpositive_prices_and_absurd_open_gaps_and_reports_counts():
    df = make([100, 101, 102, 103, 104, 105])
    df.iloc[2, df.columns.get_loc("open")] = 0.0              # Yahoo 偶爾給開盤價 0
    df.iloc[4, df.columns.get_loc("open")] = 130.0            # 開盤比前收盤高 25%:超過漲跌幅限制
    out = daily.clean_daily(df)
    assert list(out["close"]) == [100.0, 101.0, 103.0, 105.0]
    assert out.attrs["dropped"]["價格<=0"] == 1 and out.attrs["dropped"]["開盤價異常"] == 1
    assert np.isfinite(daily.adj_open(out).shift(-1) / daily.adj_open(out)).iloc[:-1].all()   # 不再有無限大報酬


def test_clean_keeps_zero_volume_rows_with_price_change():
    df = make([100, 101, 102], volume=0.0)                  # 指數可能沒有量;價格有變就不能丟
    assert len(daily.clean_daily(df)) == 3


def test_clean_adjusts_4_to_1_split_for_all_price_columns():
    closes = [58.0, 58.5, 58.7, 14.68, 14.6, 14.7]           # 58.7 -> 14.68 = 1/4
    out = daily.clean_daily(make(closes))
    assert out["close"].iloc[2] == pytest.approx(58.7 / 4, rel=0.01)
    assert out["adjclose"].iloc[2] == pytest.approx(out["close"].iloc[2])
    assert out["open"].iloc[0] == pytest.approx(58.0 / 4, rel=0.01)
    assert out["close"].iloc[3] == 14.68                     # 分割後不動
    assert out["close"].pct_change().abs().max() < 0.05      # 不再有 75% 的假暴跌


def test_clean_handles_reverse_split_and_multiple_splits():
    out = daily.clean_daily(make([10.0, 10.1, 40.4, 40.6]))             # 1 比 4 合併
    assert out["close"].pct_change().abs().max() < 0.05
    out2 = daily.clean_daily(make([80.0, 80.4, 40.2, 40.0, 13.4, 13.5]))  # 先 1/2 再 1/3
    assert out2["close"].pct_change().abs().max() < 0.05


def test_clean_raises_on_unexplained_jump():
    with pytest.raises(ValueError, match="疑似資料錯誤"):
        daily.clean_daily(make([100.0, 101.0, 47.0, 47.5]))             # -53%:不是常見分割比例


# ---------- 指標與訊號
def test_signals_use_no_future_data():
    rng = np.random.default_rng(0)
    closes = list(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))))
    base = make(closes)
    for fn in (daily.sig_sma_filter, daily.sig_dual_ma, daily.sig_momentum, daily.sig_rsi2):
        full = fn(base)
        cut = fn(base.iloc[:300])
        pd.testing.assert_series_equal(full.iloc[:300], cut)             # 前 300 天的訊號不受後面資料影響
        altered = base.copy()
        altered.iloc[300:, :5] = altered.iloc[300:, :5] * 3              # 改掉未來價格
        pd.testing.assert_series_equal(fn(altered).iloc[:300], full.iloc[:300])


def test_sma_filter_and_warmup_are_flat():
    closes = [100.0] * 200 + [110.0] * 5
    s = daily.sig_sma_filter(make(closes), 200)
    assert (s.iloc[:199] == 0).all()                         # 暖機期不持有
    assert s.iloc[-1] == 1.0


def test_momentum_signal_positive_and_negative():
    up = daily.sig_momentum(make(list(np.linspace(100, 200, 300))))
    dn = daily.sig_momentum(make(list(np.linspace(200, 100, 300))))
    assert up.iloc[-1] == 1.0 and dn.iloc[-1] == 0.0
    assert (up.iloc[:252] == 0).all()


def test_rsi_extremes_and_rsi2_state_machine():
    assert daily.rsi(pd.Series(np.linspace(1, 50, 50)), 2).iloc[-1] == pytest.approx(100.0)
    assert daily.rsi(pd.Series(np.linspace(50, 1, 50)), 2).iloc[-1] == pytest.approx(0.0, abs=1e-6)
    # 長期上升趨勢中突然連跌 -> RSI(2) 很低 -> 買進;反彈站上 5 日均線 -> 賣出
    closes = list(np.linspace(100, 160, 260)) + [155, 150, 146, 150, 156, 162]
    s = daily.sig_rsi2(make(closes))
    assert s.iloc[259] == 0.0                                 # 還在上漲,RSI 高,不買
    assert s.iloc[260] == 1.0 and s.iloc[263] == 1.0          # 第一天大跌 RSI(2) 很低 -> 買進並持有
    assert s.iloc[-1] == 0.0                                  # 反彈站上 SMA5 後出場


# ---------- 回測
def test_buy_hold_matches_compounding_minus_entry_cost_only():
    closes = [100.0] * 5 + [110.0] * 5
    df = make(closes)
    cost = daily.DailyCost(fee_rate=0.001, tax_rate=0.002, slippage=0.0)
    res = daily.backtest_daily(df, daily.sig_buy_hold(df), cost)
    # 第一天還沒有「前一天的訊號」,所以第 2 天開盤才買進;之後不賣,不扣賣出成本與稅
    o = daily.adj_open(df)
    gross = o.iloc[-1] / o.iloc[1]
    assert res["final"] == pytest.approx(gross * (1 - 0.001), rel=1e-9)
    assert res["entries"] == 1 and res["in_market"] == pytest.approx(8 / 9)


def test_signal_is_executed_next_open_not_same_day():
    # 第 3 天收盤出現訊號,隔天(第 4 天)開盤才進場:報酬應從第 4 天開盤算起
    idx = pd.bdate_range("2020-01-01", periods=8)
    opens = pd.Series([100, 100, 100, 100, 120, 120, 120, 120], index=idx, dtype=float)
    df = pd.DataFrame({"open": opens, "high": opens, "low": opens, "close": opens, "adjclose": opens,
                       "volume": 1.0})
    sig = pd.Series(0.0, index=idx)
    sig.iloc[2:] = 1.0                                        # 第 3 天(index 2)收盤才知道
    res = daily.backtest_daily(df, sig, daily.DailyCost(0, 0, 0))
    # 第 4 天(index 3)開盤進場,持有到 index 4 開盤時價格 100 -> 120:應吃到這段 +20%
    assert res["final"] == pytest.approx(1.2)
    # 反例:若是「當天就成交」(偷看),進場在 index 2 開盤,同樣吃到 +20%;
    # 所以另外驗證:訊號從 index 3 開始時,不能吃到 index3->4 的漲幅
    sig2 = pd.Series(0.0, index=idx)
    sig2.iloc[3:] = 1.0
    res2 = daily.backtest_daily(df, sig2, daily.DailyCost(0, 0, 0))
    assert res2["final"] == pytest.approx(1.0)                # 隔天(index 4)開盤進場,錯過跳空


def test_round_trip_costs_buy_then_sell_with_tax():
    idx = pd.bdate_range("2020-01-01", periods=6)
    o = pd.Series(100.0, index=idx)
    df = pd.DataFrame({"open": o, "high": o, "low": o, "close": o, "adjclose": o, "volume": 1.0})
    sig = pd.Series([1.0, 1.0, 0.0, 0.0, 0.0, 0.0], index=idx)       # 持有後賣出,價格沒變
    cost = daily.DailyCost(fee_rate=0.001, tax_rate=0.002, slippage=0.0005)
    res = daily.backtest_daily(df, sig, cost)
    expected = (1 - (0.001 + 0.0005)) * (1 - (0.001 + 0.002 + 0.0005))
    assert res["final"] == pytest.approx(expected) and res["entries"] == 1


def test_segment_starts_flat_and_pays_entry_cost():
    df = make([100.0] * 30)
    sig = pd.Series(1.0, index=df.index)
    cost = daily.DailyCost(fee_rate=0.001, tax_rate=0.0, slippage=0.0)
    a = daily.backtest_daily(df, sig, cost, start="2020-01-01", end="2020-01-15")
    b = daily.backtest_daily(df, sig, cost, start="2020-01-16", end="2020-02-10")
    assert a["entries"] == 1 and b["entries"] == 1            # 每個區段都從空手開始
    assert b["final"] == pytest.approx(1 - 0.001)


def test_metrics_max_drawdown_and_empty_segment():
    closes = [100.0, 100.0, 120.0, 120.0, 60.0, 60.0, 60.0]
    df = make(closes)
    res = daily.backtest_daily(df, daily.sig_buy_hold(df), daily.DailyCost(0, 0, 0))
    assert res["max_dd"] == pytest.approx(-0.5)
    with pytest.raises(ValueError):
        daily.backtest_daily(df, daily.sig_buy_hold(df), daily.DailyCost(), start="2030-01-01")


# ---------- 抓取
class FakeResp:
    def __init__(self, body, status_code=200):
        self._body, self.status_code = body, status_code

    def json(self):
        return self._body


def test_fetch_daily_uses_period_params_not_range_max_and_cleans(monkeypatch, tmp_path):
    monkeypatch.setattr(daily, "CACHE_DIR", tmp_path)
    seen = {}
    stamps = [int(T(f"2020-01-0{d}", tz="Asia/Taipei").tz_convert("UTC").timestamp()) for d in (2, 3, 6, 7)]
    q = {"open": [58, 58, 14.6, 14.7], "high": [58, 58, 14.6, 14.7], "low": [58, 58, 14.6, 14.7],
         "close": [58.0, 58.4, 14.6, 14.7], "volume": [1, 1, 1, 1]}
    body = {"chart": {"error": None, "result": [{"timestamp": stamps, "indicators": {
        "quote": [q], "adjclose": [{"adjclose": q["close"]}]}}]}}

    def fake_get(url, params, headers, timeout):
        seen.update(params)
        return FakeResp(body)

    monkeypatch.setattr(daily.requests, "get", fake_get)
    df = daily.fetch_daily("0050", now=1_700_000_000)
    assert "range" not in seen and seen["period1"] < seen["period2"] and seen["interval"] == "1d"
    assert df["close"].pct_change().abs().max() < 0.05          # 4 比 1 分割已調整
    assert (tmp_path / "daily" / "0050_tse.parquet").exists()


def test_fetch_daily_errors_hide_details(monkeypatch, tmp_path):
    monkeypatch.setattr(daily, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(daily.requests, "get", lambda *a, **k: FakeResp({"chart": {"result": None, "error": {"x": 1}}}, 404))
    with pytest.raises(yahoo.YahooError):
        daily.fetch_daily("9999")

    def boom(*a, **k):
        raise requests.ConnectionError("https://secret")

    monkeypatch.setattr(daily.requests, "get", boom)
    with pytest.raises(yahoo.YahooError) as e:
        daily.fetch_daily("9999")
    assert "secret" not in str(e.value)
