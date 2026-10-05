import numpy as np
import pandas as pd
import pytest

from twdt import etf_alloc as ea


def idx(*days):
    return pd.DatetimeIndex([pd.Timestamp(d) for d in days])


# ---------- weights_grid
def test_weights_grid_properties():
    W = ea.weights_grid(3, 10)
    assert W.shape == (66, 3)                                  # C(12,2)
    assert np.allclose(W.sum(axis=1), 1.0)
    assert np.allclose(W * 10, np.round(W * 10))               # 全是 10% 的倍數
    assert len({tuple(r) for r in W}) == len(W)                # 不重複
    assert [1.0, 0.0, 0.0] in W.tolist() and [0.5, 0.3, 0.2] in W.tolist()
    assert ea.weights_grid(7, 10).shape == (8008, 7)           # C(16,6)
    assert ea.weights_grid(1, 10).tolist() == [[1.0]]


# ---------- tr_metrics
def test_tr_metrics_known_drawdown_and_return():
    days = pd.bdate_range("2024-01-01", periods=4)
    adj = pd.DataFrame({"A": [100, 110, 55, 60.0], "B": [100, 100, 100, 100.0]}, index=days)
    out = ea.tr_metrics(adj, np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]))
    assert out["max_dd"][0] == pytest.approx(-0.5)             # 110 -> 55
    assert out["max_dd"][1] == pytest.approx(0.0) and out["vol"][1] == pytest.approx(0.0)
    assert out["max_dd"][2] > out["max_dd"][0]                 # 混一半現金式資產,回檔較小


def test_tr_metrics_rebalanced_return_is_weighted_daily_return():
    days = pd.bdate_range("2024-01-01", periods=3)
    adj = pd.DataFrame({"A": [100, 110, 121.0], "B": [100, 100, 100.0]}, index=days)
    out = ea.tr_metrics(adj, np.array([[0.5, 0.5]]))
    # 每天 A +10%、B 0%,50/50 每日再平衡 -> 每天 +5%,兩天 1.05^2
    years = (days[-1] - days[0]).days / 365.25
    assert out["ann"][0] == pytest.approx((1.05 ** 2) ** (1 / years) - 1)


# ---------- dca_simulate:配息入帳在「買進前」
def test_dca_dividend_uses_shares_held_before_same_day_purchase():
    days = idx("2024-01-02", "2024-01-15", "2024-02-01", "2024-02-15")
    close = pd.DataFrame({"X": [10.0, 10.0, 10.0, 10.0]}, index=days)
    div = pd.DataFrame({"X": [0.0, 0.5, 0.5, 0.0]}, index=days)   # 1/15 與 2/1 除息
    sim = ea.dca_simulate(close, div, np.array([[1.0]]), contrib=10000)
    # 1/2 買 1000 股;1/15 配息 1000*0.5=500;2/1 先配息 1000*0.5=500(用買進前持股),再買 1000 股
    assert sim["monthly_div"][0, 0] == pytest.approx(500)
    assert sim["monthly_div"][1, 0] == pytest.approx(500)
    assert sim["invested"] == 20000 and sim["invested_m"].tolist() == [10000, 20000]
    assert sim["market_value"][0] == pytest.approx(2000 * 10)
    assert sim["total_div"][0] == pytest.approx(1000)


def test_dca_weights_split_contribution_and_dividends_scale():
    days = idx("2024-01-02", "2024-01-20")
    close = pd.DataFrame({"A": [10.0, 10.0], "B": [20.0, 20.0]}, index=days)
    div = pd.DataFrame({"A": [0.0, 1.0], "B": [0.0, 2.0]}, index=days)
    W = np.array([[0.6, 0.4]])
    sim = ea.dca_simulate(close, div, W, contrib=10000)
    # A 買 6000/10=600 股,B 買 4000/20=200 股;配息 600*1 + 200*2 = 1000
    assert sim["monthly_div"][0, 0] == pytest.approx(1000)
    assert sim["market_value"][0] == pytest.approx(600 * 10 + 200 * 20)


# ---------- 配息再投入
def test_reinvest_buys_back_same_etf_on_next_buy_and_total_return_uses_own_money_only():
    days = idx("2024-01-02", "2024-01-15", "2024-02-01", "2024-02-15")
    close = pd.DataFrame({"X": [10.0, 10.0, 10.0, 10.0]}, index=days)
    div = pd.DataFrame({"X": [0.0, 0.5, 0.0, 0.0]}, index=days)          # 只有 1/15 除息
    W = np.array([[1.0]])
    plain = ea.dca_simulate(close, div, W, contrib=10000)
    rein = ea.dca_simulate(close, div, W, contrib=10000, reinvest=True)
    # 1/2 買 1000 股;1/15 配息 500 元;不再投入:2/1 再買 1000 股,共 2000 股 + 手上 500 現金
    assert plain["market_value"][0] == pytest.approx(2000 * 10) and plain["dividends_kept"][0] == pytest.approx(500)
    # 再投入:2/1 買進時把 500 元也買回 -> 2000 + 50 股,現金 0
    assert rein["market_value"][0] == pytest.approx(2050 * 10) and rein["dividends_kept"][0] == pytest.approx(0)
    assert rein["invested"] == plain["invested"] == 20000             # 投入仍只算自己掏的錢
    assert rein["monthly_div"].tolist() == plain["monthly_div"].tolist()     # 領到的現金流不變
    # 價格不變時,兩者「市值 + 手上現金」相同(再投入只是換成股票)
    assert rein["market_value"][0] + rein["dividends_kept"][0] == pytest.approx(plain["market_value"][0] + plain["dividends_kept"][0])


def test_reinvest_compounds_when_price_rises_and_cash_is_kept_if_no_later_buy_day():
    days = idx("2024-01-02", "2024-01-15", "2024-02-01", "2024-02-20")
    close = pd.DataFrame({"X": [10.0, 10.0, 10.0, 20.0]}, index=days)     # 最後漲到 20
    div = pd.DataFrame({"X": [0.0, 0.5, 0.0, 0.5]}, index=days)           # 2/20 除息後沒有下一次買進日
    sim = ea.dca_simulate(close, div, np.array([[1.0]]), contrib=10000, reinvest=True)
    assert sim["market_value"][0] == pytest.approx(2050 * 20)
    assert sim["dividends_kept"][0] == pytest.approx(2050 * 0.5)          # 2/20 的配息還來不及再投入
    ret = ea.income_metrics(sim_with_enough_months(sim), last_n=12)["total_return"][0]
    assert ret == pytest.approx((2050 * 20 + 1025 - 20000) / 20000)


def test_reinvest_per_etf_cash_is_not_mixed_across_assets():
    days = idx("2024-01-02", "2024-01-15", "2024-02-01")
    close = pd.DataFrame({"A": [10.0, 10.0, 10.0], "B": [20.0, 20.0, 20.0]}, index=days)
    div = pd.DataFrame({"A": [0.0, 1.0, 0.0], "B": [0.0, 0.0, 0.0]}, index=days)      # 只有 A 配息
    sim = ea.dca_simulate(close, div, np.array([[0.5, 0.5]]), contrib=10000, reinvest=True)
    # A 1/2 買 500 股,配息 500 元;2/1 買進:A 再 +500 股 +50 股(配息只買回 A),B 買 2 次 = 500 股
    assert sim["market_value"][0] == pytest.approx(1050 * 10 + 500 * 20)


def sim_with_enough_months(sim):
    """income_metrics 需要 >= 6 個完整月份;補足假資料只為了取 total_return。"""
    pad = 6
    return {**sim, "monthly_div": np.vstack([np.zeros((pad, sim["monthly_div"].shape[1])), sim["monthly_div"]]),
            "invested_m": np.concatenate([np.full(pad, 1000.0), sim["invested_m"]]),
            "complete": np.concatenate([np.ones(pad, dtype=bool), sim["complete"]])}


# ---------- income_metrics
def sim_of(monthly_div, invested, complete=None, mv=None, total_div=None, inv_total=None):
    md = np.array(monthly_div, dtype=float)
    return {"monthly_div": md, "invested_m": np.array(invested, dtype=float),
            "complete": np.array(complete if complete is not None else [True] * len(md)),
            "market_value": np.array(mv if mv is not None else [1000.0] * md.shape[1]),
            "total_div": md.sum(axis=0) if total_div is None else np.array(total_div),
            "invested": inv_total if inv_total is not None else float(invested[-1])}


def test_income_constant_yield_has_zero_cv_and_expected_levels():
    n = 8
    inv = [1000.0 * (i + 1) for i in range(n)]
    md = [[10.0 * (i + 1)] for i in range(n)]                   # 每月都是累積投入的 1%
    m = ea.income_metrics(sim_of(md, inv, mv=[8000.0]), last_n=8)
    assert m["yoc_cv"][0] == pytest.approx(0.0, abs=1e-9)
    assert m["yoc_min_month"][0] == pytest.approx(0.03, rel=0.05)        # 3 個月約 3%
    assert m["yoc_annual"][0] == pytest.approx(sum(x[0] for x in md) / np.mean(inv) * (12 / n))


def test_income_quarterly_payer_is_not_penalised_but_erratic_payer_is():
    n = 12
    inv = [1000.0] * n
    quarterly = [30.0 if i % 3 == 0 else 0.0 for i in range(n)]           # 每季一次,每月其實等價
    erratic = [90.0 if i in (2, 3, 4, 9) else 0.0 for i in range(n)]
    drift = [10, 10, 0, 20, 10, 10, 10, 10, 0, 20, 10, 10]                # 除息日在月底月初飄移:斷一個月、下個月兩次
    md = np.array([quarterly, erratic, drift], dtype=float).T
    m = ea.income_metrics(sim_of(md, inv, mv=[1000.0] * 3), last_n=12)
    assert m["yoc_cv"][0] == pytest.approx(0.0, abs=1e-9)                 # 滾動 3 個月對季配不敏感
    assert m["yoc_cv"][1] > 0.5
    # 除息日飄移(斷一個月、下個月兩次):單月看 CV 約 0.65,滾動 3 個月只剩約 0.22,
    # 明顯改善但不會完全消除,所以只斷言「大幅小於單月看」
    raw_monthly_cv = md[:, 2].std(ddof=1) / md[:, 2].mean()
    assert m["yoc_cv"][2] < 0.3 and m["yoc_cv"][2] < raw_monthly_cv / 2


def test_income_ignores_incomplete_last_month_and_zero_dividend_cv_is_nan():
    n = 8
    inv = [1000.0] * n
    md = np.array([[10.0] * (n - 1) + [0.0], [0.0] * n]).T               # 第一檔最後一個月沒配息(月份未走完)
    m_incomplete = ea.income_metrics(sim_of(md, inv, complete=[True] * (n - 1) + [False], mv=[1000.0, 1000.0]), last_n=12)
    m_complete = ea.income_metrics(sim_of(md, inv, complete=[True] * n, mv=[1000.0, 1000.0]), last_n=12)
    assert m_incomplete["yoc_cv"][0] == pytest.approx(0.0, abs=1e-9)      # 排除未走完的月份後,配息是固定的
    assert m_complete["yoc_cv"][0] > 0.1                                  # 若誤把它算進去,會被灌水
    assert np.isnan(m_incomplete["yoc_cv"][1]) and m_incomplete["yoc_annual"][1] == 0.0


def test_income_needs_at_least_six_complete_months():
    with pytest.raises(ValueError):
        ea.income_metrics(sim_of([[1.0]] * 5, [1000.0] * 5), last_n=12)


def test_dca_marks_last_month_incomplete_unless_it_ends_on_month_end():
    mid = pd.bdate_range("2024-01-02", "2024-03-05")
    c = pd.DataFrame({"X": 10.0}, index=mid)
    d = pd.DataFrame({"X": 0.0}, index=mid)
    assert ea.dca_simulate(c, d, np.array([[1.0]]))["complete"].tolist() == [True, True, False]   # 3 月只有 3 天
    full = pd.bdate_range("2024-01-02", "2024-03-29")
    c2, d2 = pd.DataFrame({"X": 10.0}, index=full), pd.DataFrame({"X": 0.0}, index=full)
    assert ea.dca_simulate(c2, d2, np.array([[1.0]]))["complete"].all()


# ---------- align
def test_align_uses_common_days_drops_pre_window_dividends_and_shifts_nontrading_exdates():
    a = pd.DataFrame({"close": [10.0, 11, 12, 13, 14], "adjclose": [10.0, 11, 12, 13, 14]},
                     index=idx("2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"))
    b = pd.DataFrame({"close": [20.0, 21, 22, 23], "adjclose": [20.0, 21, 22, 23]},
                     index=idx("2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"))
    # 1/2 的配息在共同期間(從 1/3 起)之前 -> 丟掉;1/4 照算;1/6 是週六 -> 順延到 1/8;同日多筆加總
    da = pd.Series([0.1, 0.2, 0.3, 0.05], index=idx("2024-01-02", "2024-01-04", "2024-01-06", "2024-01-08"))
    close, adj, div = ea.align({"A": (a, da), "B": (b, pd.Series(dtype=float))})
    assert list(close.index) == list(idx("2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"))
    assert div["A"].tolist() == pytest.approx([0.0, 0.2, 0.0, 0.35])
    assert div["B"].sum() == 0 and not close.isna().any().any()
    _, _, d2 = ea.align({"A": (a, da), "B": (b, pd.Series(dtype=float))}, start="2024-01-05")
    assert list(d2.index) == list(idx("2024-01-05", "2024-01-08")) and d2["A"].tolist() == pytest.approx([0.0, 0.35])


# ---------- fetch
class FakeResp:
    def __init__(self, body, status_code=200):
        self._b, self.status_code = body, status_code

    def json(self):
        return self._b


def test_fetch_etf_parses_prices_and_dividends(monkeypatch):
    t = lambda s: int(pd.Timestamp(s, tz="Asia/Taipei").tz_convert("UTC").timestamp())
    body = {"chart": {"result": [{"timestamp": [t("2024-01-02"), t("2024-01-03"), t("2024-01-04")],
                                  "indicators": {"quote": [{"close": [10.0, None, 12.0]}],
                                                 "adjclose": [{"adjclose": [9.0, None, 11.5]}]},
                                  "events": {"dividends": {"x": {"date": t("2024-01-04"), "amount": 0.1}}}}]}}
    monkeypatch.setattr(ea.requests, "get", lambda *a, **k: FakeResp(body))
    px, divs = ea.fetch_etf("00001", now=1_800_000_000)
    assert list(px.columns) == ["close", "adjclose"] and len(px) == 2          # 空值列被丟掉
    assert divs.iloc[0] == 0.1 and divs.index[0] == pd.Timestamp("2024-01-04")
    monkeypatch.setattr(ea.requests, "get", lambda *a, **k: FakeResp({}, 404))
    with pytest.raises(RuntimeError):
        ea.fetch_etf("99999")


def test_fetch_etf_dividend_date_is_taiwan_local_date_even_at_midnight_taipei(monkeypatch):
    t = lambda s: int(pd.Timestamp(s, tz="Asia/Taipei").tz_convert("UTC").timestamp())
    body = {"chart": {"result": [{"timestamp": [t("2024-01-03"), t("2024-01-04")],
                                  "indicators": {"quote": [{"close": [10.0, 11.0]}], "adjclose": [{"adjclose": [10.0, 11.0]}]},
                                  "events": {"dividends": {"a": {"date": t("2024-01-04"), "amount": 0.1},      # 台灣 00:00
                                                           "b": {"date": t("2024-01-03") + 9 * 3600, "amount": 0.2}}}}]}}  # 台灣 09:00
    monkeypatch.setattr(ea.requests, "get", lambda *a, **k: FakeResp(body))
    _, divs = ea.fetch_etf("00001", now=1_800_000_000)
    assert list(divs.index) == [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")]
