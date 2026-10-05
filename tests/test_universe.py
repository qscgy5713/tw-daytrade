import pandas as pd
import pytest

import compare_universe as cu
from compare_daily import STRATEGIES


def fake_result(bh, strat_cagr, strat_sharpe=0.5, bh_sharpe=0.4, strat_dd=-0.2, bh_dd=-0.3):
    rows = []
    for name in STRATEGIES:
        for seg in ("開發段", "驗證段"):
            is_bh = name == cu.BASELINE
            rows.append({"strategy": name, "seg": seg, "cagr": bh if is_bh else strat_cagr,
                         "sharpe": bh_sharpe if is_bh else strat_sharpe,
                         "max_dd": bh_dd if is_bh else strat_dd, "entries_per_year": 0.0 if is_bh else 3.0,
                         "in_market": 1.0 if is_bh else 0.8})
    return pd.DataFrame(rows)


def test_aggregate_differences_and_win_rates():
    res = {"A": fake_result(0.10, 0.08),                      # 年化落後 2pp,夏普贏,回撤較淺
           "B": fake_result(0.10, 0.12, strat_sharpe=0.3)}    # 年化贏 2pp,夏普輸
    agg = cu.aggregate(res)
    row = agg[(agg["策略"] == "站上200日線才持有") & (agg["區段"] == "開發段")].iloc[0]
    assert row["標的數"] == 2
    assert row["年化差(平均)"] == pytest.approx(0.0) and row["年化差(中位)"] == pytest.approx(0.0)
    assert row["夏普勝過買進持有"] == 0.5 and row["年化勝過買進持有"] == 0.5
    assert row["回撤改善(平均)"] == pytest.approx(0.10)       # -0.2 相對 -0.3:淺 10 個百分點
    assert row["每年進場"] == 3.0 and row["持有時間比"] == pytest.approx(0.8)
    assert cu.BASELINE not in set(agg["策略"])                # 基準不跟自己比


def test_aggregate_single_asset_and_all_segments_present():
    agg = cu.aggregate({"A": fake_result(0.05, 0.05)})
    assert set(agg["區段"]) == {"開發段", "驗證段"} and len(agg) == (len(STRATEGIES) - 1) * 2
    assert (agg["年化差(平均)"] == 0).all()


def test_evaluate_asset_returns_all_strategies_and_segments():
    import numpy as np
    idx = pd.bdate_range("2008-01-01", "2024-12-31")
    rng = np.random.default_rng(1)
    c = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(idx)))), index=idx)
    df = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "adjclose": c, "volume": 1.0}, index=idx)
    out = cu.evaluate_asset(df, tax=0.001)
    assert len(out) == len(STRATEGIES) * 2
    assert set(out["seg"]) == {"開發段", "驗證段"}
    assert out[(out.strategy == cu.BASELINE)]["entries_per_year"].max() < 0.2   # 買進持有每段只進場一次
