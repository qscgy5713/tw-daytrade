import sys

import pandas as pd
import pytest

import paper_trade
import record_minutes
from twdt import logutil
from twdt.data import realtime
from twdt.data import stats as stats_mod
from twdt.data.stats import QuoteStats
from twdt.paper import journal

T = pd.Timestamp


def q(sym, t, price=100.0, vol=1000):
    return {"symbol": sym, "ts": T(f"2026-10-05 {t}"), "price": price, "cum_volume": vol}


def test_unchanged_snapshots_count_as_polls_not_updates():
    s = QuoteStats()
    s.observe(q("2330", "09:00:05", vol=1000))
    s.observe(q("2330", "09:00:05", vol=1000))   # 同一個快照被輪詢到第二次
    s.observe(q("2330", "09:00:05", vol=1000))
    s.observe(q("2330", "09:00:35", vol=1010))
    r = s.summary("2330")
    assert r["polls"] == 4 and r["updates"] == 2
    assert r["unchanged_poll_ratio"] == pytest.approx(0.5)


def test_coverage_gaps_and_per_minute():
    s = QuoteStats()
    # 09:00、09:01、(09:02 無)、09:03 有更新;共 4 分鐘跨度、3 分鐘有資料
    for t, v in [("09:00:10", 1), ("09:00:40", 2), ("09:01:10", 3), ("09:03:10", 4)]:
        s.observe(q("2330", t, vol=v))
    pm = s.per_minute("2330")
    assert list(pm) == [2, 1, 0, 1]
    r = s.summary("2330")
    assert r["span_minutes"] == 4 and r["minutes_with_update"] == 3
    assert r["coverage"] == pytest.approx(0.75)
    assert r["gap_max_s"] == pytest.approx(120) and r["gap_median_s"] == pytest.approx(30)


def test_open15_window_coverage_only_counts_observed_minutes():
    s = QuoteStats()
    for t, v in [("09:05:10", 1), ("09:06:10", 2), ("09:08:10", 3), ("09:20:10", 4)]:
        s.observe(q("2330", t, vol=v))
    r = s.summary("2330")
    # 開盤 15 分鐘內觀察到 09:05~09:14(有資料的第一分鐘起算),其中 3 分鐘有更新
    assert r["open15_minutes_observed"] == 10
    assert r["open15_coverage"] == pytest.approx(0.3)
    assert r["open15_updates"] == 3


def test_started_after_open_window_reports_no_open15_data():
    s = QuoteStats()
    s.observe(q("2330", "10:00:10", vol=1))
    s.observe(q("2330", "10:01:10", vol=2))
    r = s.summary("2330")
    assert r["open15_coverage"] is None
    assert any("無資料" in line for line in s.format_lines())


def test_single_update_does_not_crash_and_is_labeled_insufficient():
    s = QuoteStats()
    s.observe(q("2330", "09:00:10"))
    r = s.summary("2330")
    assert r["verdict"] == "資料不足" and r["gap_median_s"] is None
    assert any("只有一次更新" in line for line in s.format_lines())


def test_symbol_with_polls_but_no_updates_is_reported():
    s = QuoteStats()
    assert s.summary("9999") == {"symbol": "9999", "polls": 0, "updates": 0}
    assert s.format_lines()[0].startswith("報價密度統計")


def test_verdict_levels():
    dense, mid, sparse = QuoteStats(), QuoteStats(), QuoteStats()
    for k in range(60):                                   # 每 10 秒一次,連續 10 分鐘
        dense.observe(q("A", "09:00:00", vol=0)) if k == 0 else None
    for k in range(60):
        ts_ = T("2026-10-05 09:00:00") + pd.Timedelta(seconds=10 * k)
        dense.observe({"symbol": "A", "ts": ts_, "price": 100.0, "cum_volume": k})
        if k % 6 in (0, 1, 2, 3):                         # 約 2/3 時間有更新 -> 覆蓋率仍高但間隔變大
            mid.observe({"symbol": "A", "ts": ts_ + pd.Timedelta(minutes=k // 6 * 2), "price": 100.0, "cum_volume": k})
    assert dense.summary("A")["verdict"] == "密"
    for k, m in enumerate([0, 1, 9, 10, 25, 26, 40]):     # 稀疏:很多分鐘沒有更新
        sparse.observe({"symbol": "A", "ts": T("2026-10-05 09:00:10") + pd.Timedelta(minutes=m), "price": 100.0, "cum_volume": k})
    assert sparse.summary("A")["verdict"] == "稀疏"


def test_save_csv_and_distinct_session_filenames(tmp_path):
    s = QuoteStats()
    for t, v in [("09:00:10", 1), ("09:01:10", 2)]:
        s.observe(q("2330", t, vol=v))
    p1 = s.save(T("2026-10-05 08:55:00"), tmp_path)
    p2 = s.save(T("2026-10-05 10:00:00"), tmp_path)
    assert p1 != p2 and p1.exists() and p2.exists()      # 同一天重啟不會互相覆蓋
    df = pd.read_csv(p1)
    assert list(df.columns) == ["symbol", "minute", "updates"] and len(df) == 2
    assert QuoteStats().save(T("2026-10-05 08:55:00"), tmp_path) is None


def _setup_script(monkeypatch, tmp_path, script_name, fake_poll, explode_stats=True):
    monkeypatch.setattr(realtime, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(journal, "JOURNAL_DIR", tmp_path / "paper")
    monkeypatch.setattr(logutil, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(stats_mod, "STATS_DIR", tmp_path / "stats")
    mod = paper_trade if script_name == "paper_trade.py" else record_minutes
    monkeypatch.setattr(sys, "argv", [script_name, "2330", "--flush-every", "0"])
    monkeypatch.setattr(mod.live, "poll_loop", fake_poll)
    if explode_stats:
        def boom(self, *a, **k):
            raise RuntimeError("stats exploded")
        monkeypatch.setattr(mod.QuoteStats, "format_lines", boom)
        monkeypatch.setattr(mod.QuoteStats, "save", boom)
    return mod


@pytest.mark.parametrize("script", ["paper_trade.py", "record_minutes.py"])
def test_stats_failure_never_breaks_shutdown(monkeypatch, tmp_path, script):
    def fake_poll(symbols, market, interval, handler, **kw):
        for i, t in enumerate(["09:00:05", "09:01:05", "09:02:05"]):
            handler(q("2330", t, vol=1000 + i))
        kw["flush_fn"]()                                  # 定期存檔時統計也會失敗

    mod = _setup_script(monkeypatch, tmp_path, script, fake_poll)
    mod.main()                                            # 不可拋例外
    assert (tmp_path / "realtime" / "2330_2026-10-05.parquet").exists()   # 資料仍存下來
    text = next((tmp_path / "logs").glob("*.log")).read_text(encoding="utf-8")
    assert "統計" in text and "stats exploded" in text


def test_script_prints_density_report_at_shutdown(monkeypatch, tmp_path):
    def fake_poll(symbols, market, interval, handler, **kw):
        for i, t in enumerate(["09:00:05", "09:00:35", "09:01:05", "09:03:05"]):
            handler(q("2330", t, vol=1000 + i))

    mod = _setup_script(monkeypatch, tmp_path, "record_minutes.py", fake_poll, explode_stats=False)
    mod.main()
    text = next((tmp_path / "logs").glob("record_*.log")).read_text(encoding="utf-8")
    assert "報價密度統計" in text and "2330" in text and "評等" in text and "密度明細" in text
    assert list((tmp_path / "stats").glob("quote_stats_*.csv"))
