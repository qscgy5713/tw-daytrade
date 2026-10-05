import sys

import pandas as pd
import pytest

import paper_trade
from twdt import live, logutil
from twdt.data import realtime
from twdt.data import stats as stats_mod
from twdt.data.realtime import DayRecorder, MinuteBarBuilder
from twdt.paper import journal
from twdt.paper.engine import PaperTrader

T = pd.Timestamp


def test_dayrecorder_repeated_flush_does_not_double_count(monkeypatch, tmp_path):
    monkeypatch.setattr(realtime, "CACHE_DIR", tmp_path)
    b = MinuteBarBuilder()
    rec = DayRecorder("2330")
    b.update(T("2026-10-05 09:30:10"), 100, 1000)   # 盤中啟動:首筆累計量不計入
    b.update(T("2026-10-05 09:30:40"), 101, 1010)
    for _ in range(3):                       # 同一個 session 內反覆 flush
        rec.flush(b.to_frame(include_open_bar=True))
    b.update(T("2026-10-05 09:31:10"), 102, 1030)
    rec.flush(b.to_frame(include_open_bar=True))
    df = realtime.load_minute("2330")
    assert len(df) == 2
    assert df.iloc[0]["volume"] == 10        # 沒有被重複相加
    assert list(tmp_path.rglob("*.tmp")) == []


def test_dayrecorder_uses_preexisting_file_as_fixed_baseline(monkeypatch, tmp_path):
    monkeypatch.setattr(realtime, "CACHE_DIR", tmp_path)
    old = MinuteBarBuilder()
    old.update(T("2026-10-05 10:00:05"), 100, 1000)
    old.update(T("2026-10-05 10:00:30"), 103, 1020)   # 前一個 session:量 20
    realtime.save_day("2330", old.to_frame(include_open_bar=True))

    new = MinuteBarBuilder()
    rec = DayRecorder("2330")
    new.update(T("2026-10-05 10:00:40"), 99, 1030)
    new.update(T("2026-10-05 10:00:50"), 101, 1050)    # 新 session 同一分鐘:量 20
    rec.flush(new.to_frame(include_open_bar=True))
    rec.flush(new.to_frame(include_open_bar=True))     # 再 flush 不可讓舊檔基準被重複計入
    row = realtime.load_minute("2330").iloc[0]
    assert row["volume"] == 40 and row["open"] == 100 and row["high"] == 103 and row["low"] == 99


def test_file_logger_writes_lines_and_survives_unwritable_path(tmp_path, capsys):
    log = logutil.file_logger("t", tmp_path)
    log("hello")
    log("world")
    lines = log.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and lines[0].endswith("hello") and lines[1].endswith("world")
    assert "hello" in capsys.readouterr().out
    bad = logutil.file_logger("t", tmp_path)
    bad.path.unlink()
    bad.path.mkdir()                                   # 讓寫檔必定失敗
    bad("still ok")                                    # 不可拋例外
    assert "無法寫入 log 檔" in capsys.readouterr().out


def test_poll_loop_flushes_periodically_and_survives_flush_errors():
    times = [T("2026-10-05 10:00:00", tz=live.TZ)] + [
        T("2026-10-05 10:00:00", tz=live.TZ) + pd.Timedelta(seconds=10 * k) for k in range(1, 40)]
    it = iter(times)

    def now():
        try:
            return next(it)
        except StopIteration:
            return T("2026-10-05 14:00:00", tz=live.TZ)

    flushed, logs = [], []

    def flush():
        flushed.append(1)
        if len(flushed) == 1:
            raise OSError("disk full")

    live.poll_loop(["2330"], "tse", 6, lambda q: None, now_fn=now, fetch_fn=lambda s, m: [],
                   sleep_fn=lambda s: None, log=logs.append, flush_fn=flush, flush_every=60)
    assert len(flushed) >= 2                                   # 失敗後仍持續定期存檔
    assert sum("定期存檔失敗" in l for l in logs) == 1


def _run_script(monkeypatch, tmp_path, feed, extra_argv=()):
    monkeypatch.setattr(realtime, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(journal, "JOURNAL_DIR", tmp_path / "paper")
    monkeypatch.setattr(logutil, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(stats_mod, "STATS_DIR", tmp_path / "stats")
    monkeypatch.setattr(sys, "argv", ["paper_trade.py", "2330", "--flush-every", "0", *extra_argv])
    real_trader = PaperTrader
    # 腳本用牆上時鐘當 started_at;測試固定成 09:00 才不會被「盤中啟動」規則停用
    monkeypatch.setattr(paper_trade, "PaperTrader", lambda *a, **k: real_trader(
        *a, **{**k, "started_at": T("2026-10-05 09:00:00")}))
    monkeypatch.setattr(paper_trade.live, "poll_loop", feed)


def _breakout_quotes():
    out, vol = [], 1000
    for m in range(15):
        for sec in (5, 35):
            out.append((f"09:{m:02d}:{sec:02d}", 100.0))
    out += [("09:15:05", 101.0), ("09:15:40", 101.0), ("09:16:05", 101.0)]  # 09:16:05 進場
    return out


def test_paper_trade_script_end_to_end_flushes_and_logs(monkeypatch, tmp_path):
    seen = {}

    def fake_poll(symbols, market, interval, handler, *, log, flush_fn, flush_every, **kw):
        for i, (t, p) in enumerate(_breakout_quotes()):
            handler({"symbol": "2330", "ts": T(f"2026-10-05 {t}"), "price": p, "cum_volume": 1000 + i})
        flush_fn()                                      # 定期存檔:此時程式還沒結束
        seen["parquet_mid"] = (tmp_path / "realtime" / "2330_2026-10-05.parquet").exists()
        seen["log_mid"] = "進場" in next((tmp_path / "logs").glob("paper_*.log")).read_text(encoding="utf-8")

    _run_script(monkeypatch, tmp_path, fake_poll)
    paper_trade.main()
    assert seen == {"parquet_mid": True, "log_mid": True}        # 結束前資料就已落地
    trades = journal.load_trades(tmp_path / "paper")
    assert len(trades) == 1 and trades.iloc[0]["exit_reason"] == "force_close"
    text = next((tmp_path / "logs").glob("paper_*.log")).read_text(encoding="utf-8")
    assert "成交紀錄" in text and "trades" in text


def test_paper_trade_script_ctrl_c_marks_interrupted_and_saves(monkeypatch, tmp_path):
    def fake_poll(symbols, market, interval, handler, **kw):
        for i, (t, p) in enumerate(_breakout_quotes()):
            handler({"symbol": "2330", "ts": T(f"2026-10-05 {t}"), "price": p, "cum_volume": 1000 + i})
        raise KeyboardInterrupt

    _run_script(monkeypatch, tmp_path, fake_poll)
    paper_trade.main()
    trades = journal.load_trades(tmp_path / "paper")
    assert trades.iloc[0]["exit_reason"] == "interrupted"
    assert (tmp_path / "realtime" / "2330_2026-10-05.parquet").exists()


def test_paper_trade_script_logs_logic_error_then_saves_and_raises(monkeypatch, tmp_path):
    def fake_poll(symbols, market, interval, handler, **kw):
        handler({"symbol": "2330", "ts": T("2026-10-05 09:00:05"), "price": 100.0, "cum_volume": 1})
        raise ValueError("bug in trading logic")

    _run_script(monkeypatch, tmp_path, fake_poll)
    with pytest.raises(ValueError):
        paper_trade.main()
    text = next((tmp_path / "logs").glob("paper_*.log")).read_text(encoding="utf-8")
    assert "[error]" in text and "bug in trading logic" in text
    assert (tmp_path / "realtime" / "2330_2026-10-05.parquet").exists()


def test_paper_trade_script_capital_option_blocks_oversized_entry(monkeypatch, tmp_path):
    def fake_poll(symbols, market, interval, handler, **kw):
        for i, (t, p) in enumerate(_breakout_quotes()):
            handler({"symbol": "2330", "ts": T(f"2026-10-05 {t}"), "price": p, "cum_volume": 1000 + i})

    _run_script(monkeypatch, tmp_path, fake_poll, extra_argv=("--capital", "50000"))
    paper_trade.main()                                  # 一張約 10 萬 > 5 萬上限,不可進場
    assert journal.load_trades(tmp_path / "paper").empty
    text = next((tmp_path / "logs").glob("paper_*.log")).read_text(encoding="utf-8")
    assert "單日買賣額度 50,000" in text and "單日額度不足" in text and "略過 1 次訊號" in text


def test_paper_trade_script_default_capital_is_300k(monkeypatch, tmp_path):
    def fake_poll(symbols, market, interval, handler, **kw):
        pass

    _run_script(monkeypatch, tmp_path, fake_poll)
    paper_trade.main()
    text = next((tmp_path / "logs").glob("paper_*.log")).read_text(encoding="utf-8")
    assert "單日買賣額度 300,000" in text and "今日已用 0" in text
