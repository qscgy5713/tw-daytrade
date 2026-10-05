"""報價密度統計:MIS 是快照而非逐筆,更新可能很稀疏,這裡量化「到底收到多少新資料」。

「更新」指 (快照時間, 累計量, 價格) 與上一次不同的報價;輪詢到完全相同的快照只算 poll、不算更新。
"""
import os
from collections import Counter
from datetime import time as dtime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

STATS_DIR = Path(os.environ.get("TWDT_CACHE", Path(__file__).resolve().parents[2] / "cache")) / "stats"
OPEN_WINDOW = (dtime(9, 0), dtime(9, 15))  # ORB 用的開盤區間,是最在意密度的時段


class QuoteStats:
    def __init__(self):
        self.polls: Counter = Counter()
        self.updates: Counter = Counter()
        self._last: Dict[str, tuple] = {}
        self._minutes: Dict[str, Counter] = {}
        self._times: Dict[str, List[pd.Timestamp]] = {}

    def observe(self, q: dict) -> None:
        sym = q["symbol"]
        self.polls[sym] += 1
        key = (q["ts"], q["cum_volume"], q["price"])
        if self._last.get(sym) == key:
            return
        self._last[sym] = key
        self.updates[sym] += 1
        self._minutes.setdefault(sym, Counter())[q["ts"].floor("min")] += 1
        self._times.setdefault(sym, []).append(q["ts"])

    def per_minute(self, sym: str) -> pd.Series:
        """第一筆到最後一筆之間每分鐘的更新次數(沒有更新的分鐘為 0)。"""
        c = self._minutes.get(sym)
        if not c:
            return pd.Series(dtype=int)
        idx = pd.date_range(min(c), max(c), freq="1min")
        return pd.Series([c.get(m, 0) for m in idx], index=idx, name=sym)

    def summary(self, sym: str) -> dict:
        n_polls, n_upd = self.polls[sym], self.updates[sym]
        if n_upd == 0:
            return {"symbol": sym, "polls": n_polls, "updates": 0}
        pm = self.per_minute(sym)
        times = sorted(set(self._times[sym]))
        gaps = pd.Series(times).diff().dropna().dt.total_seconds()
        out = {
            "symbol": sym,
            "polls": n_polls,
            "updates": n_upd,
            "unchanged_poll_ratio": 1 - n_upd / n_polls,
            "span_minutes": len(pm),
            "minutes_with_update": int((pm > 0).sum()),
            "coverage": float((pm > 0).mean()),
            "updates_per_minute": float(pm.mean()),
            "gap_median_s": float(gaps.median()) if len(gaps) else None,
            "gap_p90_s": float(gaps.quantile(0.9)) if len(gaps) else None,
            "gap_max_s": float(gaps.max()) if len(gaps) else None,
        }
        day = pm.index[0].normalize()
        lo, hi = day + pd.Timedelta(hours=9), day + pd.Timedelta(hours=9, minutes=15)
        win = pm[(pm.index >= lo) & (pm.index < hi)]
        out["open15_minutes_observed"] = len(win)
        out["open15_coverage"] = float((win > 0).mean()) if len(win) else None
        out["open15_updates"] = int(win.sum()) if len(win) else None
        out["verdict"] = _verdict(out)
        return out

    def symbols(self) -> List[str]:
        return sorted(set(self.polls))

    def format_lines(self) -> List[str]:
        lines = ["報價密度統計(更新=快照有變化;覆蓋率=至少有一次更新的分鐘占比):"]
        for sym in self.symbols():
            s = self.summary(sym)
            if s["updates"] == 0:
                lines.append(f"  {sym}: 輪詢 {s['polls']} 次,沒有任何更新")
                continue
            o15 = ("開盤15分鐘 覆蓋率 {:.0%} 更新 {} 次".format(s["open15_coverage"], s["open15_updates"])
                   if s["open15_coverage"] is not None else "開盤15分鐘 無資料(盤中才啟動)")
            gap = ("更新間隔 中位 {:.0f}s / P90 {:.0f}s / 最長 {:.0f}s".format(
                s["gap_median_s"], s["gap_p90_s"], s["gap_max_s"])
                if s["gap_median_s"] is not None else "更新間隔 無法計算(只有一次更新)")
            lines.append(
                f"  {sym}: 輪詢 {s['polls']} 次,更新 {s['updates']} 次"
                f"(重複快照 {s['unchanged_poll_ratio']:.0%});覆蓋 {s['minutes_with_update']}/"
                f"{s['span_minutes']} 分鐘 ({s['coverage']:.0%});每分鐘 {s['updates_per_minute']:.2f} 次;"
                f"{gap};{o15};評等 {s['verdict']}")
        return lines

    def save(self, session_started: pd.Timestamp, directory: Optional[Path] = None) -> Optional[Path]:
        """每檔每分鐘的更新次數存成 CSV(檔名含啟動時間,同一天重啟不會互相覆蓋)。"""
        frames = [self.per_minute(s).rename("updates").rename_axis("minute").reset_index().assign(symbol=s)
                  for s in self.symbols() if self.updates[s]]
        if not frames:
            return None
        directory = directory or STATS_DIR
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"quote_stats_{session_started.strftime('%Y-%m-%d_%H%M%S')}.csv"
        pd.concat(frames, ignore_index=True)[["symbol", "minute", "updates"]].to_csv(path, index=False)
        return path


def _verdict(s: dict) -> str:
    """粗略分級(經驗門檻,僅供快速判讀,以實際數字為準)。"""
    if s["gap_p90_s"] is None:  # 只有一筆更新,算不出間隔
        return "資料不足"
    if s["coverage"] >= 0.95 and s["gap_p90_s"] <= 30:
        return "密"
    if s["coverage"] >= 0.7:
        return "中"
    return "稀疏"
