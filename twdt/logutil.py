"""事件 log:同時印在終端機並逐行寫進檔案(每行立刻 flush,程式被強制結束也不會漏)。"""
import os
from pathlib import Path
from typing import Callable

import pandas as pd

LOG_DIR = Path(os.environ.get("TWDT_CACHE", Path(__file__).resolve().parents[1] / "cache")) / "logs"
TZ = "Asia/Taipei"


def file_logger(name: str, directory: Path = None) -> Callable[[str], None]:
    """回傳 log(msg):終端機印原文,檔案(<name>_<日期>.log,附加寫入)每行加時間戳。"""
    directory = directory or LOG_DIR
    directory.mkdir(parents=True, exist_ok=True)
    day = pd.Timestamp.now(tz=TZ).strftime("%Y-%m-%d")
    path = directory / f"{name}_{day}.log"

    def log(msg: str) -> None:
        print(msg, flush=True)
        stamp = pd.Timestamp.now(tz=TZ).strftime("%H:%M:%S")
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"{stamp} {msg}\n")
        except OSError as e:  # 寫不進 log 檔不應讓交易程式掛掉
            print(f"[warn] 無法寫入 log 檔 {path}: {e}", flush=True)

    log.path = path
    return log
