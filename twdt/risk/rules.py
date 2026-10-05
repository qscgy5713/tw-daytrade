"""寫死的風控規則,不交給 AI 決定。"""
from dataclasses import dataclass
from datetime import time


@dataclass(frozen=True)
class RiskConfig:
    stop_loss_pct: float = 0.01  # 單筆停損
    take_profit_pct: float = 0.02  # 單筆停利
    max_daily_loss: float = 3000.0  # 單日最大虧損(元),達到後停止開新倉
    force_close_at: time = time(13, 25)  # 強制平倉時間
    no_entry_after: time = time(13, 0)  # 此後不開新倉
    shares_per_trade: int = 1000


def stop_price(entry: float, direction: int, cfg: RiskConfig) -> float:
    return entry * (1 - cfg.stop_loss_pct * direction)


def target_price(entry: float, direction: int, cfg: RiskConfig) -> float:
    return entry * (1 + cfg.take_profit_pct * direction)
