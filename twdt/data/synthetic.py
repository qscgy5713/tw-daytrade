"""合成分鐘線(隨機漫步),只用來跑通流程,績效沒有參考價值。"""
import numpy as np
import pandas as pd


def synthetic_minute(start: str, end: str, seed: int = 0, start_price: float = 500.0,
                     minute_vol: float = 0.0006) -> pd.DataFrame:
    """平日 09:00-13:30 每分鐘一根,與 FinMind 分鐘線同欄位、naive 時間。"""
    rng = np.random.default_rng(seed)
    frames = []
    price = start_price
    for day in pd.bdate_range(start, end):
        idx = pd.date_range(day + pd.Timedelta(hours=9), day + pd.Timedelta(hours=13, minutes=30),
                            freq="1min")
        rets = rng.normal(0, minute_vol, len(idx))
        closes = price * np.exp(np.cumsum(rets))
        opens = np.concatenate([[price], closes[:-1]])
        spread = np.abs(rng.normal(0, minute_vol / 2, len(idx))) * closes
        frames.append(pd.DataFrame({
            "open": opens,
            "high": np.maximum(opens, closes) + spread,
            "low": np.minimum(opens, closes) - spread,
            "close": closes,
            "volume": rng.integers(50, 500, len(idx)).astype(float),
        }, index=idx))
        price = closes[-1]
    return pd.concat(frames)
