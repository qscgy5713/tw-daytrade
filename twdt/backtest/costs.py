"""台股當沖交易成本模型。

手續費:買賣各收 0.1425% x 折扣,單邊最低 20 元。
證交稅:當沖賣出 0.15%(現行減半優惠),只在賣出時收。
滑價:以每邊 tick 數估計。
"""
from dataclasses import dataclass

FEE_RATE = 0.001425
DAYTRADE_TAX_RATE = 0.0015
MIN_FEE = 20.0


def tick_size(price: float) -> float:
    """台股股票升降單位。"""
    if price < 10:
        return 0.01
    if price < 50:
        return 0.05
    if price < 100:
        return 0.1
    if price < 500:
        return 0.5
    if price < 1000:
        return 1.0
    return 5.0


@dataclass(frozen=True)
class CostModel:
    fee_discount: float = 0.6  # 6 折;無折讓請設 1.0
    slippage_ticks: float = 1.0  # 每邊不利方向的 tick 數
    tax_rate: float = DAYTRADE_TAX_RATE
    min_fee: float = MIN_FEE

    def fee(self, price: float, shares: int) -> float:
        return max(self.min_fee, price * shares * FEE_RATE * self.fee_discount)

    def tax(self, sell_price: float, shares: int) -> float:
        return sell_price * shares * self.tax_rate

    def fill_price(self, price: float, side: str) -> float:
        """套用滑價:買貴賣便宜。side 為 'buy' 或 'sell'。"""
        if side == "buy":
            return price + tick_size(price) * self.slippage_ticks
        # 賣出往下走,在整數關卡(100、500、1000)要用下方級距的 tick
        return price - tick_size(price - 1e-9) * self.slippage_ticks

    def round_trip_cost(self, entry: float, exit_: float, shares: int, direction: int) -> float:
        """一來一回的手續費加稅(不含滑價,滑價已反映在成交價)。

        direction: 1 為先買後賣(做多),-1 為先賣後買(現股當沖放空)。
        """
        buy_px, sell_px = (entry, exit_) if direction == 1 else (exit_, entry)
        return self.fee(buy_px, shares) + self.fee(sell_px, shares) + self.tax(sell_px, shares)
