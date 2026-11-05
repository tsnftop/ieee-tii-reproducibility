import math
from lib.Device import Device


class BasePrice(Device):
    """Cyclic electricity-price profile in EUR/kWh.

    prices contains the price samples. Data repeat cyclically beyond the
    available length. data_delta_t is the sample period in hours and defaults
    to Device.delta_t(). update_state() advances and returns the current price.
    """

    def __init__(self, prices, data_delta_t=None, dt=None):
        super().__init__(dt)
        if isinstance(prices, (list, tuple)):
            self.prices = [float(p) for p in prices]
        else:
            self.prices = [float(prices)]
        if data_delta_t is not None:
            self.data_delta_t = float(data_delta_t)
        else:
            self.data_delta_t = self.delta_t()
        # Price history determines the temporal position.
        self.Price_history = []  # len(Price_history) equals the number of executed steps.

    def actual_price(self):
        if not self.Price_history:
            return 0.0
        return self.Price_history[-1]

    def update_state(self):
        largo = len(self.prices)
        step_index = len(self.Price_history)
        indx = math.floor((self.delta_t() / self.data_delta_t) * step_index) + 1
        power_price = self.prices[((indx - 1) % largo)]
        self.Price_history.append(power_price)
        return power_price

    @property
    def stage_k(self):
        return len(self.Price_history) + 1
