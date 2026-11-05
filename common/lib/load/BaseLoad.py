import math
from lib.Device import Device


class BaseLoad(Device):
    """Cyclic electrical-demand profile.

    Power_Load contains demand power samples. Data repeat cyclically when
    simulation extends beyond their length. data_delta_t is the sample
    period in hours and defaults to Device.delta_t(). update_state()
    records and returns demand power at the current step.
    """

    def __init__(self, Power_Load, data_delta_t=None, dt=None):
        super().__init__(dt)
        if isinstance(Power_Load, (list, tuple)):
            self.Power_Load = [float(p) for p in Power_Load]
        else:
            self.Power_Load = [float(Power_Load)]
        if data_delta_t is not None:
            self.data_delta_t = float(data_delta_t)
        else:
            self.data_delta_t = self.delta_t()
        # Demand history determines the temporal position.
        self.Load_history = []  # len(Load_history) equals the number of executed steps.

    def actual_power(self):
        if not self.Load_history:
            return 0.0
        return self.Load_history[-1]

    def update_state(self):
        largo = len(self.Power_Load)
        step_index = len(self.Load_history)
        indx = math.floor((self.delta_t() / self.data_delta_t) * step_index) + 1
        power_load = self.Power_Load[((indx - 1) % largo)]
        self.Load_history.append(power_load)
        return power_load

    @property
    def stage_k(self):
        return len(self.Load_history) + 1
