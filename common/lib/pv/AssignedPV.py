from lib.Device import Device


class AssignedPV(Device):
    """PV power supplied externally by a community coordinator.

    set_power(value) sets power in W for the next update_state() call.
    update_state() appends that power to the history and returns it.
    """

    def __init__(self, dt=None):
        super().__init__(dt)
        self._next_power = 0.0
        self.PV_history: list[float] = []

    def set_power(self, value: float) -> None:
        self._next_power = float(value)

    def update_state(self) -> float:
        power = float(self._next_power)
        self.PV_history.append(power)
        return power

    def actual_power(self) -> float:
        if not self.PV_history:
            return 0.0
        return self.PV_history[-1]

    @property
    def stage_k(self) -> int:
        return len(self.PV_history) + 1
