from lib.Device import Device


class BaseController(Device):
    """Base class for battery current controllers.

    Ib_history stores requested currents in A, and site references the
    associated Site. Subclasses implement calculate_command(). The base
    implementation returns 0.0 A. update_state() records and returns the
    current requested from the battery.
    """

    def __init__(self, dt=None):
        super().__init__(dt)
        # Current-command history determines the number of completed steps.
        self.Ib_history = []  # len(Ib_history) equals the number of executed steps.
        self.site = None      # Site controlled by this instance.

    def setSite(self, site):
        # Site on which the controller operates.
        self.site = site

    def calculate_command(self):
        """Return the requested current in A, or 0.0 in the base class."""
        Ib = 0.0
        return Ib

    def actual_Ib(self):
        """Return the last requested current, or zero before the first step."""
        if not self.Ib_history:
            return 0.0
        return self.Ib_history[-1]

    def update_state(self):
        """Record and return the current requested from the battery in A."""
        Ib = self.calculate_command()
        self.Ib_history.append(Ib)
        return Ib


if __name__ == "__main__":
    bc = BaseController()
    for _ in range(3):
        print("Ib:", bc.update_state())
