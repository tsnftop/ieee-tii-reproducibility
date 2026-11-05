class Device:
    """Shared simulation time step in hours.

    The class attribute stores the value used by device instances.
    The default time step is 0.25 h.
    """

    _delta_t_val = None  # Value shared by all instances.

    @classmethod
    def delta_t(cls, new_dt=None):
        """Get or set the shared simulation time step in hours.

        When new_dt is supplied, update the shared value before returning it.
        """
        if new_dt is not None:
            cls._delta_t_val = float(new_dt)
        if cls._delta_t_val is None:
            cls._delta_t_val = 0.25
        return cls._delta_t_val

    def __init__(self, dt=None):
        # Optionally set the shared simulation time step.
        if dt is not None:
            self.delta_t(dt)
