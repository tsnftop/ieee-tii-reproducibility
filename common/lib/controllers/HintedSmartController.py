"""Local battery control with externally supplied supervisory constraints.

The discharge_rate parameter in [0, 1] scales the allowed discharging
current. A value of 1.0 permits the full current and 0.0 disables discharge.
The soc_reserve parameter in [0, 1] maps the operating SOC lower bound
linearly from soc_min to soc_reserve_ceil, whose default is 50%.
Physical battery limits remain unchanged. The default supervisory values
are discharge_rate=1.0 and soc_reserve=0.0.
"""

from __future__ import annotations

from lib.controllers.AggressiveSmartController import AggressiveSmartController


class HintedSmartController(AggressiveSmartController):
    """Rule-based battery controller with dynamic supervisory constraints."""

    def __init__(
        self,
        soc_max: float,
        soc_min: float,
        soc_threshold: float = 50.0,
        soc_reserve_ceil: float = 50.0,
    ):
        super().__init__(soc_max, soc_min, soc_threshold)
        # Reserve SOC ceiling reached when soc_reserve=1.0.
        self.soc_reserve_ceil = soc_reserve_ceil

        # Default supervisory values: full discharge rate and no SOC reserve.
        self._discharge_rate: float = 1.0
        self._soc_reserve: float = 0.0

        # Supervisory-constraint histories.
        self.discharge_rate_history: list[float] = []
        self.soc_reserve_history: list[float] = []

    # ------------------------------------------------------------------
    # Supervisory-constraint interface.
    # ------------------------------------------------------------------

    def set_hints(
        self,
        discharge_rate: float = 1.0,
        soc_reserve: float = 0.0,
    ) -> None:
        """Set the supervisory discharge limit and SOC reserve for the next step.

        Args:
            discharge_rate: Fraction of the permitted discharge current in [0, 1].
            soc_reserve: Reserve parameter in [0, 1], mapped linearly to
                [soc_min, soc_reserve_ceil].
        """
        self._discharge_rate = max(0.0, min(1.0, float(discharge_rate)))
        self._soc_reserve = max(0.0, min(1.0, float(soc_reserve)))

    def set_supervisory_constraints(
        self,
        discharge_rate: float = 1.0,
        soc_reserve: float = 0.0,
    ) -> None:
        """Set supervisory constraints through the set_hints interface."""
        self.set_hints(discharge_rate=discharge_rate, soc_reserve=soc_reserve)

    # ------------------------------------------------------------------
    # Local control logic.
    # ------------------------------------------------------------------

    def calculate_command(self) -> float:
        capacidad_bateria = self.site.battery.capacidad_bateria
        voltaje_bateria = self.site.battery.voltaje_bateria
        Ibmax = self.site.battery.Ibmax
        Ibmin = self.site.battery.Ibmin
        eff_c = self.site.battery.eff_carga
        eff_d = self.site.battery.eff_descarga
        delta_t = self.site.delta_t() if callable(self.site.delta_t) else self.site.delta_t

        SOC = self.site.battery.actual_SOC()
        Ppv = self.site.pv.actual_power()
        Pload = self.site.load.actual_power()
        C = self.site.price.actual_price()

        # Dynamic minimum SOC set by the supervisory reserve.
        # Map soc_reserve in [0, 1] to [soc_min, soc_reserve_ceil].
        soc_min_dynamic = self.soc_min + self._soc_reserve * (
            self.soc_reserve_ceil - self.soc_min
        )

        # Current limits based on the dynamic minimum SOC.
        Icharge = (
            (self.soc_max - SOC) * capacidad_bateria / (100 * delta_t * eff_c)
            if self.soc_max > SOC
            else 0.0
        )
        Idischarge = (
            (soc_min_dynamic - SOC) * capacidad_bateria * eff_d / (100 * delta_t)
            if soc_min_dynamic < SOC
            else 0.0
        )  # Discharging current is negative.

        pv_for_ctrl = (
            Ppv
            if getattr(self.site, "_community_noscale", False)
            else (self.site.beta * Ppv)
        )
        Isurplus = (pv_for_ctrl - Pload) / voltaje_bateria

        Ib = 0.0

        if Pload <= pv_for_ctrl:
            # PV surplus.
            if SOC < self.soc_max:
                Ib = min(Isurplus, Icharge, Ibmax)
                opcion = 1
            else:
                Ib = 0.0
                opcion = 2
        else:
            # Demand exceeds allocated PV.
            if SOC > soc_min_dynamic:
                # Scale the permitted discharge current by discharge_rate.
                Ib_full = max(Idischarge, Ibmin)
                Ib = Ib_full * self._discharge_rate
                opcion = 3

                # Allow opportunistic charging at low prices and low SOC.
                if C <= self.priceLow and SOC < self.soc_threshold:
                    Ib = min(Icharge, Ibmax)
                    opcion = 4
            else:
                # SOC <= soc_min_dynamic.
                if C <= self.priceLow:
                    Ib = min(Icharge, Ibmax)
                    opcion = 5
                else:
                    Ib = 0.0
                    opcion = 6

        # Record the selected control branch.
        self.option_history.append(opcion)
        self.discharge_rate_history.append(self._discharge_rate)
        self.soc_reserve_history.append(self._soc_reserve)

        return Ib
