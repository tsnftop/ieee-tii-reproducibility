from typing import List, Tuple
from lib.Device import Device


class BaseBattery(Device):
    """Battery model with current limits, conversion efficiencies, and SOC bounds.

    Construction parameters:
        capacidad_bateria: Battery capacity in Ah.
        voltaje_bateria: Positive battery voltage in V.
        Ibmax: Maximum charging current in A, positive.
        Ibmin: Maximum discharging current in A, negative.
        eff_carga: Charging efficiency in (0, 1].
        eff_descarga: Discharging efficiency in (0, 1].

    Operating state:
        min_SOC, max_SOC: SOC limits in percent.
        SOC: Current SOC in percent.
        SOC_history: SOC history including the initial state.
        realIb_history, Pbat_history: Applied current and power histories.

    The simulation time step is obtained from Device.delta_t().
    """

    def __init__( 
        self,
        capacidad: float = 100.0,
        voltaje: int = 230,
        ibmax: float = 79.0,
        ibmin: float = -79.0,
        eff_carga: float = 0.9,
        eff_descarga: float = 0.95,
        min_SOC: float = 0.0,
        max_SOC: float = 100.0,
    ) -> None:
        # Validate battery parameter ranges.
        if voltaje <= 0 or not isinstance(voltaje, int):
            raise ValueError("voltaje_bateria debe ser un entero positivo")
        if ibmax <= 0:
            raise ValueError("Ibmax debe ser positivo")
        if ibmin >= 0:
            raise ValueError("Ibmin debe ser negativo")
        if not (0 < eff_carga <= 1):
            raise ValueError("eff_carga debe estar entre 0 y 1")
        if not (0 < eff_descarga <= 1):
            raise ValueError("eff_descarga debe estar entre 0 y 1")
        if min_SOC < 0 or max_SOC < 0 or min_SOC > 100 or max_SOC > 100:
            raise ValueError("min_SOC y max_SOC deben estar entre 0 y 100")

        # Parameters set at construction.
        self.capacidad_bateria: float = capacidad
        self.voltaje_bateria: int = voltaje
        self.Ibmax: float = ibmax
        self.Ibmin: float = ibmin
        self.eff_carga: float = eff_carga
        self.eff_descarga: float = eff_descarga

        # Operating limits and state.
        # Minimum allowed state of charge (%).
        self.min_SOC: float = min_SOC
        # Maximum allowed state of charge (%).
        self.max_SOC: float = max_SOC
        # Current state of charge (%).
        self.SOC: float = self.min_SOC
        # SOC history, including the initial state.
        self.SOC_history: List[float] = [self.SOC]
        # Applied current history (A).
        self.realIb_history: List[float] = []
        # Battery power history (W).
        self.Pbat_history: List[float] = []

    # Device manages the shared time step.

    def set_min_SOC(self, ms: float) -> None:
        if isinstance(ms, (list, tuple)):
            ms = ms[0]
        self.min_SOC = float(ms)

    def set_max_SOC(self, ms: float) -> None:
        if isinstance(ms, (list, tuple)):
            ms = ms[0]
        self.max_SOC = float(ms)

    def set_SOC(self, soc: float) -> None:
        if isinstance(soc, (list, tuple)):
            soc = soc[0]
        soc = float(soc)
        if (soc < self.min_SOC) or (soc > self.max_SOC):
            raise ValueError('SOC fuera de los limites definidos para la batería')
        self.SOC = soc

    def Pbat2Ibat(self, Pbat: float) -> float:
        """Convert requested battery power to current.

        Positive values denote charging and negative values denote discharging.
        """
        if Pbat > 0:
            return (Pbat * self.eff_carga) / self.voltaje_bateria
        else:
            return Pbat / (self.eff_descarga * self.voltaje_bateria)

    def actual_Ib(self) -> float:
        if not self.realIb_history:
            return 0.0
        return self.realIb_history[-1]

    def actual_SOC(self) -> float:
        return self.SOC

    def actual_power(self) -> float:
        if not self.Pbat_history:
            return 0.0
        return self.Pbat_history[-1]

    def update_state(self, Ib: float) -> Tuple[float, float]:

        if isinstance(Ib, (list, tuple)):
            Ib = Ib[0]
        Ib = float(Ib)
        Ib = min(self.Ibmax, max(Ib, self.Ibmin))

        oldSOC = self.SOC
        if Ib >= 0:
            # Charging.
            # DSOC = Ib * dt * 100 * eff_carga / capacidad_bateria.
            dt = self.delta_t()
            DSOC = ((Ib * dt * 100 * self.eff_carga) / self.capacidad_bateria)
            # Limit the SOC increase to the remaining charging headroom.
            realDSOC = max(0.0, min(DSOC, self.max_SOC - oldSOC))
            # Recover applied current from the limited SOC increase.
            realIb = (realDSOC * self.capacidad_bateria) / (dt * 100 * self.eff_carga)
            # Charging input power includes charging efficiency.
            Pb = (self.voltaje_bateria * realIb) / self.eff_carga
        else:
            # Discharging.
            # DSOC = Ib * dt * 100 / (capacidad_bateria * eff_descarga).
            dt = self.delta_t()
            DSOC = ((Ib * dt * 100) / (self.capacidad_bateria * self.eff_descarga))
            # Limit the SOC decrease to the available charge above min_SOC.
            realDSOC = min(0.0, max(DSOC, self.min_SOC - oldSOC))
            # Recover applied current from the limited SOC decrease.
            realIb = (realDSOC * self.capacidad_bateria * self.eff_descarga) / (dt * 100)
            # Discharge power includes discharge efficiency.
            Pb = (self.voltaje_bateria * realIb) * self.eff_descarga

        # Update the battery state.
        tentative_SOC = oldSOC + realDSOC
        # Report tentative SOC values outside the operating limits.
        if tentative_SOC > self.max_SOC + 1e-9 or tentative_SOC < self.min_SOC - 1e-9:  # pragma: no cover (diagnostic branch)
            print(
                f"[BaseBattery:DEBUG] tentative_SOC fuera de rango antes de clamp: "
                f"tent={tentative_SOC:.6f} old={oldSOC:.6f} realDSOC={realDSOC:.6f} DSOC={DSOC:.6f} Ib_cmd={Ib:.6f} "
                f"Ib_real={realIb:.6f} min={self.min_SOC:.3f} max={self.max_SOC:.3f} dt={self.delta_t():.4f}"
            )
        self.SOC = tentative_SOC
        # Clamp SOC to the operating limits after floating-point calculations.
        if self.SOC < self.min_SOC:
            self.SOC = self.min_SOC
        if self.SOC > self.max_SOC:
            print(
                f"[BaseBattery:DEBUG] Clamp aplicado: SOC={self.SOC:.6f} > max={self.max_SOC:.3f}. Ajustando a max. "
                f"(old={oldSOC:.6f}, realDSOC={realDSOC:.6f})"
            )  # pragma: no cover
            self.SOC = self.max_SOC

        # Check that the updated SOC remains within the allowed range.
        # Include state and current values in the exception context.
        if not (self.min_SOC - 1e-6 <= self.SOC <= self.max_SOC + 1e-6):  # pragma: no cover (diagnostic branch)
            raise RuntimeError(
                f"[BaseBattery] SOC fuera de rango tras update_state: {self.SOC} (old={oldSOC}, realDSOC={realDSOC}, Ib_cmd={Ib}, Ib_real={realIb}, DSOC={DSOC}, min={self.min_SOC}, max={self.max_SOC}, dt={self.delta_t()})"
            )

        # Append state, current, and power histories.
        self.SOC_history.append(self.SOC)
        self.realIb_history.append(realIb)
        self.Pbat_history.append(Pb)

        return realIb, Pb

    @property
    def stage_k(self) -> int:
        """Return the battery stage index, including its initial SOC state.

        The SOC history length equals the number of completed steps plus one,
        consistent with the stage convention used by Site.
        """
        return len(self.SOC_history)

if __name__ == "__main__":
    # Standalone battery usage example.
    b = BaseBattery()
    print("Initial SOC:", b.actual_SOC())
    realIb, Pb = b.update_state(10.0)
    print("After update SOC:", b.actual_SOC(), "realIb:", realIb, "Pb:", Pb)
