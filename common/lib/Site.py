from typing import Any, List
from lib.Device import Device


class Site(Device):
    """Coupled simulation of demand, PV, battery, controller, and electricity price.

    The class inherits the shared time step from Device. update_state()
    advances the components and records grid power Pg and purchased-energy
    cost Cost. Expected component interfaces are:
        load.update_state(): Demand power Pl and Load_history.
        pv.update_state(): PV power Ppv and PV_history.
        battery.update_state(Ib): Applied current and battery power.
        controller.update_state(): Requested current and Ib_history.
        price.update_state(): Price in EUR/kWh and Price_history.
    """

    def __init__(self, load: Any, beta: float, pv: Any, battery: Any, controller: Any, price: Any, dt=None, allow_export: bool = False, pv_charge_only: bool = False):
        super().__init__(dt)
        self.load = load
        self.beta = beta
        self.pv = pv
        self.battery = battery
        self.controller = controller
        self.price = price
        self.allow_export = bool(allow_export)
        self.pv_charge_only = bool(pv_charge_only)
        # Component histories.
        self.Pg_history: List[float] = []
        self.Cost_history: List[float] = []
        # Blocked export power (W).
        self.Export_blocked_history: List[float] = []
        # Attach the site reference to the controller when supported.
        if hasattr(self.controller, 'setSite'):
            self.controller.setSite(self)

    def update_state(self) -> None:
        """Advance the components and update site power and energy cost."""

        Pl = self.load.update_state()
        Ppv = self.pv.update_state()
        priceAct = self.price.update_state()
        Ib_comando = self.controller.update_state()

        if self.pv_charge_only and Ib_comando > 0.0:
            pv_surplus = max(0.0, Ppv * self.beta - Pl)
            if pv_surplus <= 0.0:
                ib_limit = 0.0
            else:
                voltage = float(getattr(self.battery, "voltaje_bateria", getattr(self.battery, "voltaje", 230.0)))
                voltage = max(voltage, 1e-3)
                ib_limit = pv_surplus / voltage
            if Ib_comando > ib_limit:
                Ib_comando = ib_limit
                if hasattr(self.controller, "Ib_history") and self.controller.Ib_history:
                    self.controller.Ib_history[-1] = Ib_comando

        # battery.update_state expected to return (realIb, Pbat)
        bat_ret = self.battery.update_state(Ib_comando)
        # Accept a battery result as either a tuple/list or a scalar current.
        if isinstance(bat_ret, (tuple, list)):
            _, Pbat = bat_ret
        else:
            Pbat = bat_ret

        # Pg_raw = Pl - Ppv * beta + Pbat.
        Pg_raw = Pl - Ppv * self.beta + Pbat  # Battery power is negative during discharge.
        # Optionally prevent export to the grid.
        if self.allow_export:
            Pg = Pg_raw
            export_blocked = 0.0
        else:
            if Pg_raw < 0:
                export_blocked = -Pg_raw  # Curtailed power in W.
                Pg = 0.0
            else:
                export_blocked = 0.0
                Pg = Pg_raw

        # Append power and cost histories.
        self.Pg_history.append(Pg)
        self.Export_blocked_history.append(export_blocked)
        # Step cost (EUR) = priceAct (EUR/kWh) * max(Pg, 0) * dt / 1000.
        self.Cost_history.append(priceAct * max(Pg, 0.0) * self.delta_t() / 1000.0)

    # The number of executed steps equals len(Pg_history).

    def visualizarResultados(self, silent: bool = True):
        """Plot power, battery current, SOC, cost, and price histories.

        Include requested and applied currents and the battery current limits.
        Return the time vector used for the plots.
        """
        import numpy as np  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore

        if not silent:
            print("Resultados para el Site:")
            costeTotal = sum(self.Cost_history)
            print(f"- Coste total energía: {costeTotal:.2f} €")

        Pg_arr = np.array(self.Pg_history) if self.Pg_history else np.array([])
        energia_red = np.sum(np.maximum(Pg_arr, 0) * self.delta_t()) / 1000.0
        if not silent:
            print(f"- Energía total comprada a la red: {energia_red:.2f} kWh")

        # Power timestamps span completed steps at delta_t intervals.
        # Power histories have one sample per completed step.
        # SOC_history also contains the initial battery state.
        # Construct the power and SOC time axes.
        steps = len(self.Pg_history)
        t = np.arange(0, steps) * self.delta_t()
        # Align the plotted SOC and power arrays by truncating an extra SOC sample.
        # Preserve the complete SOC_history for numerical calculations.
        # The plotted arrays therefore have equal lengths.
        if hasattr(self.battery, 'SOC_history'):
            soc_array = np.array(self.battery.SOC_history)
            if len(soc_array) == steps + 1:
                soc_plot = soc_array[:-1]
            else:
                soc_plot = soc_array
        else:
            soc_plot = np.array([])

        plt.figure(num="Site Simulation", figsize=(10, 8))
        numSp = 4
        gr = 1.0

        # Panel 1: power profiles.
        plt.subplot(numSp, 1, 1)
        if hasattr(self.load, "Load_history"):
            plt.plot(t, np.array(self.load.Load_history), 'r', linewidth=gr, label='Load')
        if hasattr(self.pv, "PV_history"):
            pv_hist = np.array(self.pv.PV_history)
            if getattr(self, "_community_noscale", False):
                pv_plot = pv_hist
                pv_label = 'PV (asignada)'
            else:
                pv_plot = self.beta * pv_hist
                pv_label = 'PV'
            plt.plot(t, pv_plot, 'g', linewidth=gr, label=pv_label)
        if Pg_arr.size:
            plt.plot(t, Pg_arr, 'b', linewidth=gr, label='Pg')
        if hasattr(self.battery, "Pbat_history"):
            plt.plot(t, np.array(self.battery.Pbat_history), 'c', linewidth=gr, label='Pbat')
        plt.grid(True)
        plt.title('Perfiles de Potencia')
        plt.ylabel('Potencia (W)')
        plt.legend()

        # Panel 2: requested and applied battery currents.
        plt.subplot(numSp, 1, 2)
        if t.size > 0:
            plt.plot([t[0], t[-1]], [0, 0], color="black", linewidth=gr + 0.5, label='_nolegend_')
        if hasattr(self.battery, "Ibmax") and t.size > 0:
            plt.plot([t[0], t[-1]], [self.battery.Ibmax, self.battery.Ibmax], "--", label='IMax')
        if hasattr(self.battery, "Ibmin") and t.size > 0:
            plt.plot([t[0], t[-1]], [self.battery.Ibmin, self.battery.Ibmin], "--", label='Imin')
        if hasattr(self.controller, "Ib_history"):
            plt.plot(t, np.array(self.controller.Ib_history), label='Ib', linewidth=gr)
        if hasattr(self.battery, "realIb_history"):
            plt.plot(t, np.array(self.battery.realIb_history), label='Ib Real', linewidth=gr)
        plt.title('Intensidades')
        plt.grid(True)
        plt.ylabel('I (A)')
        plt.legend()

        # Panel 3: battery SOC.
        plt.subplot(numSp, 1, 3)
        if soc_plot.size:
            soc_clamped = np.clip(soc_plot, 0, 100)
            plt.plot(t, soc_clamped, linewidth=gr)
        plt.ylim(0, 100)
        plt.title('Estado de Carga de la Batería (SOC)')
        plt.ylabel('SOC (%)')
        plt.grid(True)

        # Panel 4: energy cost and price.
        ax4 = plt.subplot(numSp, 1, 4)
        if self.Cost_history:
            plt.step(t, np.array(self.Cost_history), where='post', label='Cost', linewidth=gr)
        plt.ylabel('Costo (€)')
        ax4_right = ax4.twinx()
        if hasattr(self.price, "Price_history"):
            price_arr = np.array(self.price.Price_history)
            ax4_right.step(t, price_arr, where='post', label='Price', linewidth=gr, color='#ff8800')  # Orange.
            ax4_right.set_ylabel('Precio (€)')
            # Use fixed price limits when all values lie within [1e-4, 4e-4].
            # Otherwise derive limits from the observed extrema with a margin.
            try:
                pmin = float(price_arr.min()) if price_arr.size else 0.0
                pmax = float(price_arr.max()) if price_arr.size else 1.0
                fixed_low, fixed_high = 1e-4, 4e-4
                use_fixed = pmin >= fixed_low and pmax <= fixed_high
                from matplotlib.ticker import FormatStrFormatter
                if use_fixed:
                    ticks = [1e-4, 2e-4, 3e-4, 4e-4]
                    ax4_right.set_yticks(ticks)
                    ax4_right.set_ylim(fixed_low, fixed_high)
                    ax4_right.yaxis.set_major_formatter(FormatStrFormatter('%.0e'))
                else:
                    # Add a 5% margin around the extrema.
                    if pmin == pmax:
                        pmin *= 0.9
                        pmax *= 1.1 if pmax != 0 else 1.0
                    span = pmax - pmin
                    low = pmin - 0.05 * span
                    high = pmax + 0.05 * span
                    ax4_right.set_ylim(low, high)
                    # Generate up to five price-axis ticks.
                    import math
                    # Estimate the major-tick spacing.
                    exp = math.floor(math.log10(span)) if span > 0 else 0
                    base = 10 ** exp
                    for mult in (1, 2, 5):
                        if span / (base * mult) <= 6:
                            step = base * mult
                            break
                    else:
                        step = base * 10
                    start_tick = math.floor(low / step) * step
                    ticks = []
                    val = start_tick
                    while val <= high and len(ticks) < 12:
                        if val >= low:
                            ticks.append(val)
                        val += step
                    ax4_right.set_yticks(ticks)
                    # Use scientific notation for values below 0.01.
                    if pmax < 0.01:
                        ax4_right.yaxis.set_major_formatter(FormatStrFormatter('%.0e'))
                    else:
                        ax4_right.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            except Exception:  # pragma: no cover
                pass
        plt.title('Costo de la Energía')
        plt.xlabel('Tiempo (horas)')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        return t

    @property
    def stage_k(self):
        return len(self.Pg_history) + 1

    def export_histories(self, path: str | None = None, formats: tuple[str, ...] = ("npz",), include: list[str] | None = None):
        """Return selected histories and optionally save them as NPZ or MAT files.

        Args:
            path: Output path without an extension, or None to return data only.
            formats: Subset of {'npz', 'mat'}. Unknown formats are ignored.
            include: Optional collection of history keys to retain.

        Returns:
            Dictionary mapping history names to sequences of recorded values.
        """
        import numpy as np  # type: ignore
        data: dict[str, any] = {}

        # Collect the component histories.
        data['Pg_history'] = list(self.Pg_history)
        data['Cost_history'] = list(self.Cost_history)

        if hasattr(self.load, 'Load_history'):
            data['Load_history'] = list(getattr(self.load, 'Load_history'))
        if hasattr(self.pv, 'PV_history'):
            data['PV_history'] = list(getattr(self.pv, 'PV_history'))
        if hasattr(self.battery, 'Pbat_history'):
            data['Pbat_history'] = list(getattr(self.battery, 'Pbat_history'))
        if hasattr(self.battery, 'SOC_history'):
            data['SOC_history'] = list(getattr(self.battery, 'SOC_history'))
        if hasattr(self.battery, 'realIb_history'):
            data['realIb_history'] = list(getattr(self.battery, 'realIb_history'))
        if hasattr(self.controller, 'Ib_history'):
            data['Ib_history'] = list(getattr(self.controller, 'Ib_history'))
        if hasattr(self.controller, 'option_history'):
            data['option_history'] = list(getattr(self.controller, 'option_history'))
        if hasattr(self.price, 'Price_history'):
            data['Price_history'] = list(getattr(self.price, 'Price_history'))

        if include is not None:
            data = {k: v for k, v in data.items() if k in include}

        if path:
            base = path
            if 'npz' in formats:
                np.savez(base + '.npz', **{k: np.array(v) for k, v in data.items()})
            if 'mat' in formats:
                try:  # pragma: no cover
                    from scipy.io import savemat  # type: ignore
                    savemat(base + '.mat', {k: np.array(v) for k, v in data.items()})
                except Exception:
                    pass  # Skip MAT export when SciPy is unavailable.
        return data
