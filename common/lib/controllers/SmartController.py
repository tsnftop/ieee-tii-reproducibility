from lib.controllers.BaseController import BaseController


class SmartController(BaseController):
    """Rule-based battery control using SOC, solar power, demand, and price.

    Calculate charge or discharge currents subject to soc_min, soc_max,
    and the battery current limits.
    """

    def __init__(self, soc_max, soc_min):
        super().__init__()
        self.soc_max = soc_max
        self.soc_min = soc_min
        self.option_history = []
        self.priceLow = None

    def setSite(self, site):
        super().setSite(site)

    def getPriceLow(self):
        """Returns the price threshold for the current calendar day (day-ahead knowledge)."""
        return self._daily_price_low()

    def _daily_price_low(self):
        """Return 1.10 times the minimum price in the current calendar day.

        The input provides all prices for the current midnight-to-midnight day.
        """
        if self.priceLow is not None:
            return float(self.priceLow)

        all_prices = self.site.price.prices
        if not all_prices:
            return 0.0
        delta_t = self.site.delta_t() if callable(self.site.delta_t) else self.site.delta_t
        data_delta_t = getattr(self.site.price, "data_delta_t", delta_t)
        prices_per_day = max(1, round(24.0 / max(data_delta_t, 1e-9)))
        # current step index (0-based): Price_history has already been appended this step
        step_i = max(len(self.site.price.Price_history) - 1, 0)
        current_hour = step_i * delta_t
        day_idx = int(current_hour // 24.0)
        day_start = day_idx * prices_per_day
        day_prices = [
            all_prices[(day_start + k) % len(all_prices)]
            for k in range(prices_per_day)
        ]
        return min(day_prices) * 1.10

    def calculate_command(self):
        capacidad_bateria = self.site.battery.capacidad_bateria
        voltaje_bateria = self.site.battery.voltaje_bateria
        Ibmax = self.site.battery.Ibmax
        Ibmin = self.site.battery.Ibmin
        eff_c = self.site.battery.eff_carga
        eff_d = self.site.battery.eff_descarga
        # Obtain the shared time step from Device.
        delta_t = self.site.delta_t() if callable(self.site.delta_t) else self.site.delta_t
        beta = self.site.beta

        SOC = self.site.battery.actual_SOC()
        Ppv = self.site.pv.actual_power()
        Pload = self.site.load.actual_power()
        C = self.site.price.actual_price()

        Ib = 0.0
        # Compute target currents using charge and discharge efficiencies.
        # Without current saturation, each target reaches its SOC bound in one step.
        # BaseBattery.update_state() uses the following SOC relations.
        # Charging: DSOC = Ib * dt * 100 * eff_c / capacity.
        # Discharging: DSOC = Ib * dt * 100 / (capacity * eff_d).
        # DSOC and Ib are positive for charging and negative for discharging.
        # Positive charging current accounts for charging efficiency.
        # Ib = DSOC * capacity / (dt * 100 * eff_c).
        Icharge = (self.soc_max - SOC) * capacidad_bateria / (100 * delta_t * eff_c) if self.soc_max > SOC else 0.0
        # The discharge target is negative because soc_min - SOC is negative.
        Idischarge = (self.soc_min - SOC) * capacidad_bateria * eff_d / (100 * delta_t) if self.soc_min < SOC else 0.0
        # Available solar power, scaled by beta unless already allocated.
        pv_for_ctrl = Ppv if getattr(self.site, "_community_noscale", False) else (beta * Ppv)
        Isurplus = (pv_for_ctrl - Pload) / voltaje_bateria

        # Day-ahead price threshold: min price of the current calendar day × 1.10
        # All prices for the current calendar day are available to the controller.
        priceLow = self._daily_price_low()

        # Select the current command from SOC, PV surplus, and price.
        if Pload <= pv_for_ctrl:
            if SOC < self.soc_max:
                if C <= priceLow:
                    Ib = min(Icharge, Ibmax)  # Branch 1: efficiency-adjusted charging.
                    opcion = 1
                else:
                    Ib = min(Isurplus, Icharge, Ibmax)  # Branch 2: surplus-limited charging.
                    opcion = 2
            else:
                Ib = 0  # 3
                opcion = 3
        else:
            if SOC > self.soc_min:
                if SOC < self.soc_max:
                    if C <= priceLow:
                        Ib = min(Icharge, Ibmax)  # Branch 4: efficiency-adjusted charging.
                        opcion = 4
                    else:
                        Ib = max(Idischarge, Ibmin)  # Branch 5: efficiency-adjusted discharging.
                        opcion = 5
                else:
                    if C > priceLow:
                        Ib = max(Idischarge, Ibmin)  # Branch 6: efficiency-adjusted discharging.
                        opcion = 6
                    else:
                        Ib = 0  # 7
                        opcion = 7
            else:
                if C <= priceLow:
                    Ib = min(Icharge, Ibmax)  # Branch 8: efficiency-adjusted charging.
                    opcion = 8
                else:
                    Ib = 0  # 9
                    opcion = 9
        # Record the selected control branch.
        self.option_history.append(opcion)
        return Ib
