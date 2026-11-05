from lib.controllers.BaseController import BaseController


class AggressiveSmartController(BaseController):
    """Rule-based controller for surplus charging and deficit discharge.

    Charge from PV surplus while SOC remains below soc_max. During a power
    deficit, discharge above soc_min. At prices not exceeding priceLow,
    allow grid charging when SOC is below soc_threshold, or at the lower
    SOC limit. Current limits and battery efficiencies determine the
    requested charge or discharge current.
    """

    def __init__(self, soc_max, soc_min, soc_threshold=50.0):
        super().__init__()
        self.soc_max = soc_max
        self.soc_min = soc_min
        self.soc_threshold = soc_threshold  # Allow opportunistic charging below this SOC threshold.
        self.option_history = []
        self.priceLow = 0.0

    def setSite(self, site):
        super().setSite(site)
        self.priceLow = min(site.price.prices) * 1.10

    def getPriceLow(self):
        return self.priceLow

    def calculate_command(self):
        capacidad_bateria = self.site.battery.capacidad_bateria
        voltaje_bateria = self.site.battery.voltaje_bateria
        Ibmax = self.site.battery.Ibmax
        Ibmin = self.site.battery.Ibmin
        eff_c = self.site.battery.eff_carga
        eff_d = self.site.battery.eff_descarga
        delta_t = self.site.delta_t() if callable(self.site.delta_t) else self.site.delta_t
        beta = self.site.beta

        SOC = self.site.battery.actual_SOC()
        Ppv = self.site.pv.actual_power()
        Pload = self.site.load.actual_power()
        C = self.site.price.actual_price()

        # Efficiency-adjusted current limits for the SOC bounds.
        Icharge = (
            (self.soc_max - SOC) * capacidad_bateria / (100 * delta_t * eff_c)
            if self.soc_max > SOC else 0.0
        )
        Idischarge = (
            (self.soc_min - SOC) * capacidad_bateria * eff_d / (100 * delta_t)
            if self.soc_min < SOC else 0.0
        )

        pv_for_ctrl = Ppv if getattr(self.site, "_community_noscale", False) else (beta * Ppv)
        Isurplus = (pv_for_ctrl - Pload) / voltaje_bateria

        Ib = 0.0

        if Pload <= pv_for_ctrl:
            # PV surplus.
            if SOC < self.soc_max:
                # Charge from the available surplus within current and SOC limits.
                Ib = min(Isurplus, Icharge, Ibmax)
                opcion = 1
            else:
                Ib = 0.0  # Full battery: no charging current.
                opcion = 2
        else:
            # Demand exceeds allocated PV.
            if SOC > self.soc_min:
                # Discharge during a deficit whenever SOC exceeds the lower limit.
                Ib = max(Idischarge, Ibmin)
                opcion = 3

                # At low prices and low SOC, charge from the grid instead.
                if C <= self.priceLow and SOC < self.soc_threshold:
                    Ib = min(Icharge, Ibmax)
                    opcion = 4
            else:
                # SOC <= soc_min.
                if C <= self.priceLow:
                    # Charge opportunistically at the SOC lower limit when the price is low.
                    Ib = min(Icharge, Ibmax)
                    opcion = 5
                else:
                    Ib = 0.0  # No discharge at the SOC lower limit when the price is high.
                    opcion = 6

        self.option_history.append(opcion)
        return Ib
