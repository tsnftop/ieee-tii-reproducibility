"""Gymnasium environment for community PV allocation and battery supervision.

A central policy allocates PV among participants and optionally supplies
battery discharge limits and SOC reserves. Local controllers determine
battery operation. Observations include normalized device states and
rolling demand, PV, and price windows. Reward components describe cost
advantage over simulated proportional-allocation reference sites, PV
curtailment, allocation changes, and fairness.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from repo_paths import ROOT, inside, profile_files

from lib.Device import Device
from lib.Site import Site
from lib.batteries.BaseBattery import BaseBattery
from lib.controllers.SmartController import SmartController
from lib.controllers.AggressiveSmartController import AggressiveSmartController
from lib.controllers.HintedSmartController import HintedSmartController
from lib.controllers.CommunityRLController import (
    CommunityRLController,
    CommunityRLControllerConfig,
)
from lib.load.BaseLoad import BaseLoad
from lib.pv.AssignedPV import AssignedPV
from lib.prices.BasePrice import BasePrice
from lib.data_loader import (
    DEFAULT_NOISE_PCT,
    WeeklyBaseProfiles,
    load_weekly_base_profiles,
    perturb_cost_profile,
)


# ======================================================================
# Environment configuration.
# ======================================================================

@dataclass
class CommunityEnvConfig:
    """Configuration of the community simulation and observation spaces."""

    # --- Paths ---
    csv_dir: str | os.PathLike[str] | None = None
    mat_path: str | os.PathLike[str] | None = None

    # Community size.
    n_sites: int = 3

    # Simulation timing.
    delta_t_hours: float = 0.25
    episode_hours: float = 24 * 7  # One week.
    action_repeat: int = 4          # sim steps per agent decision (4 × 0.25h = 1h)

    # Participant battery parameters.
    battery_capacities: List[float] = field(
        default_factory=lambda: [350.0, 150.0, 250.0]
    )
    battery_voltage: int = 230
    battery_ibmax: float = 200.0
    battery_ibmin: float = -200.0
    initial_soc: float = 50.0
    soc_min: float = 10.0
    soc_max: float = 90.0

    # Participant demand-profile time offsets and amplitude scales.
    site_load_offsets_h: List[float] = field(
        default_factory=lambda: [0.0, 8.0, 16.0]
    )
    site_load_scales: List[float] = field(
        default_factory=lambda: [1.2, 0.8, 1.0]     # Participant demand amplitude multipliers.
    )

    # --- PV ---
    pv_scale: float = 2.0
    pv_installed_ratios: Optional[List[float]] = None

    # --- Reward ---
    cost_weight: float = 0.0
    advantage_weight: float = 100.0
    waste_weight: float = 5.0            # Penalty weight for curtailed PV.
    smoothness_weight: float = 0.05     # Penalty weight for allocation changes.
    fairness_weight: float = 2.0        # Penalty weight for deviations from PV reference ratios.

    # Residual allocation action.
    max_beta_deviation: float = 0.6     # Maximum residual deviation around the PV reference ratios.

    # Local battery controller.
    controller_type: str = "hinted"  # Local controller type: smart, aggressive, or hinted.
    aggressive_soc_threshold: float = 50.0  # SOC threshold for opportunistic charging.
    use_hints: bool = True                   # Include per-participant discharge limits and SOC reserves in the action.
    soc_reserve_ceil: float = 50.0           # Upper SOC value represented by the reserve parameter.
    static_discharge_rate: float = 0.2       # Allocation-only policies use fixed battery supervisory constraints.
    static_soc_reserve: float = 0.0

    # --- Fairness ---
    min_beta_ratio: float = 0.05        # Minimum share relative to each PV reference ratio.

    # Electricity prices.
    price_noise_pct: float = DEFAULT_NOISE_PCT
    price_daily_variation: float = 0.2

    # Observation configuration.
    forecast_horizon_hours: float = 8.0  # Rolling demand and PV forecast horizon.
    price_window_hours: float = 8.0      # Future price-window length.


# ======================================================================
# Environment implementation.
# ======================================================================

class CommunityEnv(gym.Env[np.ndarray, np.ndarray]):
    """Gymnasium environment for community PV allocation and battery supervision.

    The agent determines allocation coefficients among N participants.
    Local battery controllers use the configured supervisory constraints.
    """

    metadata: Dict[str, Any] = {"render_modes": []}

    def __init__(self, config: CommunityEnvConfig | None = None):
        super().__init__()
        self.config = config or CommunityEnvConfig()
        self._rng = np.random.default_rng()

        # Set the shared simulation time step.
        Device.delta_t(self.config.delta_t_hours)

        # Simulation and decision-step counts.
        self.max_episode_steps = self._compute_episode_steps()
        self._total_sim_steps = self.max_episode_steps * max(1, self.config.action_repeat)
        self._steps_per_day = max(1, int(round(24.0 / self.config.delta_t_hours)))
        self._price_window_steps = max(
            1, int(round(self.config.price_window_hours))
        )  # hourly resolution (not sub-step)
        self._forecast_hours = max(
            1, int(round(self.config.forecast_horizon_hours))
        )  # hourly resolution (not sub-step)

        # Input profiles.
        self._site_profiles: List[Dict[str, np.ndarray]] = []
        self._base_cost_profile = np.array([], dtype=np.float32)
        self._price_expanded = np.array([], dtype=np.float32)
        self._pv_total_profile = np.array([], dtype=np.float32)
        self._load_total_profile = np.array([], dtype=np.float32)   # Physical-step resolution.
        self._load_total_hourly = np.array([], dtype=np.float32)    # hourly (24/day)

        self._load_all_profiles()

        # PV reference ratios.
        if self.config.pv_installed_ratios is not None:
            r = np.array(self.config.pv_installed_ratios, dtype=np.float32)
            self._pv_ratios = r / r.sum()
        else:
            self._pv_ratios = self._calculate_pv_ratios()

        # Internal episode state.
        self._sites: List[Site] = []
        self._ref_sites: List[Site] = []   # Independent reference sites with proportional allocation.
        self._global_controller: CommunityRLController | None = None
        self._current_step = 0
        self._last_betas = self._pv_ratios.copy()

        # Episode statistics.
        self._episode_costs: List[float] = []
        self._episode_baseline_costs: List[float] = []

        # Exponential moving average for cost normalization.
        self._cost_ema = 1.0  # Update the cost scale during simulation.

        # Create participant objects to determine observation and action spaces.
        self._create_sites()

        # Observation and action spaces.
        obs_dim = self._compute_obs_dim()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # Action: allocation deltas, optionally followed by discharge limits and SOC reserves.
        act_dim = self.config.n_sites * 3 if self.config.use_hints else self.config.n_sites
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(act_dim,), dtype=np.float32
        )

    # ==================================================================
    # Gymnasium API
    # ==================================================================

    def reset(
        self, *, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Sample new participant profiles.
        self._load_all_profiles()

        if self.config.pv_installed_ratios is None:
            self._pv_ratios = self._calculate_pv_ratios()

        # Apply price noise and daily variation.
        perturbed, stats = perturb_cost_profile(
            self._base_cost_profile, rng=self._rng,
            noise_pct=self.config.price_noise_pct,
        )
        price_daily = self._apply_daily_price_variation(perturbed)
        self._price_expanded = self._expand_hourly_prices(price_daily)

        # Create independent site states.
        self._create_sites()

        # Reset episode state.
        self._current_step = 0
        self._last_betas = self._pv_ratios.copy()
        self._episode_costs.clear()
        self._episode_baseline_costs.clear()
        self._cost_ema = 1.0

        obs = self._build_obs()
        info = {
            "step": 0,
            "betas": self._last_betas.tolist(),
            "pv_ratios": self._pv_ratios.tolist(),
            "cost_stats": stats,
        }
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Parse allocation deltas and optional supervisory actions.
        action = np.asarray(action, dtype=np.float32).flatten()
        n = self.config.n_sites
        if self.config.use_hints and action.size >= 3 * n:
            beta_deltas = action[:n]
            # Map supervisory actions from [-1, 1] to [0, 1].
            discharge_rates = np.clip((action[n:2*n] + 1.0) / 2.0, 0.0, 1.0)
            soc_reserves    = np.clip((action[2*n:3*n] + 1.0) / 2.0, 0.0, 1.0)
        else:
            beta_deltas = action[:n]
            discharge_rates = np.full(
                n, np.clip(self.config.static_discharge_rate, 0.0, 1.0),
                dtype=np.float32,
            )
            soc_reserves = np.full(
                n, np.clip(self.config.static_soc_reserve, 0.0, 1.0),
                dtype=np.float32,
            )

        betas = self._deltas_to_betas(beta_deltas)

        # Apply dynamic or fixed supervisory constraints to local controllers.
        for i, site in enumerate(self._sites):
            ctrl = site.controller
            if hasattr(ctrl, 'set_supervisory_constraints'):
                ctrl.set_supervisory_constraints(
                    discharge_rate=float(discharge_rates[i]),
                    soc_reserve=float(soc_reserves[i]),
                )
            elif hasattr(ctrl, 'set_hints'):
                ctrl.set_hints(
                    discharge_rate=float(discharge_rates[i]),
                    soc_reserve=float(soc_reserves[i]),
                )

        ar = max(1, self.config.action_repeat)

        # Accumulate values over physical substeps.
        accum_cost = 0.0
        accum_baseline = 0.0
        accum_wasted = np.zeros(self.config.n_sites, dtype=np.float32)
        accum_pv_total = 0.0
        last_site_costs: List[float] = []
        last_pv_assigned = np.zeros(self.config.n_sites, dtype=np.float32)
        last_price = 0.0

        for sub in range(ar):
            sim_step = self._current_step * ar + sub

            # Read total PV power at this physical substep.
            pv_total = self._get_pv_total(sim_step)
            accum_pv_total += pv_total

            # Allocate PV using the same coefficients throughout the control interval.
            pv_assigned = betas * pv_total
            for i, site in enumerate(self._sites):
                site.pv.set_power(pv_assigned[i])

            # Read the current electricity price.
            last_price = self._get_price(sim_step)

            # Advance the controlled sites by one physical step.
            sub_costs: List[float] = []
            for site in self._sites:
                site.update_state()
                sub_costs.append(site.Cost_history[-1] if site.Cost_history else 0.0)

            # Advance the reference sites with proportional allocation.
            pv_prop = self._pv_ratios * pv_total
            for i, ref_site in enumerate(self._ref_sites):
                ref_site.pv.set_power(pv_prop[i])
            sub_ref: List[float] = []
            for ref_site in self._ref_sites:
                ref_site.update_state()
                sub_ref.append(ref_site.Cost_history[-1] if ref_site.Cost_history else 0.0)

            accum_cost += sum(sub_costs)
            accum_baseline += sum(sub_ref)

            for i, site in enumerate(self._sites):
                accum_wasted[i] += site.Export_blocked_history[-1] if site.Export_blocked_history else 0.0

            last_site_costs = sub_costs
            last_pv_assigned = pv_assigned

        # Average costs over physical substeps to preserve the reward scale.
        total_cost = accum_cost / ar
        baseline_cost = accum_baseline / ar
        pv_wasted = accum_wasted / ar
        pv_total_avg = accum_pv_total / ar

        self._episode_costs.append(accum_cost)
        self._episode_baseline_costs.append(accum_baseline)

        # Update the exponential moving average of cost.
        alpha_ema = 0.01
        self._cost_ema = (1.0 - alpha_ema) * self._cost_ema + alpha_ema * max(abs(total_cost), 1e-6)

        # Allocation history.
        if self._global_controller is not None:
            self._global_controller.beta_history.append(betas.copy())

        self._current_step += 1

        # Reward (computed on per-sub-step averages to keep same scale)
        reward, reward_info = self._compute_reward(
            betas, total_cost, baseline_cost, pv_wasted, pv_total_avg
        )

        self._last_betas = betas.copy()

        # Construct the next observation and termination flag.
        obs = self._build_obs()
        terminated = self._current_step >= self.max_episode_steps
        truncated = False

        info = {
            "step": self._current_step,
            "sim_step": self._current_step * ar,
            "betas": betas.tolist(),
            "site_costs": last_site_costs,
            "total_cost": total_cost,
            "baseline_cost": baseline_cost,
            "pv_total": pv_total_avg,
            "pv_assigned": last_pv_assigned.tolist(),
            "pv_wasted": pv_wasted.tolist(),
            "total_wasted_w": float(pv_wasted.sum()),
            "price": last_price,
            "discharge_rates": discharge_rates.tolist(),
            "soc_reserves": soc_reserves.tolist(),
            **reward_info,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        if not self._sites:
            return None
        return {
            "step": self._current_step,
            "betas": self._last_betas.tolist(),
            "socs": [s.battery.actual_SOC() for s in self._sites],
        }

    def close(self):
        self._sites = []
        self._global_controller = None

    # ==================================================================
    # Reward
    # ==================================================================

    def _compute_reward(
        self,
        betas: np.ndarray,
        total_cost: float,
        baseline_cost: float,
        pv_wasted: np.ndarray,
        pv_total: float,
    ) -> Tuple[float, Dict[str, float]]:
        """Combine economic advantage, PV utilization, smoothing, and fairness.

        The economic term is advantage_weight times the baseline cost advantage
        in EUR, clipped to [-5, 5]. Remaining terms penalize the clipped fraction
        of wasted PV, sum(abs(beta - last_beta)), and sum((beta - pv_ratio)**2).
        A cost term normalized by the moving cost scale is included only when
        cost_weight is positive. Costs are averaged over physical substeps.
        """
        norm = max(self._cost_ema, 1e-6)

        # Economic advantage in EUR relative to the simulated reference sites.
        advantage = baseline_cost - total_cost
        adv_clipped = np.clip(advantage, -5.0, 5.0)
        advantage_term = self.config.advantage_weight * float(adv_clipped)

        # Fraction of PV power not used.
        wasted_total = float(pv_wasted.sum())
        waste_ratio_raw = wasted_total / max(pv_total, 1.0)
        waste_frac = float(np.clip(waste_ratio_raw, 0.0, 1.0))
        waste_term = -self.config.waste_weight * waste_frac

        # Penalty for changes in allocation coefficients.
        beta_change = float(np.abs(betas - self._last_betas).sum())
        smooth_term = -self.config.smoothness_weight * beta_change

        # Fairness penalty for deviations from PV reference ratios.
        deviation_sq = float(np.sum((betas - self._pv_ratios) ** 2))
        fairness_term = -self.config.fairness_weight * deviation_sq

        # Normalized cost term, enabled only when cost_weight is positive.
        cost_term = -self.config.cost_weight * (total_cost / norm) if self.config.cost_weight > 0 else 0.0

        reward = advantage_term + waste_term + smooth_term + fairness_term + cost_term

        info = {
            "reward_cost": float(cost_term),
            "reward_advantage": float(advantage_term),
            "reward_waste": float(waste_term),
            "reward_smooth": float(smooth_term),
            "reward_fairness": float(fairness_term),
            "reward_total": float(reward),
            "raw_cost": total_cost,
            "advantage_eur": float(advantage),
            "waste_fraction": waste_frac,
            "waste_ratio_raw": float(waste_ratio_raw),
            "cost_ema": float(self._cost_ema),
        }
        return float(reward), info

    # ==================================================================
    # Observation construction.
    # ==================================================================

    def _compute_obs_dim(self) -> int:
        """Return the dimension of the observation vector.

        Per participant: SOC, demand ratio, normalized energy headroom, and net need.
        Global features: PV ratio, price ratio, time of day, episode progress,
        price-times-PV, previous allocation, and PV reference ratios.
        Optional supervisory state: previous discharge limits and SOC reserves.
        Future inputs: an hourly price window, total PV forecast, and
        participant demand forecasts over the configured horizon.
        """
        n = self.config.n_sites
        base = 4 * n + 5 + 2 * n + self._price_window_steps
        base += self._forecast_hours          # rolling PV forecast
        base += self._forecast_hours * n      # rolling per-site load forecast
        if self.config.use_hints:
            base += 2 * n  # discharge_rates + soc_reserves
        return base

    @property
    def _sim_step(self) -> int:
        """Current simulation step (sub-step resolution)."""
        return self._current_step * max(1, self.config.action_repeat)

    def _build_obs(self) -> np.ndarray:
        obs: List[float] = []
        total_cap_w = sum(
            c * self.config.battery_voltage for c in self.config.battery_capacities
        )
        sim_step = self._sim_step
        pv_total = self._get_pv_total(sim_step)

        for i, site in enumerate(self._sites):
            soc = site.battery.actual_SOC() / 100.0
            load_now = self._get_site_load(i, sim_step)
            cap_w = self.config.battery_capacities[i] * self.config.battery_voltage
            load_ratio = load_now / max(cap_w, 1.0)
            # Battery energy headroom normalized by total community battery capacity.
            headroom_kwh = max(0.0, self.config.soc_max - site.battery.actual_SOC()) / 100.0 * cap_w / max(total_cap_w, 1.0)
            pv_now = self._last_betas[i] * pv_total
            need = (load_now - pv_now) / max(cap_w, 1.0)
            obs.extend([soc, load_ratio, headroom_kwh, need])

        # Global
        obs.append(pv_total / max(total_cap_w, 1.0))

        price_now = self._get_price(sim_step)
        price_mean = float(np.mean(self._base_cost_profile)) if self._base_cost_profile.size else 0.15
        obs.append(price_now / max(price_mean, 1e-6))

        # Time of day at physical-step resolution.
        time_of_day = (sim_step % self._steps_per_day) / self._steps_per_day
        obs.append(time_of_day)

        # Fraction of the episode completed in [0, 1].
        obs.append(self._current_step / max(self.max_episode_steps, 1))

        # Price-times-PV feature representing the economic value of allocation.
        price_ratio = price_now / max(price_mean, 1e-6)
        pv_ratio_total = pv_total / max(total_cap_w, 1.0)
        obs.append(price_ratio * pv_ratio_total)

        obs.extend(self._last_betas.tolist())
        obs.extend(self._pv_ratios.tolist())

        # Most recently applied supervisory constraints.
        if self.config.use_hints:
            for site in self._sites:
                ctrl = site.controller
                obs.append(getattr(ctrl, '_discharge_rate', 1.0))
            for site in self._sites:
                ctrl = site.controller
                obs.append(getattr(ctrl, '_soc_reserve', 0.0))

        # Normalized future price window at hourly resolution.
        pw = self._get_price_window(sim_step)
        obs.extend((pw / max(price_mean, 1e-6)).tolist())

        # Hourly rolling forecasts of total PV and participant demand.
        pv_forecast = self._get_pv_forecast_rolling(sim_step)
        obs.extend((pv_forecast / max(total_cap_w, 1.0)).tolist())

        load_forecast = self._get_load_forecast_rolling(sim_step)
        for h in range(load_forecast.shape[0]):
            for i in range(self.config.n_sites):
                cap_w = self.config.battery_capacities[i] * self.config.battery_voltage
                obs.append(float(load_forecast[h, i]) / max(cap_w, 1.0))

        return np.array(obs, dtype=np.float32)

    # ==================================================================
    # PV and electricity-price helpers.
    # ==================================================================

    def _max_useful_pv(self, site_idx: int) -> float:
        """Return the maximum PV power a participant can absorb, in W.

        Add current demand to the maximum battery charging power, using the
        configured battery voltage, charging efficiency, and available SOC headroom.
        """
        site = self._sites[site_idx]
        profile = self._site_profiles[site_idx]

        step_idx = self._sim_step % len(profile["load"])
        load = float(profile["load"][step_idx])

        soc = site.battery.actual_SOC()
        capacity_ah = self.config.battery_capacities[site_idx]
        voltage = float(self.config.battery_voltage)
        eff_c = site.battery.eff_carga
        dt = self.config.delta_t_hours

        # Maximum charging current permitted by SOC headroom.
        soc_headroom = max(0.0, self.config.soc_max - soc)
        if dt > 0 and eff_c > 0:
            ib_max_soc = soc_headroom * capacity_ah / (dt * 100.0 * eff_c)
        else:
            ib_max_soc = 0.0

        ib_max = min(ib_max_soc, self.config.battery_ibmax)

        # Charging input power equals current times voltage divided by charge efficiency.
        max_charge_w = ib_max * voltage / eff_c if eff_c > 0 else 0.0

        return load + max_charge_w

    def _get_pv_total(self, step: int) -> float:
        if self._pv_total_profile.size == 0:
            return 0.0
        return float(self._pv_total_profile[step % len(self._pv_total_profile)])

    def _get_site_load(self, site_idx: int, step: int) -> float:
        if not self._site_profiles:
            return 0.0
        profile = self._site_profiles[site_idx % len(self._site_profiles)]
        load = np.asarray(profile.get("load", []), dtype=np.float32)
        if load.size == 0:
            return 0.0
        return float(load[step % load.size])

    def _get_price(self, step: int) -> float:
        if self._price_expanded.size == 0:
            return 0.15
        return float(self._price_expanded[step % len(self._price_expanded)])

    def _get_price_window(self, step: int) -> np.ndarray:
        """Return the future price window at hourly resolution."""
        n = self._price_window_steps  # hours
        if self._price_expanded.size == 0:
            return np.zeros(n, dtype=np.float32)
        steps_per_hour = max(1, int(round(1.0 / self.config.delta_t_hours)))
        w = np.zeros(n, dtype=np.float32)
        sz = len(self._price_expanded)
        for i in range(n):
            w[i] = self._price_expanded[(step + i * steps_per_hour) % sz]
        return w

    @staticmethod
    def _mean_profile_window(values: np.ndarray, start: int, width: int) -> float:
        """Mean over a cyclic sub-step window."""
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            return 0.0
        width = max(1, int(width))
        idx = (start + np.arange(width)) % arr.size
        return float(np.mean(arr[idx]))

    def _get_pv_forecast_rolling(self, sim_step: int) -> np.ndarray:
        """Rolling PV forecast visible to the agent, hourly resolution."""
        h = self._forecast_hours
        forecast = np.zeros(h, dtype=np.float32)
        if self._pv_total_profile.size == 0:
            return forecast
        steps_per_hour = max(1, int(round(1.0 / self.config.delta_t_hours)))
        for k in range(h):
            start = sim_step + k * steps_per_hour
            forecast[k] = self._mean_profile_window(
                self._pv_total_profile, start, steps_per_hour
            )
        return forecast

    def _get_load_forecast_rolling(self, sim_step: int) -> np.ndarray:
        """Rolling per-site load forecast visible to the agent."""
        h = self._forecast_hours
        n = self.config.n_sites
        forecast = np.zeros((h, n), dtype=np.float32)
        if not self._site_profiles:
            return forecast
        steps_per_hour = max(1, int(round(1.0 / self.config.delta_t_hours)))
        for k in range(h):
            start = sim_step + k * steps_per_hour
            for i in range(n):
                profile = self._site_profiles[i % len(self._site_profiles)]
                forecast[k, i] = self._mean_profile_window(
                    np.asarray(profile.get("load", []), dtype=np.float32),
                    start,
                    steps_per_hour,
                )
        return forecast

    def _expand_hourly_prices(self, hourly: np.ndarray) -> np.ndarray:
        """Expand hourly electricity prices to physical simulation steps."""
        n = self._total_sim_steps
        if hourly.size == 0:
            return np.zeros(n, dtype=np.float32)
        dt = max(self.config.delta_t_hours, 1e-6)
        factor = max(1, int(round(1.0 / dt)))
        expanded = np.repeat(hourly, factor)
        if expanded.size >= n:
            return expanded[:n].astype(np.float32)
        reps = int(np.ceil(n / max(expanded.size, 1)))
        return np.tile(expanded, reps)[:n].astype(np.float32)

    # ==================================================================
    # Residual action to allocation coefficients.
    # ==================================================================

    def _deltas_to_betas(self, deltas: np.ndarray) -> np.ndarray:
        """Map residual actions in [-1, 1] to normalized allocation coefficients.

        Form pv_ratios + deltas * max_beta_deviation, apply the minimum share,
        and normalize. Zero action corresponds to the proportional reference
        before the minimum-share and normalization operations.
        """
        deltas = np.clip(
            np.asarray(deltas, dtype=np.float32).flatten(), -1.0, 1.0
        )
        betas = self._pv_ratios + deltas * self.config.max_beta_deviation
        betas = np.maximum(betas, 0.01)
        return betas / betas.sum()

    # ==================================================================
    # Softmax allocation helper, separate from the residual action path.
    # ==================================================================

    def _logits_to_betas(self, logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float32).flatten()
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        raw = exp / (exp.sum() + 1e-8)
        min_b = self._pv_ratios * self.config.min_beta_ratio
        return self._apply_fairness(raw, min_b)

    def _apply_fairness(
        self, raw: np.ndarray, mins: np.ndarray
    ) -> np.ndarray:
        deficit = np.maximum(mins - raw, 0)
        td = deficit.sum()
        if td <= 0:
            return raw
        excess = np.maximum(raw - mins, 0)
        te = excess.sum()
        if te < td:
            rem = 1.0 - mins.sum()
            if rem > 0:
                return mins + (raw / max(raw.sum(), 1e-6)) * rem
            return mins / max(mins.sum(), 1e-6)
        f = td / max(te, 1e-6)
        adj = raw - excess * f + deficit
        return adj / max(adj.sum(), 1e-6)

    # ==================================================================
    # Profile loading and participant construction.
    # ==================================================================

    def _compute_episode_steps(self) -> int:
        sim_steps = max(1, int(round(self.config.episode_hours / max(self.config.delta_t_hours, 1e-6))))
        return sim_steps // max(1, self.config.action_repeat)

    def _resolve_csv_dir(self) -> Path:
        return inside(self.config.csv_dir) if self.config.csv_dir else ROOT / "data/profiles"

    def _resolve_mat_path(self) -> Path:
        return inside(self.config.mat_path) if self.config.mat_path else ROOT / "data/weekly_base_data.mat"

    def _apply_daily_price_variation(self, prices: np.ndarray) -> np.ndarray:
        out = prices.copy()
        n_days = max(1, len(out) // 24)
        var = self.config.price_daily_variation
        factors = self._rng.uniform(1.0 - var, 1.0 + var, size=n_days)
        for d in range(n_days):
            s, e = d * 24, min((d + 1) * 24, len(out))
            out[s:e] *= factors[d]
        return np.maximum(out, 0.0).astype(np.float32)

    def _load_all_profiles(self) -> None:
        csv_dir = self._resolve_csv_dir()
        mat_path = self._resolve_mat_path()
        csv_files = sorted(csv_dir.glob("*.csv")) if self.config.csv_dir else profile_files("training")
        if not csv_files:
            raise RuntimeError(f"No se encontraron CSV en {csv_dir}")

        n = self.config.n_sites
        replace = len(csv_files) < n
        selected = list(self._rng.choice(csv_files, size=n, replace=replace))

        self._site_profiles = []
        pv_list = []
        profiles = None
        for i, csv_path in enumerate(selected):
            profiles = load_weekly_base_profiles(
                csv_dir=csv_dir, mat_path=mat_path,
                delta_minutes=self.config.delta_t_hours * 60.0,
                steps=self._total_sim_steps,
                rng=self._rng, csv_path=csv_path,
                pv_scale=self.config.pv_scale,
            )
            load_raw = np.asarray(profiles.load, dtype=np.float32)

            # Apply the participant demand time offset and amplitude scale.
            offset_h = (self.config.site_load_offsets_h[i]
                        if i < len(self.config.site_load_offsets_h) else 0.0)
            scale = (self.config.site_load_scales[i]
                     if i < len(self.config.site_load_scales) else 1.0)
            if offset_h != 0.0:
                shift_steps = int(round(offset_h / max(self.config.delta_t_hours, 1e-6)))
                load_raw = np.roll(load_raw, shift_steps)
            if scale != 1.0:
                load_raw = load_raw * scale

            self._site_profiles.append({
                "pv": np.asarray(profiles.pv, dtype=np.float32),
                "load": load_raw,
                "csv_name": profiles.csv_name,
            })
            pv_list.append(profiles.pv)

        self._pv_total_profile = np.sum(pv_list, axis=0).astype(np.float32)

        # Load total (sum of all sites) at sub-step resolution
        load_list = [p["load"] for p in self._site_profiles]
        self._load_total_profile = np.sum(load_list, axis=0).astype(np.float32)

        # Average physical-step demand samples into hourly diagnostics.
        dt = max(self.config.delta_t_hours, 1e-6)
        steps_per_hour = max(1, int(round(1.0 / dt)))
        n_hours = len(self._load_total_profile) // steps_per_hour
        if n_hours > 0:
            trimmed = self._load_total_profile[:n_hours * steps_per_hour]
            self._load_total_hourly = trimmed.reshape(n_hours, steps_per_hour).mean(axis=1).astype(np.float32)
        else:
            self._load_total_hourly = self._load_total_profile.copy()

        self._base_cost_profile = np.asarray(profiles.cost_hourly, dtype=np.float32)
        self._price_expanded = self._expand_hourly_prices(self._base_cost_profile)

    def _calculate_pv_ratios(self) -> np.ndarray:
        if not self._site_profiles:
            return np.ones(self.config.n_sites, dtype=np.float32) / self.config.n_sites
        energies = np.array([float(p["pv"].sum()) for p in self._site_profiles], dtype=np.float32)
        total = energies.sum()
        if total <= 0:
            return np.ones(self.config.n_sites, dtype=np.float32) / self.config.n_sites
        return energies / total

    def _create_sites(self) -> None:
        self._sites = []
        self._ref_sites = []
        for i in range(self.config.n_sites):
            profile = self._site_profiles[i] if i < len(self._site_profiles) else self._site_profiles[0]
            cap = self.config.battery_capacities[i] if i < len(self.config.battery_capacities) else 250.0

            # Controlled site.
            battery = BaseBattery(
                capacidad=cap, voltaje=self.config.battery_voltage,
                ibmax=self.config.battery_ibmax, ibmin=self.config.battery_ibmin,
            )
            battery.set_min_SOC(self.config.soc_min)
            battery.set_max_SOC(self.config.soc_max)
            battery.set_SOC(self.config.initial_soc)

            load = BaseLoad(profile["load"].tolist(), data_delta_t=self.config.delta_t_hours)
            pv = AssignedPV()
            price = BasePrice(self._price_expanded.tolist(), data_delta_t=self.config.delta_t_hours)
            if self.config.controller_type == "hinted":
                controller = HintedSmartController(
                    self.config.soc_max, self.config.soc_min,
                    soc_threshold=self.config.aggressive_soc_threshold,
                    soc_reserve_ceil=self.config.soc_reserve_ceil,
                )
            elif self.config.controller_type == "aggressive":
                controller = AggressiveSmartController(
                    self.config.soc_max, self.config.soc_min,
                    soc_threshold=self.config.aggressive_soc_threshold,
                )
            else:
                controller = SmartController(self.config.soc_max, self.config.soc_min)

            site = Site(
                load=load, beta=1.0, pv=pv, battery=battery,
                controller=controller, price=price,
                allow_export=False, pv_charge_only=False,
            )
            site._community_noscale = True
            self._sites.append(site)

            # Independent reference site for baseline cost calculations.
            ref_battery = BaseBattery(
                capacidad=cap, voltaje=self.config.battery_voltage,
                ibmax=self.config.battery_ibmax, ibmin=self.config.battery_ibmin,
            )
            ref_battery.set_min_SOC(self.config.soc_min)
            ref_battery.set_max_SOC(self.config.soc_max)
            ref_battery.set_SOC(self.config.initial_soc)

            ref_load = BaseLoad(profile["load"].tolist(), data_delta_t=self.config.delta_t_hours)
            ref_pv = AssignedPV()
            ref_price = BasePrice(self._price_expanded.tolist(), data_delta_t=self.config.delta_t_hours)
            # Reference sites use AggressiveSmartController without supervisory actions.
            ref_controller = AggressiveSmartController(
                self.config.soc_max, self.config.soc_min,
                soc_threshold=self.config.aggressive_soc_threshold,
            )

            ref_site = Site(
                load=ref_load, beta=1.0, pv=ref_pv, battery=ref_battery,
                controller=ref_controller, price=ref_price,
                allow_export=False, pv_charge_only=False,
            )
            ref_site._community_noscale = True
            self._ref_sites.append(ref_site)

        self._global_controller = CommunityRLController(
            CommunityRLControllerConfig(
                n_sites=self.config.n_sites,
                pv_installed_ratios=self._pv_ratios.tolist(),
                min_beta_ratio=self.config.min_beta_ratio,
            )
        )

    def get_episode_summary(self) -> Dict[str, Any]:
        if not self._sites:
            return {}
        total_cost = sum(sum(s.Cost_history) for s in self._sites)
        total_bl = sum(self._episode_baseline_costs) if self._episode_baseline_costs else 0.0
        grid_kwh = sum(
            sum(max(pg, 0) for pg in s.Pg_history) * self.config.delta_t_hours / 1000.0
            for s in self._sites
        )
        fairness = self._global_controller.get_fairness_metrics() if self._global_controller else {}
        return {
            "total_cost_eur": total_cost,
            "baseline_cost_eur": total_bl,
            "cost_reduction_pct": 100.0 * (1.0 - total_cost / max(total_bl, 1e-6)) if total_bl > 0 else 0.0,
            "grid_import_kwh": grid_kwh,
            "final_socs": [s.battery.actual_SOC() for s in self._sites],
            **fairness,
        }


__all__ = ["CommunityEnv", "CommunityEnvConfig"]
