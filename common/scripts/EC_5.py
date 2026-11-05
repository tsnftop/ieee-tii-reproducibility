"""Community simulation for PV allocation and battery-control benchmarks.

Central strategies determine PV allocation and optional battery discharge
limits and SOC reserves. Local battery controllers operate each participant
using its demand and allocated PV profiles. Supported strategies include
uniform, proportional, rule-based, PPO, and MPC control. Study-specific
entry points select methods and export cost, energy, fairness, and battery
metrics from these simulation functions.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from repo_paths import ROOT, inside, profile_files

from lib.Device import Device
from lib.Site import Site
from lib.batteries.BaseBattery import BaseBattery
from lib.controllers.SmartController import SmartController
from lib.controllers.AggressiveSmartController import AggressiveSmartController
from lib.controllers.HintedSmartController import HintedSmartController
from lib.controllers.CommunityRLController import CommunityRLController, CommunityRLControllerConfig
from lib.load.BaseLoad import BaseLoad
from lib.pv.AssignedPV import AssignedPV
from lib.prices.BasePrice import BasePrice
from lib.data_loader import (
    load_weekly_base_profiles,
    perturb_cost_profile,
    DEFAULT_NOISE_PCT,
)


# ---------------------------------------------------------------------------
# Simulation configuration.
# ---------------------------------------------------------------------------

@dataclass
class CommunitySimConfig:
    """Configuration for participant devices and community simulation."""
    n_sites: int = 3
    delta_t: float = 0.25  # Hours.
    sim_hours: float = 24 * 7  # One week.
    
    # Participant-specific battery capacities.
    battery_capacities: List[float] = field(default_factory=lambda: [350.0, 150.0, 250.0])
    battery_voltage: int = 230
    battery_ibmax: float = 200.0
    battery_ibmin: float = -200.0
    soc_min: float = 10.0
    soc_max: float = 90.0
    initial_soc: float = 50.0
    
    # PV
    pv_scale: float = 2.0
    
    # Participant demand-profile time offsets and amplitude scales.
    site_load_offsets_h: List[float] = field(default_factory=lambda: [0.0, 8.0, 16.0])
    site_load_scales: List[float] = field(default_factory=lambda: [1.2, 0.8, 1.0])
    
    # Fairness
    min_beta_ratio: float = 0.2
    
    # Recompute allocation every action_repeat physical steps.
    action_repeat: int = 4

    # RL observation with rolling forecasts and price windows.
    forecast_horizon_hours: float = 8.0
    price_window_hours: float = 8.0


@dataclass
class ScenarioResult:
    """Simulation result with aggregate metrics and component histories."""
    label: str
    sites: List[Site]
    total_cost: float
    grid_import_kwh: float
    beta_history: List[np.ndarray]
    fairness_metrics: Dict[str, Any]
    # Recorded histories for analysis.
    soc_history: Optional[List[np.ndarray]] = None
    load_history: Optional[List[np.ndarray]] = None
    need_history: Optional[List[np.ndarray]] = None  # Demand minus allocated PV power.
    discharge_rate_history: Optional[List[np.ndarray]] = None
    soc_reserve_history: Optional[List[np.ndarray]] = None
    elapsed_seconds: float = 0.0  # Elapsed evaluation time in seconds.


@dataclass(frozen=True)
class RLModelSpec:
    """Policy and normalization resources for an RL control strategy."""

    mode: str
    label: str
    policy_path: Path
    vecnorm_path: Optional[Path]
    use_hints: bool


# ---------------------------------------------------------------------------
# Input profile loading.
# ---------------------------------------------------------------------------

def load_community_profiles(
    n_sites: int,
    csv_dir: Path,
    mat_path: Path,
    delta_t: float,
    steps: int,
    pv_scale: float,
    rng: np.random.Generator,
    site_load_offsets_h: Optional[List[float]] = None,
    site_load_scales: Optional[List[float]] = None,
    csv_paths: Optional[List[Path]] = None,
) -> tuple[List[Dict[str, np.ndarray]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load participant profiles and a shared perturbed price series.

    Args:
        site_load_offsets_h: Demand-profile shifts in hours. Positive offsets
            shift the profile to the right.
        site_load_scales: Participant demand amplitude multipliers.

    Returns:
        Participant dictionaries containing pv, load, and csv_name.
        Total community PV power.
        Shared perturbed price profile.
        Participant PV reference ratios.
        Unperturbed base price profile for observation normalization.
    """
    csv_files = sorted(csv_paths) if csv_paths is not None else sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"No se encontraron CSV en {csv_dir}")
    
    # Select participant CSV profiles, with replacement only when required.
    if len(csv_files) < n_sites:
        selected = list(rng.choice(csv_files, size=n_sites, replace=True))
    else:
        selected = list(rng.choice(csv_files, size=n_sites, replace=False))
    
    site_profiles = []
    pv_totals = []
    
    for i, csv_path in enumerate(selected):
        profiles = load_weekly_base_profiles(
            csv_dir=csv_dir,
            mat_path=mat_path,
            delta_minutes=delta_t * 60.0,
            steps=steps,
            rng=rng,
            csv_path=csv_path,
            pv_scale=pv_scale,
        )
        load_raw = np.asarray(profiles.load, dtype=np.float32)

        # Apply participant demand time offsets and amplitude scales.
        offset_h = (site_load_offsets_h[i]
                    if site_load_offsets_h and i < len(site_load_offsets_h) else 0.0)
        scale = (site_load_scales[i]
                 if site_load_scales and i < len(site_load_scales) else 1.0)
        if offset_h != 0.0:
            shift_steps = int(round(offset_h / max(delta_t, 1e-6)))
            load_raw = np.roll(load_raw, shift_steps)
        if scale != 1.0:
            load_raw = load_raw * scale

        site_profiles.append({
            "pv": np.asarray(profiles.pv, dtype=np.float32),
            "load": load_raw,
            "csv_name": profiles.csv_name,
        })
        pv_totals.append(profiles.pv)
    
    # Total community PV power.
    pv_total = np.sum(pv_totals, axis=0).astype(np.float32)
    
    # Use the base price profile returned by the final participant profile.
    base_price = np.asarray(profiles.cost_hourly, dtype=np.float32)
    price_profile, _ = perturb_cost_profile(base_price, rng=rng, noise_pct=DEFAULT_NOISE_PCT)
    
    # PV reference ratios.
    pv_energies = [p["pv"].sum() for p in site_profiles]
    total_energy = sum(pv_energies)
    if total_energy > 0:
        pv_ratios = np.array(pv_energies, dtype=np.float32) / total_energy
    else:
        pv_ratios = np.ones(n_sites, dtype=np.float32) / n_sites
    
    return site_profiles, pv_total, price_profile, pv_ratios, base_price


# ---------------------------------------------------------------------------
# Participant construction.
# ---------------------------------------------------------------------------

def create_sites(
    site_profiles: List[Dict[str, np.ndarray]],
    price_profile: np.ndarray,
    config: CommunitySimConfig,
) -> List[Site]:
    """Construct participant devices and local battery controllers."""
    sites = []
    
    for i, profile in enumerate(site_profiles):
        # Battery capacity for this participant.
        capacity = config.battery_capacities[i] if i < len(config.battery_capacities) else 250.0
        battery = BaseBattery(
            capacidad=capacity,
            voltaje=config.battery_voltage,
            ibmax=config.battery_ibmax,
            ibmin=config.battery_ibmin,
        )
        battery.set_min_SOC(config.soc_min)
        battery.set_max_SOC(config.soc_max)
        battery.set_SOC(config.initial_soc)
        
        load = BaseLoad(profile["load"].tolist(), data_delta_t=config.delta_t)
        pv = AssignedPV()
        price = BasePrice(price_profile.tolist(), data_delta_t=1.0)
        controller = HintedSmartController(
            config.soc_max, config.soc_min,
            soc_threshold=50.0, soc_reserve_ceil=50.0,
        )
        
        site = Site(
            load=load,
            beta=1.0,
            pv=pv,
            battery=battery,
            controller=controller,
            price=price,
            allow_export=False,
            pv_charge_only=False,
        )
        site._community_noscale = True
        sites.append(site)
    
    return sites


def _profile_value(values: np.ndarray, step: int) -> float:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    return float(arr[step % arr.size])


def _mean_profile_window(values: np.ndarray, start: int, width: int) -> float:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    width = max(1, int(width))
    idx = (start + np.arange(width)) % arr.size
    return float(np.mean(arr[idx]))


# ---------------------------------------------------------------------------
# Allocation strategies.
# ---------------------------------------------------------------------------

def uniform_beta_strategy(n_sites: int, step: int, **kwargs) -> np.ndarray:
    """Allocate the same PV share, 1/N, to each participant."""
    return np.ones(n_sites, dtype=np.float32) / n_sites


def proportional_beta_strategy(n_sites: int, step: int, pv_ratios: np.ndarray, **kwargs) -> np.ndarray:
    """Allocate PV in proportion to the reference PV ratios."""
    return pv_ratios.copy()


def create_static_supervisory_strategy(
    base_strategy: Callable,
    discharge_rate: float = 0.5,
    soc_reserve: float = 0.1,
) -> Callable:
    """Wrap an allocation strategy with fixed battery supervisory constraints.

    Return allocation coefficients together with constant discharge limits
    and SOC reserves for all participants.
    """
    def strategy(n_sites: int, step: int, **kwargs) -> dict:
        betas = base_strategy(n_sites=n_sites, step=step, **kwargs)
        return {
            'betas': betas,
            'discharge_rates': np.full(n_sites, discharge_rate, dtype=np.float32),
            'soc_reserves': np.full(n_sites, soc_reserve, dtype=np.float32),
        }
    return strategy


def create_hinted_static_strategy(
    base_strategy: Callable,
    discharge_rate: float = 0.5,
    soc_reserve: float = 0.1,
) -> Callable:
    """Return an allocation strategy with fixed supervisory constraints."""
    return create_static_supervisory_strategy(
        base_strategy,
        discharge_rate=discharge_rate,
        soc_reserve=soc_reserve,
    )


def create_rule_based_beta_strategy(
    n_sites: int,
    pv_ratios: np.ndarray,
    w_need: float = 0.4,
    w_soc: float = 0.3,
    w_price: float = 0.3,
) -> Callable:
    """Construct a heuristic PV allocation rule.

    Combine positive net demand, inverse SOC, and price-weighted demand using
    w_need, w_soc, and w_price. Blend the normalized scores with pv_ratios,
    apply the coefficient floor, and normalize the allocation.
    """

    def rule_strategy(n_sites: int, step: int, observation: np.ndarray, **kwargs) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)

        scores = np.zeros(n_sites, dtype=np.float32)

        for i in range(n_sites):
            soc_norm   = obs[4 * i]       # SOC / 100
            load_ratio = obs[4 * i + 1]   # Load / cap_w
            need_ratio = obs[4 * i + 3]   # (Load − PV) / cap_w

            # Positive net demand identifies participants with a PV deficit.
            need_score = max(need_ratio, 0.0)

            # Inverse SOC prioritizes participants with lower battery charge.
            soc_score = max(1.0 - soc_norm, 0.0)

            # Price-weighted need prioritizes deficits when electricity is expensive.
            price_ratio = obs[4 * n_sites + 1]   # price / price_mean
            price_need_score = price_ratio * need_score

            scores[i] = (
                w_need  * need_score +
                w_soc   * soc_score +
                w_price * price_need_score
            )

        # Blend dynamic rule scores with PV reference ratios.
        if scores.sum() > 1e-8:
            score_weights = scores / scores.sum()
            alpha = 0.6  # Relative weight of rule scores.
            betas = (1 - alpha) * pv_ratios + alpha * score_weights
        else:
            betas = pv_ratios.copy()

        betas = np.maximum(betas, 0.01)
        betas = betas / betas.sum()
        return betas

    return rule_strategy


def create_mpc_beta_strategy(
    site_profiles: List[Dict[str, np.ndarray]],
    pv_total: np.ndarray,
    price_profile: np.ndarray,
    config: "CommunitySimConfig",
    pv_ratios: np.ndarray,
    horizonte_horas: float = 8.0,
    peso_equidad: float = 0.1,
    discharge_rate_hint: float = 0.2,
    soc_reserve_hint: float = 0.0,
    mode: str = "legacy",
    maxiter: int = 500,
) -> Callable:
    """Construct an MPC strategy for PV allocation and battery supervision.

    The first-block allocation mode optimizes the executable allocation while
    using reference allocations in subsequent blocks. The beta-sequence mode
    optimizes allocation throughout the horizon. The supervisory mode also
    optimizes discharge limits and SOC reserves throughout the horizon.

    The internal battery model uses charging efficiency 0.9, discharge
    efficiency 0.95, current limits Ibmax and Ibmin, and the SOC transition
    used by BaseBattery. Discharge is scaled by discharge_rate. Grid power
    is max(load - allocated_PV + battery_power, 0), with no grid export.
    """
    from scipy.optimize import minimize

    mode = mode.strip().lower()
    if mode not in {"legacy", "beta-sequence", "supervisory"}:
        raise ValueError(f"Modo MPC no soportado: {mode}")

    n_sites = config.n_sites
    delta_t = config.delta_t
    action_repeat = config.action_repeat
    total_sim_steps = int(config.sim_hours / delta_t)
    horizon_sim_steps = int(horizonte_horas / delta_t)
    n_blocks = max(1, int(np.ceil(horizon_sim_steps / max(action_repeat, 1))))
    beta_blocks = 1 if mode == "legacy" else n_blocks
    static_dr = float(np.clip(discharge_rate_hint, 0.0, 1.0))
    static_sr = float(np.clip(soc_reserve_hint, 0.0, 1.0))

    # Complete participant input profiles.
    all_loads = [np.asarray(p["load"], dtype=np.float64) for p in site_profiles]
    pv_arr = np.asarray(pv_total, dtype=np.float64)
    price_arr = np.asarray(price_profile, dtype=np.float64)

    # Battery parameters shared with the local controller and battery model.
    bat_caps = np.array([
        config.battery_capacities[i] if i < len(config.battery_capacities) else 250.0
        for i in range(n_sites)
    ], dtype=np.float64)
    voltage = float(config.battery_voltage)
    soc_max = config.soc_max
    soc_min_base = config.soc_min
    eff_c = 0.9   # BaseBattery default
    eff_d = 0.95  # BaseBattery default
    Ibmax = config.battery_ibmax
    Ibmin = config.battery_ibmin

    soc_reserve_ceil = 50.0

    # Price threshold for opportunistic charging in the local controller.
    price_low = float(np.min(price_arr)) * 1.10
    soc_threshold = 50.0  # SOC threshold used by HintedSmartController.

    _call_count = [0]
    _default_beta_seq = np.tile(pv_ratios.astype(np.float64), (n_blocks, 1))
    _default_dr_seq = np.full((n_blocks, n_sites), static_dr, dtype=np.float64)
    _default_sr_seq = np.full((n_blocks, n_sites), static_sr, dtype=np.float64)
    _last_beta_seq = [_default_beta_seq.copy()]
    _last_dr_seq = [_default_dr_seq.copy()]
    _last_sr_seq = [_default_sr_seq.copy()]

    def _normalize_beta_vector(beta: np.ndarray) -> np.ndarray:
        beta = np.nan_to_num(np.asarray(beta, dtype=np.float64), nan=0.0)
        beta = np.maximum(beta, 0.01)
        total = float(beta.sum())
        if total <= 1e-12:
            return np.ones(n_sites, dtype=np.float64) / n_sites
        return beta / total

    def _normalize_beta_sequence(seq: np.ndarray) -> np.ndarray:
        seq = np.asarray(seq, dtype=np.float64).reshape(-1, n_sites)
        return np.vstack([_normalize_beta_vector(row) for row in seq])

    def _shift_sequence(seq: np.ndarray, fill: np.ndarray) -> np.ndarray:
        seq = np.asarray(seq, dtype=np.float64).reshape(n_blocks, n_sites)
        out = np.empty_like(seq)
        if n_blocks > 1:
            out[:-1] = seq[1:]
        out[-1] = fill
        return out

    def _pack(beta_seq: np.ndarray,
              dr_seq: Optional[np.ndarray] = None,
              sr_seq: Optional[np.ndarray] = None) -> np.ndarray:
        beta_seq = _normalize_beta_sequence(beta_seq)
        if mode == "legacy":
            return beta_seq[0].astype(np.float64)

        parts = [beta_seq.reshape(-1)]
        if mode == "supervisory":
            if dr_seq is None:
                dr_seq = _default_dr_seq
            if sr_seq is None:
                sr_seq = _default_sr_seq
            parts.append(np.clip(np.asarray(dr_seq, dtype=np.float64), 0.0, 1.0).reshape(-1))
            parts.append(np.clip(np.asarray(sr_seq, dtype=np.float64), 0.0, 1.0).reshape(-1))
        return np.concatenate(parts).astype(np.float64)

    def _unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0)
        if mode == "legacy":
            beta_seq = _default_beta_seq.copy()
            beta_seq[0] = _normalize_beta_vector(x[:n_sites])
            return beta_seq, _default_dr_seq.copy(), _default_sr_seq.copy()

        beta_size = n_blocks * n_sites
        beta_seq = _normalize_beta_sequence(x[:beta_size].reshape(n_blocks, n_sites))
        dr_seq = _default_dr_seq.copy()
        sr_seq = _default_sr_seq.copy()
        if mode == "supervisory":
            dr_start = beta_size
            sr_start = dr_start + beta_size
            dr_seq = np.clip(x[dr_start:sr_start].reshape(n_blocks, n_sites), 0.0, 1.0)
            sr_seq = np.clip(x[sr_start:sr_start + beta_size].reshape(n_blocks, n_sites), 0.0, 1.0)
        return beta_seq, dr_seq, sr_seq

    def _simulate_horizon(beta_seq: np.ndarray,
                          dr_seq: np.ndarray,
                          sr_seq: np.ndarray,
                          start_step: int,
                          current_socs: np.ndarray) -> float:
        """Evaluate horizon cost using local control and battery transitions."""
        remaining = total_sim_steps - start_step
        h_steps = min(horizon_sim_steps, remaining)
        socs = current_socs.copy()
        cost = 0.0

        for ds in range(h_steps):
            s = start_step + ds
            if s >= total_sim_steps:
                break

            block_idx = min(ds // max(action_repeat, 1), n_blocks - 1)
            b = beta_seq[block_idx]
            dr_values = dr_seq[block_idx]
            sr_values = sr_seq[block_idx]

            pv_step = float(pv_arr[s % len(pv_arr)])
            price_step = float(price_arr[int(s * delta_t) % len(price_arr)])

            for i in range(n_sites):
                load_i = float(all_loads[i][s % len(all_loads[i])])
                pv_i = b[i] * pv_step
                surplus = pv_i - load_i  # Positive surplus denotes excess allocated PV.
                cap_i = bat_caps[i]
                soc_i = socs[i]
                soc_min_dyn = soc_min_base + sr_values[i] * (soc_reserve_ceil - soc_min_base)

                if surplus >= 0:
                    # PV surplus: charge within SOC and current limits.
                    if soc_i < soc_max:
                        Isurplus = surplus / voltage
                        Icharge_max = ((soc_max - soc_i) * cap_i
                                       / (100.0 * delta_t * eff_c))
                        Ib = min(Isurplus, Icharge_max, Ibmax)
                        # SOC: DSOC = Ib * dt * 100 * eff_c / cap
                        DSOC = Ib * delta_t * 100.0 * eff_c / cap_i
                        realDSOC = max(0.0, min(DSOC, soc_max - soc_i))
                        socs[i] += realDSOC
                        realIb = (realDSOC * cap_i) / (delta_t * 100.0 * eff_c)
                        Pb = voltage * realIb / eff_c
                    else:
                        Pb = 0.0
                else:
                    # Demand deficit: determine discharge or opportunistic charging.
                    if soc_i > soc_min_dyn:
                        # Scale permitted discharge by discharge_rate.
                        Idischarge_max = ((soc_min_dyn - soc_i) * cap_i * eff_d
                                          / (100.0 * delta_t))  # Negative discharge current.
                        Ib_full = max(Idischarge_max, Ibmin)
                        Ib = Ib_full * dr_values[i]

                        # At low prices and low SOC, charge opportunistically.
                        if price_step <= price_low and soc_i < soc_threshold:
                            Icharge_max = ((soc_max - soc_i) * cap_i
                                           / (100.0 * delta_t * eff_c))
                            Ib = min(Icharge_max, Ibmax)
                            DSOC = Ib * delta_t * 100.0 * eff_c / cap_i
                            realDSOC = max(0.0, min(DSOC, soc_max - soc_i))
                            socs[i] += realDSOC
                            realIb = (realDSOC * cap_i) / (delta_t * 100.0 * eff_c)
                            Pb = voltage * realIb / eff_c
                        else:
                            # Discharge subject to the dynamic SOC floor.
                            DSOC = Ib * delta_t * 100.0 / (cap_i * eff_d)
                            realDSOC = min(0.0, max(DSOC, soc_min_dyn - soc_i))
                            socs[i] += realDSOC
                            realIb = (realDSOC * cap_i * eff_d) / (delta_t * 100.0)
                            Pb = voltage * realIb * eff_d
                    else:
                        # SOC at or below the operating lower limit.
                        if price_step <= price_low:
                            Icharge_max = ((soc_max - soc_i) * cap_i
                                           / (100.0 * delta_t * eff_c))
                            Ib = min(Icharge_max, Ibmax)
                            DSOC = Ib * delta_t * 100.0 * eff_c / cap_i
                            realDSOC = max(0.0, min(DSOC, soc_max - soc_i))
                            socs[i] += realDSOC
                            realIb = (realDSOC * cap_i) / (delta_t * 100.0 * eff_c)
                            Pb = voltage * realIb / eff_c
                        else:
                            Pb = 0.0

                # Pg = Pl - Ppv + Pbat, clamped ≥ 0  (Site.update_state)
                Pg = load_i - pv_i + Pb
                Pg = max(Pg, 0.0)
                cost += Pg / 1000.0 * price_step * delta_t

        # Jain-based fairness penalty.
        fairness_beta = beta_seq[:1] if mode == "legacy" else beta_seq
        mean_beta = fairness_beta.mean(axis=0)
        jain = (mean_beta.sum() ** 2) / (n_sites * (mean_beta ** 2).sum() + 1e-8)
        cost += (1.0 - jain) * peso_equidad * max(cost, 0.01)

        return cost

    def _constraints() -> list[dict[str, Any]]:
        constraints = []
        for block in range(beta_blocks):
            start = block * n_sites
            constraints.append({
                'type': 'eq',
                'fun': lambda x, s=start: np.sum(x[s:s + n_sites]) - 1.0,
            })
        return constraints

    def _bounds() -> list[tuple[float, float]]:
        beta_bounds = [(0.01, 1.0)] * (beta_blocks * n_sites)
        if mode == "supervisory":
            return beta_bounds + [(0.0, 1.0)] * (2 * n_blocks * n_sites)
        return beta_bounds

    def _candidate_vectors() -> list[np.ndarray]:
        uniform_beta_seq = np.tile(np.ones(n_sites, dtype=np.float64) / n_sites, (n_blocks, 1))
        if mode == "legacy":
            return [
                _last_beta_seq[0][0].copy(),
                pv_ratios.astype(np.float64).copy(),
                np.ones(n_sites, dtype=np.float64) / n_sites,
            ]

        shifted_beta = _shift_sequence(_last_beta_seq[0], pv_ratios.astype(np.float64))
        candidates = [
            _pack(shifted_beta, _shift_sequence(_last_dr_seq[0], _default_dr_seq[-1]), _shift_sequence(_last_sr_seq[0], _default_sr_seq[-1])),
            _pack(_default_beta_seq, _default_dr_seq, _default_sr_seq),
            _pack(uniform_beta_seq, _default_dr_seq, _default_sr_seq),
        ]
        return candidates

    def mpc_strategy(n_sites: int, step: int, observation: np.ndarray, **kwargs) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)

        _call_count[0] += 1
        if _call_count[0] % 24 == 1:
            print(f"  [MPC {mode}] Decisión {_call_count[0]}, step={step}")

        # Recover current SOC percentages from the observation.
        current_socs = np.array(
            [obs[4 * i] * 100.0 for i in range(n_sites)], dtype=np.float64
        )

        def objective(x):
            beta_seq, dr_seq, sr_seq = _unpack(x)
            return _simulate_horizon(beta_seq, dr_seq, sr_seq, step, current_socs)

        best_cost = np.inf
        best_x = _pack(_default_beta_seq, _default_dr_seq, _default_sr_seq)

        for x0 in _candidate_vectors():
            sol = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=_bounds(),
                constraints=_constraints(),
                options={'disp': False, 'maxiter': maxiter, 'ftol': 1e-7},
            )
            if np.isfinite(sol.fun) and sol.fun < best_cost:
                best_cost = sol.fun
                best_x = sol.x.copy()

        beta_seq, dr_seq, sr_seq = _unpack(best_x)
        _last_beta_seq[0] = beta_seq.copy()
        _last_dr_seq[0] = dr_seq.copy()
        _last_sr_seq[0] = sr_seq.copy()

        betas = beta_seq[0].astype(np.float32)
        if mode == "supervisory":
            return {
                'betas': betas,
                'discharge_rates': dr_seq[0].astype(np.float32),
                'soc_reserves': sr_seq[0].astype(np.float32),
            }
        return betas

    return mpc_strategy


def create_rl_beta_strategy(
    policy_path: Path,
    vecnorm_path: Optional[Path],
    n_sites: int,
    pv_ratios: np.ndarray,
    min_beta_ratio: float,
) -> Callable[[int, int, Any], np.ndarray]:
    """Load a trained policy and its paired observation normalizer."""
    
    # Load the selected policy.
    policy = None
    normalizer = None
    expected_obs_dim: Optional[int] = None
    
    from stable_baselines3 import PPO
    policy_path = inside(policy_path)
    vecnorm_path = inside(vecnorm_path) if vecnorm_path else None
    if not policy_path.is_file() or vecnorm_path is None or not vecnorm_path.is_file():
        raise FileNotFoundError("A PPO model and its paired normalizer are required")
    policy = PPO.load(str(policy_path))
    obs_shape = getattr(getattr(policy, "observation_space", None), "shape", None)
    if obs_shape:
        expected_obs_dim = int(np.prod(obs_shape))

    # Load the paired normalization statistics.
    if vecnorm_path and vecnorm_path.is_file():
        try:
            try:
                import cloudpickle as pickle
            except ImportError:
                import pickle
            
            with open(vecnorm_path, "rb") as f:
                vecnorm = pickle.load(f)
            obs_rms = getattr(vecnorm, "obs_rms", None)
            if obs_rms is not None:
                mean = np.array(obs_rms.mean, dtype=np.float32)
                var = np.array(obs_rms.var, dtype=np.float32)
                clip = float(getattr(vecnorm, "clip_obs", 10.0))
                expected_obs_dim = int(mean.size)
                
                def normalize(obs: np.ndarray) -> np.ndarray:
                    vec = np.asarray(obs, dtype=np.float32).flatten()
                    if vec.size != mean.size:
                        raise ValueError(
                            f"VecNormalize espera obs_dim={mean.size}, pero EC_5 construyo obs_dim={vec.size}. "
                            "Reentrena el modelo con la observacion rolling actual."
                        )
                    return np.clip((vec - mean) / np.sqrt(var + 1e-8), -clip, clip)
                
                normalizer = normalize
                print(f"[EC_5] Normalizador cargado desde {vecnorm_path}")
        except Exception as e:
            raise RuntimeError("Could not load paired normalizer") from e
    if normalizer is None:
        raise ValueError("Normalizer lacks observation statistics")
    
    # Maximum residual allocation deviation shared with the environment.
    
    max_beta_dev = 0.6
    
    def rl_strategy(n_sites: int, step: int, observation: np.ndarray, **kwargs) -> dict:
        """Return allocation coefficients, discharge limits, and SOC reserves."""
        result = {
            'betas': pv_ratios.copy(),
            'discharge_rates': np.full(n_sites, 0.2, dtype=np.float32),
            'soc_reserves': np.zeros(n_sites, dtype=np.float32),
        }
        if policy is None:
            return result
        
        obs = np.asarray(observation, dtype=np.float32)
        if expected_obs_dim is not None and obs.size != expected_obs_dim:
            raise ValueError(
                f"La politica espera obs_dim={expected_obs_dim}, pero EC_5 construyo obs_dim={obs.size}. "
                "Los modelos antiguos quedan invalidados por la observacion Markov-friendly; reentrena antes de validar RL."
            )
        if normalizer is not None:
            obs = normalizer(obs)
        
        action, _ = policy.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32).flatten()
        
        # Parse action: first n_sites = beta deltas, next 2*n_sites = dr/sr
        beta_deltas = np.clip(action[:n_sites], -1.0, 1.0)
        betas = pv_ratios + beta_deltas * max_beta_dev
        betas = np.maximum(betas, 0.01)
        betas = betas / betas.sum()
        result['betas'] = betas
        
        if action.size >= 3 * n_sites:
            result['discharge_rates'] = np.clip((action[n_sites:2*n_sites] + 1.0) / 2.0, 0.0, 1.0)
            result['soc_reserves']    = np.clip((action[2*n_sites:3*n_sites] + 1.0) / 2.0, 0.0, 1.0)
        
        return result
    
    return rl_strategy


# ---------------------------------------------------------------------------
# Community simulation.
# ---------------------------------------------------------------------------

def run_scenario(
    label: str,
    site_profiles: List[Dict[str, np.ndarray]],
    pv_total: np.ndarray,
    price_profile: np.ndarray,
    pv_ratios: np.ndarray,
    config: CommunitySimConfig,
    beta_strategy: Callable,
    base_cost_profile: Optional[np.ndarray] = None,
    use_hints: bool = True,
) -> ScenarioResult:
    """Simulate a strategy over the configured community evaluation horizon."""
    
    print(f"\n[EC_5] Ejecutando: {label}")
    t_start = time.perf_counter()
    
    # Create independent participant states.
    sites = create_sites(site_profiles, price_profile, config)
    
    steps = int(config.sim_hours / config.delta_t)
    beta_history = []
    last_betas = pv_ratios.copy()  # Match the initial allocation used when resetting the training environment.
    
    steps_per_day = max(1, int(24.0 / config.delta_t))
    
    # Construct observation features using the environment conventions.
    # Normalize prices by the unperturbed base-profile mean.
    # The environment uses the same base cost profile for normalization.
    _base_for_mean = base_cost_profile if base_cost_profile is not None else price_profile
    price_mean = float(np.mean(_base_for_mean)) if len(_base_for_mean) > 0 else 0.15
    total_cap_w = sum(c * config.battery_voltage for c in config.battery_capacities)
    price_window_hours_int = max(1, int(round(config.price_window_hours)))
    forecast_horizon_hours_int = max(1, int(round(config.forecast_horizon_hours)))
    steps_per_hour_obs = max(1, int(1.0 / config.delta_t))  # 4 sim-steps per hour

    # Most recently applied supervisory constraints for observation feedback.
    last_discharge_rates = np.ones(config.n_sites, dtype=np.float32)
    last_soc_reserves = np.zeros(config.n_sites, dtype=np.float32)
    discharge_rate_history = []
    soc_reserve_history = []

    for step in range(steps):
        # Request a control action every action_repeat physical steps.
        decide = (step % config.action_repeat == 0)
        
        if decide:
            # Construct observation features in the environment's feature order.
            # Supervisory N=3: 4*N + 5 + 2*N + 2*N + price_window + PV_forecast + load_forecast = 69.
            obs_parts = []
            pv_total_now = float(pv_total[step % len(pv_total)])
            
            # Per site (4 features each): soc_norm, load_ratio, headroom, need_ratio
            for i, site in enumerate(sites):
                soc = site.battery.actual_SOC() / 100.0
                load_now = _profile_value(site_profiles[i]["load"], step)
                cap_w = config.battery_capacities[i] * config.battery_voltage
                load_ratio = load_now / max(cap_w, 1.0)
                # Battery energy headroom normalized by community battery capacity.
                headroom_kwh = max(0.0, config.soc_max - site.battery.actual_SOC()) / 100.0 * cap_w / max(total_cap_w, 1.0)
                pv_now = last_betas[i] * pv_total_now
                need = (load_now - pv_now) / max(cap_w, 1.0)
                obs_parts.extend([soc, load_ratio, headroom_kwh, need])
            
            # Global PV, price, and time-of-day features.
            obs_parts.append(pv_total_now / max(total_cap_w, 1.0))
            
            price_now = float(price_profile[int(step * config.delta_t) % len(price_profile)])
            obs_parts.append(price_now / max(price_mean, 1e-6))
            
            time_of_day = (step % steps_per_day) / steps_per_day
            obs_parts.append(time_of_day)
            
            # Episode progress, using the environment convention.
            agent_step = step // config.action_repeat
            max_agent_steps = int(config.sim_hours / config.delta_t) // config.action_repeat
            obs_parts.append(agent_step / max(max_agent_steps, 1))

            # Price-times-PV feature representing economic allocation value.
            price_ratio_obs = price_now / max(price_mean, 1e-6)
            pv_ratio_obs = pv_total_now / max(total_cap_w, 1.0)
            obs_parts.append(price_ratio_obs * pv_ratio_obs)
            
            # Last betas (n_sites)
            obs_parts.extend(last_betas.tolist())
            
            # PV ratios (n_sites)
            obs_parts.extend(pv_ratios.tolist())

            # Current supervisory constraints, with two values per participant.
            if use_hints:
                obs_parts.extend(last_discharge_rates.tolist())
                obs_parts.extend(last_soc_reserves.tolist())
            
            # Hourly rolling price window normalized by price_mean.
            for pw_i in range(price_window_hours_int):
                future_sim_step = step + pw_i * steps_per_hour_obs
                price_idx = int(future_sim_step * config.delta_t) % len(price_profile)
                obs_parts.append(float(price_profile[price_idx]) / max(price_mean, 1e-6))

            # Rolling forecast of total PV power.
            for h in range(forecast_horizon_hours_int):
                start = step + h * steps_per_hour_obs
                pv_h = _mean_profile_window(pv_total, start, steps_per_hour_obs)
                obs_parts.append(pv_h / max(total_cap_w, 1.0))

            # Rolling forecast of participant demand.
            for h in range(forecast_horizon_hours_int):
                start = step + h * steps_per_hour_obs
                for i in range(config.n_sites):
                    cap_w = config.battery_capacities[i] * config.battery_voltage
                    load_h = _mean_profile_window(
                        site_profiles[i]["load"], start, steps_per_hour_obs
                    )
                    obs_parts.append(load_h / max(cap_w, 1.0))

            observation = np.array(obs_parts, dtype=np.float32)
            
            # Compute allocation only at control-decision steps.
            strategy_result = beta_strategy(
                n_sites=config.n_sites,
                step=step,
                observation=observation,
                pv_ratios=pv_ratios,
            )
            # Strategies return an allocation array or a dictionary with supervisory signals.
            if isinstance(strategy_result, dict):
                betas = strategy_result['betas']
                last_discharge_rates = strategy_result.get('discharge_rates', np.ones(config.n_sites, dtype=np.float32))
                last_soc_reserves = strategy_result.get('soc_reserves', np.zeros(config.n_sites, dtype=np.float32))
            else:
                betas = strategy_result
                last_discharge_rates = np.ones(config.n_sites, dtype=np.float32)
                last_soc_reserves = np.zeros(config.n_sites, dtype=np.float32)

            betas = np.nan_to_num(np.asarray(betas, dtype=np.float32), nan=0.0)
            betas = np.maximum(betas, 0.01)
            betas = betas / max(float(betas.sum()), 1e-8)
            last_discharge_rates = np.clip(
                np.nan_to_num(np.asarray(last_discharge_rates, dtype=np.float32), nan=1.0),
                0.0, 1.0,
            )
            last_soc_reserves = np.clip(
                np.nan_to_num(np.asarray(last_soc_reserves, dtype=np.float32), nan=0.0),
                0.0, 1.0,
            )
            
            # Apply supervisory constraints to the local controllers.
            for i, site in enumerate(sites):
                ctrl = site.controller
                if hasattr(ctrl, 'set_supervisory_constraints'):
                    ctrl.set_supervisory_constraints(
                        discharge_rate=float(last_discharge_rates[i]),
                        soc_reserve=float(last_soc_reserves[i]),
                    )
                elif hasattr(ctrl, 'set_hints'):
                    ctrl.set_hints(
                        discharge_rate=float(last_discharge_rates[i]),
                        soc_reserve=float(last_soc_reserves[i]),
                    )
            
            last_betas = betas
        
        # Record and simulate every physical step with the currently held allocation.
        beta_history.append(last_betas.copy())
        discharge_rate_history.append(last_discharge_rates.copy())
        soc_reserve_history.append(last_soc_reserves.copy())
        
        # Distribute community PV power.
        pv_step = float(pv_total[step % len(pv_total)])
        for i, site in enumerate(sites):
            site.pv.set_power(last_betas[i] * pv_step)
        
        # Advance the participant components.
        for site in sites:
            site.update_state()
    
    # Compute episode metrics.
    total_cost = sum(sum(s.Cost_history) for s in sites)
    grid_import = sum(
        sum(max(pg, 0) for pg in s.Pg_history) * config.delta_t / 1000.0
        for s in sites
    )
    
    # Allocation fairness metrics.
    betas_arr = np.array(beta_history)
    mean_betas = betas_arr.mean(axis=0)
    std_betas = betas_arr.std(axis=0)
    jain_index = (mean_betas.sum() ** 2) / (config.n_sites * (mean_betas ** 2).sum())
    deviation = np.abs(mean_betas - pv_ratios).mean()
    
    fairness_metrics = {
        "mean_betas": mean_betas.tolist(),
        "std_betas": std_betas.tolist(),
        "jain_index": float(jain_index),
        "deviation_from_pv_ratio": float(deviation),
    }
    
    # Collect histories for analysis.
    soc_history = []
    load_history = []
    need_history = []  # Net need equals demand minus allocated PV power.
    
    for step in range(len(sites[0].Pg_history)):
        socs = np.array([s.battery.SOC_history[step] if step < len(s.battery.SOC_history) else 50.0 for s in sites])
        loads = np.array([s.load.Load_history[step] if step < len(s.load.Load_history) else 0.0 for s in sites])
        pvs = np.array([s.pv.PV_history[step] if step < len(s.pv.PV_history) else 0.0 for s in sites])
        needs = loads - pvs  # Positive values denote deficits and negative values denote surplus.
        
        soc_history.append(socs)
        load_history.append(loads)
        need_history.append(needs)
    
    elapsed = time.perf_counter() - t_start
    print(f"  → {label} completado en {elapsed:.2f}s")

    return ScenarioResult(
        label=label,
        sites=sites,
        total_cost=total_cost,
        grid_import_kwh=grid_import,
        beta_history=beta_history,
        fairness_metrics=fairness_metrics,
        soc_history=soc_history,
        load_history=load_history,
        need_history=need_history,
        discharge_rate_history=discharge_rate_history,
        soc_reserve_history=soc_reserve_history,
        elapsed_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# Result visualization.
# ---------------------------------------------------------------------------

def plot_comparison(results: List[ScenarioResult], config: CommunitySimConfig, pv_ratios: np.ndarray, pv_total: np.ndarray, price_profile: np.ndarray) -> None:
    """Generate comparison figures as PDF and PNG files.

    F1: RL PV allocation trajectories.
    F2a: Weekly total costs.
    F2b: Weekly grid imports.
    F3: Participant costs for RL and Uniform.
    F4: Participant costs for RL and constrained static strategies.
    F5: Participant costs for RL, Rule-Based, and MPC.
    F6: Percentage cost savings relative to Uniform.
    F7: Battery SOC trajectories for RL, MPC, and Rule-Based.
    F8: Allocated PV and demand for each participant under RL.
    F9: Jain index and deviation from PV reference ratios.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
    except ImportError:
        print("[EC_5] matplotlib not available, skipping plots")
        return

    # =====================================================================
    # Plot typography and line sizes.
    # =====================================================================
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.4,
        'axes.linewidth': 0.6,
        'lines.linewidth': 1.2,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.08,
        'figure.dpi': 150,
    })

    # Single- and double-column figure widths in inches.
    W1 = 7.16     # double-column
    W05 = 3.5     # single-column
    H_ROW = 1.8   # typical row height

    dt = config.delta_t
    n_sites = config.n_sites

    # Output directory
    fig_dir = inside("main/outputs/plots")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- Classify results by label ----
    def _find(keyword: str, exclude_baseline: bool = True) -> Optional[ScenarioResult]:
        for r in results:
            if keyword in r.label:
                if exclude_baseline and "(baseline)" in r.label:
                    continue
                return r
        return None

    res_rl = _find("RL")
    res_uni_base = _find("Uniform", exclude_baseline=False)
    if res_uni_base and "(baseline)" not in res_uni_base.label:
        res_uni_base = next((r for r in results if "Uniform" in r.label and "(baseline)" in r.label), res_uni_base)
    res_uni = _find("Uniform")
    res_prop = _find("Proportional")
    res_rules = _find("Rule")
    res_mpc = _find("MPC")

    # Baseline cost for percentage calculations
    baseline_cost = res_uni_base.total_cost if res_uni_base else (results[0].total_cost if results else 1.0)

    # High-contrast palette with hatches for grayscale rendering.
    C_RL = '#1B2838'        # dark blue-black
    C_UNI_B = '#7F8C8D'     # mid grey
    C_UNI = '#2980B9'       # blue
    C_PROP = '#27AE60'      # green
    C_RULES = '#E67E22'     # orange
    C_MPC = '#C0392B'       # red
    SITE_COLORS = ['#2C3E50', '#E74C3C', '#27AE60']

    # Build ordered scenario list with colors + hatches
    all_scenarios = []
    for r, color, hatch in [
        (res_rl, C_RL, ''),
        (res_uni_base, C_UNI_B, '..'),
        (res_uni, C_UNI, ''),
        (res_prop, C_PROP, ''),
        (res_rules, C_RULES, '//'),
        (res_mpc, C_MPC, '\\'),
    ]:
        if r is not None:
            all_scenarios.append((r, color, hatch))

    def _save(fig, name: str) -> None:
        fig.savefig(fig_dir / f"{name}.pdf", format='pdf')
        fig.savefig(fig_dir / f"{name}.png", format='png')
        print(f"  [Plot] {name}.pdf saved")

    def _day_lines(ax, max_h: float) -> None:
        for d in range(int(max_h // 24) + 2):
            ax.axvline(24 * d, color='0.75', linestyle=':', linewidth=0.5, alpha=0.5)

    # =====================================================================
    # F1 — RL PV Distribution (β) Over Time
    # =====================================================================
    if res_rl:
        fig1, ax1 = plt.subplots(figsize=(W1, H_ROW * 1.8))
        betas = np.array(res_rl.beta_history)
        steps_f1 = len(betas)
        t = np.arange(steps_f1) * dt

        bottom = np.zeros(steps_f1)
        for i in range(n_sites):
            ax1.fill_between(t, bottom * 100, (bottom + betas[:, i]) * 100,
                             color=SITE_COLORS[i], alpha=0.8,
                             label=f'Site {i+1}')
            bottom += betas[:, i]

        # PV ratio reference lines
        cum = 0.0
        for i in range(n_sites - 1):
            cum += pv_ratios[i]
            ax1.axhline(cum * 100, color='white', linestyle='--', linewidth=1.0, alpha=0.9)

        ax1.set_ylim(0, 100)
        ax1.set_xlabel('Time (h)')
        ax1.set_ylabel('PV Share (%)')
        ax1.set_title('RL Agent Daily PV Distribution Among Sites')
        ax1.legend(loc='upper right', framealpha=0.9, edgecolor='0.8')
        _day_lines(ax1, t[-1] if len(t) else 0)
        fig1.tight_layout()
        _save(fig1, 'F1_pv_distribution_rl')

    # =====================================================================
    # F2 — Weekly Cost + Grid Import (combined, horizontal bars)
    # =====================================================================
    # Short labels to avoid clipping on the y-axis
    _SHORT_LABELS = {
        'RL Global + Supervisory Constraints': 'RL Global',
        'Uniform (baseline)': 'Uniform (base)',
        'Uniform + Constraints': 'Unif. + Constr.',
        'Proportional + Constraints': 'Prop. + Constr.',
        'Rule-Based + Constraints': 'Rule-Based',
        'MPC + Constraints': 'MPC',
    }
    labels = [_SHORT_LABELS.get(r.label, r.label) for r, _, _ in all_scenarios]
    costs_all = [r.total_cost for r, _, _ in all_scenarios]
    imports_all = [r.grid_import_kwh for r, _, _ in all_scenarios]
    colors_all = [c for _, c, _ in all_scenarios]
    hatches_all = [h for _, _, h in all_scenarios]
    n_scen = len(all_scenarios)
    y2 = np.arange(n_scen)
    bar_h2 = 0.55

    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(W1, H_ROW * 2.6),
                                       gridspec_kw={'wspace': 0.55})

    # --- F2a: Total cost (horizontal) ---
    bars_c = ax2a.barh(y2, costs_all, bar_h2, color=colors_all, edgecolor='black',
                        linewidth=0.5)
    for bar, h in zip(bars_c, hatches_all):
        bar.set_hatch(h)
    ax2a.set_yticks(y2)
    ax2a.set_yticklabels(labels, fontsize=9)
    ax2a.set_xlabel('Total Cost (€)', fontsize=10)
    ax2a.set_title('Weekly Community Energy Cost', fontsize=11)
    ax2a.invert_yaxis()
    for bar, c in zip(bars_c, costs_all):
        # Place annotation inside the bar (right-aligned) for clean layout
        ax2a.annotate(f'{c:.1f}',
                      xy=(bar.get_width() - 1, bar.get_y() + bar.get_height() / 2),
                      xytext=(-6, 0), textcoords='offset points',
                      ha='right', va='center', fontsize=9, fontweight='bold',
                      color='white')
    ax2a.margins(x=0.02)

    # --- F2b: Grid import (horizontal) ---
    bars_i = ax2b.barh(y2, imports_all, bar_h2, color=colors_all, edgecolor='black',
                        linewidth=0.5)
    for bar, h in zip(bars_i, hatches_all):
        bar.set_hatch(h)
    ax2b.set_yticks(y2)
    ax2b.set_yticklabels(labels, fontsize=9)
    ax2b.set_xlabel('Grid Import (kWh)', fontsize=10)
    ax2b.set_title('Weekly Grid Energy Imported', fontsize=11)
    ax2b.invert_yaxis()
    for bar, im in zip(bars_i, imports_all):
        ax2b.annotate(f'{im:.0f}',
                      xy=(bar.get_width() - 1, bar.get_y() + bar.get_height() / 2),
                      xytext=(-6, 0), textcoords='offset points',
                      ha='right', va='center', fontsize=9, fontweight='bold',
                      color='white')
    ax2b.margins(x=0.02)

    fig2.tight_layout()
    _save(fig2, 'F2a_weekly_cost')
    _save(fig2, 'F2b_weekly_import')

    # =====================================================================
    # F3 — Per-Site Cost: RL vs Uniform Baseline
    # =====================================================================
    group_a = [(r, c) for r, c in [(res_rl, C_RL), (res_uni_base, C_UNI_B)] if r is not None]
    if group_a:
        fig3, ax3 = plt.subplots(figsize=(W05, H_ROW * 2.0))
        x_s = np.arange(n_sites)
        bw3 = 0.7 / max(len(group_a), 1)
        for j, (r, clr) in enumerate(group_a):
            sc = [sum(s.Cost_history) for s in r.sites]
            xj = x_s - 0.35 + (j + 0.5) * bw3
            bars = ax3.bar(xj, sc, width=bw3, color=clr, edgecolor='black',
                           linewidth=0.4, label=r.label)
            for bar, v in zip(bars, sc):
                ax3.annotate(f'{v:.1f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                             ha='center', va='bottom', fontsize=6.5)
        ax3.set_xticks(x_s)
        ax3.set_xticklabels([f'Site {i+1}' for i in range(n_sites)])
        ax3.set_ylabel('Cost per Site (€)')
        ax3.set_title('RL vs Uniform Baseline (no supervisory constraints)')
        ax3.legend(loc='upper right', framealpha=0.9, edgecolor='0.8')
        fig3.tight_layout()
        _save(fig3, 'F3_site_cost_vs_baseline')

    # =====================================================================
    # F4 — Per-Site Cost: RL vs Static-Constrained Strategies
    # =====================================================================
    group_b = [(r, c) for r, c in [(res_rl, C_RL), (res_uni, C_UNI), (res_prop, C_PROP)] if r is not None]
    if group_b:
        fig4, ax4 = plt.subplots(figsize=(W05, H_ROW * 2.0))
        x_s = np.arange(n_sites)
        bw4 = 0.7 / max(len(group_b), 1)
        for j, (r, clr) in enumerate(group_b):
            sc = [sum(s.Cost_history) for s in r.sites]
            xj = x_s - 0.35 + (j + 0.5) * bw4
            bars = ax4.bar(xj, sc, width=bw4, color=clr, edgecolor='black',
                           linewidth=0.4, label=r.label)
            for bar, v in zip(bars, sc):
                ax4.annotate(f'{v:.1f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                             ha='center', va='bottom', fontsize=6.5)
        ax4.set_xticks(x_s)
        ax4.set_xticklabels([f'Site {i+1}' for i in range(n_sites)])
        ax4.set_ylabel('Cost per Site (€)')
        ax4.set_title('RL vs Static-Constraint Strategies')
        ax4.legend(loc='upper right', framealpha=0.9, edgecolor='0.8')
        fig4.tight_layout()
        _save(fig4, 'F4_site_cost_vs_constraints')

    # =====================================================================
    # F5: per-participant cost for RL, Rule-Based, and MPC.
    # =====================================================================
    group_c = [(r, c) for r, c in [(res_rl, C_RL), (res_rules, C_RULES), (res_mpc, C_MPC)] if r is not None]
    if group_c:
        fig5, ax5 = plt.subplots(figsize=(W05, H_ROW * 2.4))
        x_s = np.arange(n_sites)
        bw5 = 0.7 / max(len(group_c), 1)
        for j, (r, clr) in enumerate(group_c):
            sc = [sum(s.Cost_history) for s in r.sites]
            xj = x_s - 0.35 + (j + 0.5) * bw5
            bars = ax5.bar(xj, sc, width=bw5, color=clr, edgecolor='black',
                           linewidth=0.4, label=r.label)
            for bar, v in zip(bars, sc):
                ax5.annotate(f'{v:.1f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                             ha='center', va='bottom', fontsize=6.5)
        ax5.set_xticks(x_s)
        ax5.set_xticklabels([f'Site {i+1}' for i in range(n_sites)])
        ax5.set_ylabel('Cost per Site (€)')
        ax5.set_title('RL vs Advanced Strategies (Rule-Based and MPC)')
        # Add ~20 % headroom above tallest bar so upper-right legend doesn't overlap
        y_top = ax5.get_ylim()[1]
        ax5.set_ylim(top=y_top * 1.20)
        ax5.legend(loc='upper right', framealpha=0.9, edgecolor='0.8')
        fig5.tight_layout()
        _save(fig5, 'F5_site_cost_vs_advanced')

    # =====================================================================
    # F3-COMBINED — All scenarios per-site cost in a single grouped bar chart
    # =====================================================================
    combo_scenarios = [(r, c) for r, c, _h in all_scenarios if r is not None]
    if combo_scenarios:
        fig_combo, ax_combo = plt.subplots(figsize=(W1, H_ROW * 2.5))
        x_s = np.arange(n_sites)
        n_sc = len(combo_scenarios)
        bw_combo = 0.8 / max(n_sc, 1)
        for j, (r, clr) in enumerate(combo_scenarios):
            sc = [sum(s.Cost_history) for s in r.sites]
            xj = x_s - 0.4 + (j + 0.5) * bw_combo
            bars = ax_combo.bar(xj, sc, width=bw_combo, color=clr,
                                edgecolor='black', linewidth=0.4, label=r.label)
            for bar, v in zip(bars, sc):
                ax_combo.annotate(f'{v:.0f}',
                                  xy=(bar.get_x() + bar.get_width() / 2,
                                      bar.get_height()),
                                  ha='center', va='bottom', fontsize=5.5)
        ax_combo.set_xticks(x_s)
        ax_combo.set_xticklabels([f'Site {i+1}' for i in range(n_sites)],
                                  fontsize=10)
        ax_combo.set_ylabel('Cost per Site (€)', fontsize=10)
        ax_combo.set_title('Per-Site Cost Comparison Across All Strategies',
                           fontsize=11)
        y_top = ax_combo.get_ylim()[1]
        ax_combo.set_ylim(top=y_top * 1.15)
        ax_combo.legend(loc='upper right', framealpha=0.9, edgecolor='0.8',
                        fontsize=7, ncol=2)
        fig_combo.tight_layout()
        _save(fig_combo, 'F3_site_cost_all')

    # =====================================================================
    # F6 — Cost Savings Relative to Uniform Baseline (%)
    # =====================================================================
    fig6, ax6 = plt.subplots(figsize=(W05, H_ROW * 2.2))
    savings_data = []
    for r, clr, htch in all_scenarios:
        if '(baseline)' in r.label:
            continue
        pct = (r.total_cost - baseline_cost) / baseline_cost * 100
        savings_data.append((r.label, pct, clr, htch))
    savings_data.sort(key=lambda x: x[1])
    s_labels = [s[0] for s in savings_data]
    s_pcts = [s[1] for s in savings_data]
    s_colors = [s[2] for s in savings_data]
    s_hatches = [s[3] for s in savings_data]

    x6 = np.arange(len(savings_data))
    bars6 = ax6.barh(x6, s_pcts, color=s_colors, edgecolor='black', linewidth=0.5, height=0.55)
    for bar, h in zip(bars6, s_hatches):
        bar.set_hatch(h)
    ax6.set_yticks(x6)
    ax6.set_yticklabels(s_labels)
    ax6.set_xlabel('Cost Change vs Uniform Baseline (%)')
    ax6.set_title('Weekly Cost Savings Relative to Baseline')
    ax6.axvline(0, color='black', linewidth=0.8)
    # Place percentage inside the bar (white text) to avoid clipping labels
    for bar, pct in zip(bars6, s_pcts):
        if abs(pct) < 0.5:
            # Near-zero bar: place text to the right of the baseline
            text_x = 0.8
            ha = 'left'
            txt_color = 'black'
        else:
            text_x = pct / 2  # center of the bar
            ha = 'center'
            txt_color = 'white' if abs(pct) > 3 else 'black'
        ax6.annotate(f'{pct:+.1f}%', xy=(text_x, bar.get_y() + bar.get_height() / 2),
                     va='center', ha=ha, fontsize=7, fontweight='bold',
                     color=txt_color)
    # Add margin so y-labels are not clipped
    ax6.margins(x=0.05)
    fig6.tight_layout()
    _save(fig6, 'F6_savings_pct')

    # =====================================================================
    # F7 — Battery SOC Evolution: RL vs MPC vs Rule-Based (3×3 grid)
    # =====================================================================
    soc_scenarios = [(r, lbl, clr) for r, lbl, clr in [
        (res_rl, 'RL Global', C_RL), (res_mpc, 'MPC', C_MPC), (res_rules, 'Rule-Based', C_RULES),
    ] if r is not None]

    if soc_scenarios:
        n_soc = len(soc_scenarios)
        fig7, axes7 = plt.subplots(n_sites, n_soc, figsize=(W1, H_ROW * n_sites),
                                   sharex=True, sharey='row')
        if n_soc == 1:
            axes7 = axes7.reshape(-1, 1)
        if n_sites == 1:
            axes7 = axes7.reshape(1, -1)

        for col, (res, lbl, clr) in enumerate(soc_scenarios):
            steps_s = len(res.sites[0].Pg_history)
            t_s = np.arange(steps_s) * dt
            for row in range(n_sites):
                ax = axes7[row, col]
                site = res.sites[row]
                soc = np.array(site.battery.SOC_history[:-1]) if len(site.battery.SOC_history) > steps_s else np.array(site.battery.SOC_history)
                t_plot = t_s[:len(soc)]
                ax.plot(t_plot, soc[:len(t_plot)], color=SITE_COLORS[row], linewidth=1.0)
                ax.axhline(config.soc_min, color='#c0392b', linestyle='--', linewidth=0.6, alpha=0.6)
                ax.axhline(config.soc_max, color='#c0392b', linestyle='--', linewidth=0.6, alpha=0.6)
                ax.fill_between(t_plot, config.soc_min, config.soc_max, alpha=0.05, color='green')
                ax.set_ylim(0, 100)
                if row == 0:
                    ax.set_title(lbl)
                if col == 0:
                    cap = config.battery_capacities[row] if row < len(config.battery_capacities) else 250
                    ax.set_ylabel(f'Site {row+1} ({int(cap)} Ah)\nSOC (%)')
                if row == n_sites - 1:
                    ax.set_xlabel('Time (h)')
                _day_lines(ax, t_plot[-1] if len(t_plot) else 0)

        fig7.suptitle('Battery State of Charge (SOC) — 1-Week Simulation', fontsize=10, fontweight='bold', y=1.01)
        fig7.tight_layout()
        _save(fig7, 'F7_soc_comparison')

    # =====================================================================
    # F8 — Assigned PV vs Load per Site (RL scenario)
    # =====================================================================
    if res_rl:
        fig8, axes8 = plt.subplots(n_sites, 1, figsize=(W1, H_ROW * n_sites + 0.5), sharex=True)
        if n_sites == 1:
            axes8 = [axes8]

        steps_f8 = len(res_rl.sites[0].Pg_history)
        t_f8 = np.arange(steps_f8) * dt

        for i, (ax, site) in enumerate(zip(axes8, res_rl.sites)):
            pv_hist = np.array(site.pv.PV_history[:steps_f8]) if hasattr(site.pv, 'PV_history') else np.zeros(steps_f8)
            load_hist = np.array(site.load.Load_history[:steps_f8]) if hasattr(site.load, 'Load_history') else np.zeros(steps_f8)
            pg_hist = np.array(site.Pg_history[:steps_f8])

            ax.fill_between(t_f8, 0, pv_hist, alpha=0.30, color='#27AE60', label='PV Assigned')
            ax.plot(t_f8, load_hist, color='#E74C3C', linewidth=0.9, label='Load Demand', alpha=0.9)
            ax.plot(t_f8, np.maximum(pg_hist, 0), color='#2C3E50', linewidth=0.9, label='Grid Import', alpha=0.9)
            ax.set_ylabel(f'Site {i+1}\nPower (W)')
            ax.legend(loc='upper right', fontsize=7, framealpha=0.9, edgecolor='0.8')

            if i == 0:
                ax2 = ax.twinx()
                price_idx = np.array([int(s * dt) % len(price_profile) for s in range(steps_f8)])
                ax2.plot(t_f8, price_profile[price_idx], color='#8E44AD', linewidth=0.8, alpha=0.4, linestyle=':')
                ax2.set_ylabel('Price (€/kWh)', color='#8E44AD')
                ax2.tick_params(axis='y', labelcolor='#8E44AD')

            _day_lines(ax, config.sim_hours)

        # Explicit x-axis: 0 to sim_hours with ticks every 24 h
        axes8[-1].set_xlim(0, config.sim_hours)
        day_ticks = np.arange(0, config.sim_hours + 1, 24)
        axes8[-1].set_xticks(day_ticks)
        axes8[-1].set_xlabel('Time (h)')
        fig8.suptitle('RL Scenario: PV Generation vs Load Demand (1-Week)', fontsize=10, fontweight='bold', y=1.01)
        fig8.tight_layout()
        _save(fig8, 'F8_pv_vs_load_rl')

    # =====================================================================
    # F9 — Fairness Metrics: Jain Index and PV Ratio Deviation
    # =====================================================================
    fig9, (ax9a, ax9b) = plt.subplots(2, 1, figsize=(W05, H_ROW * 2.8))

    eq_data = [(r.label, r.fairness_metrics.get('jain_index', 0),
                r.fairness_metrics.get('deviation_from_pv_ratio', 0),
                clr, htch)
               for r, clr, htch in all_scenarios]
    eq_labels = [d[0] for d in eq_data]
    eq_jain = [d[1] for d in eq_data]
    eq_dev = [d[2] for d in eq_data]
    eq_colors = [d[3] for d in eq_data]
    eq_hatches = [d[4] for d in eq_data]
    x9 = np.arange(len(eq_data))

    bars9a = ax9a.bar(x9, eq_jain, 0.55, color=eq_colors, edgecolor='black', linewidth=0.4)
    for bar, h in zip(bars9a, eq_hatches):
        bar.set_hatch(h)
    ax9a.set_xticks(x9)
    ax9a.set_xticklabels(eq_labels, rotation=35, ha='right')
    ax9a.set_ylabel("Jain's Fairness Index")
    ax9a.set_title("Jain's Fairness Index (higher is fairer)")
    ax9a.set_ylim(0.96, 1.005)
    for bar, j in zip(bars9a, eq_jain):
        ax9a.annotate(f'{j:.4f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                      ha='center', va='bottom', fontsize=6.5)

    bars9b = ax9b.bar(x9, eq_dev, 0.55, color=eq_colors, edgecolor='black', linewidth=0.4)
    for bar, h in zip(bars9b, eq_hatches):
        bar.set_hatch(h)
    ax9b.set_xticks(x9)
    ax9b.set_xticklabels(eq_labels, rotation=35, ha='right')
    ax9b.set_ylabel('Mean |β − PV ratio|')
    ax9b.set_title('Mean Deviation from Fair PV Share (lower is better)')
    for bar, d in zip(bars9b, eq_dev):
        ax9b.annotate(f'{d:.4f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                      ha='center', va='bottom', fontsize=6.5)

    fig9.tight_layout()
    _save(fig9, 'F9_fairness_metrics')

    print(f"\n[EC_5] {len(list(fig_dir.glob('*.pdf')))} figures saved to {fig_dir}")
    plt.show()


def print_summary(results: List[ScenarioResult], config: CommunitySimConfig, pv_ratios: np.ndarray) -> None:
    """Print aggregate and participant-level strategy comparison metrics."""
    print("\n" + "=" * 100)
    print("RESUMEN COMPARATIVO")
    print("=" * 100)
    
    # Main comparison table.
    header = f"{'Escenario':<30} | {'Coste (€)':>10} | {'Import (kWh)':>12} | {'Jain Index':>10} | {'Δ vs Uniforme':>12} | {'Tiempo (s)':>10}"
    print(header)
    print("-" * 110)
    
    # Find the Uniform baseline result.
    uniform_cost = next((r.total_cost for r in results if "Uniform" in r.label and "baseline" in r.label), results[0].total_cost)
    
    for res in results:
        jain = res.fairness_metrics.get("jain_index", 0)
        delta = ((res.total_cost - uniform_cost) / uniform_cost) * 100
        delta_str = f"{delta:+.2f}%" if abs(delta) > 0.001 else "baseline"
        print(f"{res.label:<30} | {res.total_cost:>10.2f} | {res.grid_import_kwh:>12.2f} | {jain:>10.4f} | {delta_str:>12} | {res.elapsed_seconds:>10.2f}")
    
    print("=" * 110)
    
    # PV reference ratios.
    print(f"\nPV RATIOS (distribución justa basada en PV instalado):")
    for i, ratio in enumerate(pv_ratios):
        print(f"  Site {i+1}: {ratio:.3f} ({ratio*100:.1f}%)")
    
    # Participant-level metrics.
    print("\n" + "-" * 100)
    print("DETALLE POR SITE:")
    print("-" * 100)
    
    for res in results:
        print(f"\n  {res.label}:")
        mean_betas = np.array(res.fairness_metrics["mean_betas"])
        std_betas = np.array(res.fairness_metrics["std_betas"])
        
        print(f"    {'Site':<6} | {'Coste':>8} | {'Import':>10} | {'SOC fin':>7} | {'β_mean':>7} | {'β_std':>6} | {'PV ratio':>8} | {'Δβ':>7}")
        print(f"    {'-'*6}-+-{'-'*8}-+-{'-'*10}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}-+-{'-'*8}-+-{'-'*7}")
        
        for i, site in enumerate(res.sites):
            cost = sum(site.Cost_history)
            grid = sum(max(pg, 0) for pg in site.Pg_history) * config.delta_t / 1000.0
            soc_final = site.battery.actual_SOC()
            delta_beta = mean_betas[i] - pv_ratios[i]
            print(f"    Site {i+1} | {cost:>7.2f}€ | {grid:>9.2f}kWh | {soc_final:>6.1f}% | {mean_betas[i]:>7.3f} | {std_betas[i]:>6.3f} | {pv_ratios[i]:>8.3f} | {delta_beta:>+7.3f}")
        
        # Aggregate metrics.
        total_deviation = np.abs(mean_betas - pv_ratios).sum()
        max_deviation = np.abs(mean_betas - pv_ratios).max()
        print(f"    Desv. total de PV ratio: {total_deviation:.4f} | Desv. máxima: {max_deviation:.4f}")
    
    # Compare the RL strategy with the selected benchmarks.
    print("\n" + "=" * 100)
    print("ANÁLISIS DE ESTRATEGIAS")
    print("=" * 100)
    
    rl_result = next((r for r in results if "RL" in r.label), None)
    uniform_result = next((r for r in results if "Uniform" in r.label), None)
    prop_result = next((r for r in results if "Proportional" in r.label), None)
    rule_result = next((r for r in results if "Rule" in r.label), None)
    mpc_result = next((r for r in results if "MPC" in r.label), None)
    
    if rl_result and uniform_result:
        print(f"\n  RL vs Uniform (baseline):")
        print(f"    Cost: {rl_result.total_cost:.2f}€ vs {uniform_result.total_cost:.2f}€ ({(rl_result.total_cost - uniform_result.total_cost):+.2f}€)")
        print(f"    Import: {rl_result.grid_import_kwh:.2f}kWh vs {uniform_result.grid_import_kwh:.2f}kWh")
        
        # Beta variability
        rl_betas = np.array(rl_result.beta_history)
        uni_betas = np.array(uniform_result.beta_history)
        print(f"    Beta variability (std total): RL={rl_betas.std():.4f} vs Uniform={uni_betas.std():.4f}")
    
    if rl_result and prop_result:
        print(f"\n  RL vs Proportional:")
        print(f"    Cost: {rl_result.total_cost:.2f}€ vs {prop_result.total_cost:.2f}€ ({(rl_result.total_cost - prop_result.total_cost):+.2f}€)")
        
        rl_deviation = rl_result.fairness_metrics.get("deviation_from_pv_ratio", 0)
        prop_deviation = prop_result.fairness_metrics.get("deviation_from_pv_ratio", 0)
        print(f"    PV ratio deviation: RL={rl_deviation:.4f} vs Prop={prop_deviation:.4f}")
    
    if rl_result and rule_result:
        print(f"\n  RL vs Rule-Based:")
        print(f"    Cost: {rl_result.total_cost:.2f}€ vs {rule_result.total_cost:.2f}€ ({(rl_result.total_cost - rule_result.total_cost):+.2f}€)")
        print(f"    Import: {rl_result.grid_import_kwh:.2f}kWh vs {rule_result.grid_import_kwh:.2f}kWh")
        rule_deviation = rule_result.fairness_metrics.get("deviation_from_pv_ratio", 0)
        print(f"    PV ratio deviation: RL={rl_deviation:.4f} vs Rule-Based={rule_deviation:.4f}")
    
    if rl_result and mpc_result:
        print(f"\n  RL vs MPC:")
        print(f"    Cost: {rl_result.total_cost:.2f}€ vs {mpc_result.total_cost:.2f}€ ({(rl_result.total_cost - mpc_result.total_cost):+.2f}€)")
        print(f"    Import: {rl_result.grid_import_kwh:.2f}kWh vs {mpc_result.grid_import_kwh:.2f}kWh")
        mpc_deviation = mpc_result.fairness_metrics.get("deviation_from_pv_ratio", 0)
        print(f"    PV ratio deviation: RL={rl_deviation:.4f} vs MPC={mpc_deviation:.4f}")
    
    # Correlate RL allocation with participant energy needs.
    if rl_result and rl_result.need_history:
        print("\n" + "-" * 100)
        print("ANÁLISIS DE DECISIONES RL: ¿Responde a las necesidades de los sites?")
        print("-" * 100)
        
        betas = np.array(rl_result.beta_history)
        needs = np.array(rl_result.need_history)  # Demand minus PV, positive for an energy deficit.
        socs = np.array(rl_result.soc_history)
        
        # Align allocation and component-history lengths before correlation.
        min_len = min(len(betas), len(needs))
        betas = betas[:min_len]
        needs = needs[:min_len]
        socs = socs[:min_len]
        
        print(f"\n  Correlaciones (β vs indicadores de necesidad):")
        print(f"  {'Site':<8} | {'β vs Need':>12} | {'β vs (1-SOC)':>12} | {'Interpretación':<30}")
        print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*30}")
        
        for i in range(config.n_sites):
            # Correlation between allocation and demand minus PV.
            beta_site = betas[:, i]
            need_site = needs[:, i]
            soc_site = socs[:, i]
            
            # Normalize series for correlation.
            if beta_site.std() > 0 and need_site.std() > 0:
                corr_need = np.corrcoef(beta_site, need_site)[0, 1]
            else:
                corr_need = 0.0
            
            # Correlate allocation with inverse SOC.
            inv_soc = 100 - soc_site
            if beta_site.std() > 0 and inv_soc.std() > 0:
                corr_soc = np.corrcoef(beta_site, inv_soc)[0, 1]
            else:
                corr_soc = 0.0
            
            # Classify the correlation values.
            if corr_need > 0.3:
                interp = "✓ Responde a necesidad"
            elif corr_need < -0.3:
                interp = "✗ Inverso a necesidad!"
            else:
                interp = "~ Sin correlación clara"
            
            print(f"  Site {i+1}  | {corr_need:>+12.3f} | {corr_soc:>+12.3f} | {interp:<30}")
        
        # Compare temporal changes in allocation and energy needs.
        print(f"\n  Análisis de cambios de beta:")
        beta_changes = np.abs(np.diff(betas, axis=0)).sum(axis=1)  # Total allocation change at each step.
        need_changes = np.abs(np.diff(needs, axis=0)).sum(axis=1)  # Change in energy needs.
        
        if beta_changes.std() > 0 and need_changes.std() > 0:
            corr_changes = np.corrcoef(beta_changes, need_changes)[0, 1]
        else:
            corr_changes = 0.0
        
        print(f"    Corr(Δβ, Δnecesidades) = {corr_changes:+.3f}")
        if corr_changes > 0.3:
            print(f"    → ✓ El RL cambia betas cuando cambian las necesidades")
        elif corr_changes < -0.1:
            print(f"    → ✗ El RL cambia betas cuando NO debería")
        else:
            print(f"    → ~ Cambios de beta no correlacionados con necesidades")
        
        # Allocation during high-demand intervals.
        print(f"\n  Análisis de momentos de alta necesidad:")
        total_need = needs.sum(axis=1)  # Total community energy need.
        high_need_threshold = np.percentile(total_need, 75)
        high_need_mask = total_need > high_need_threshold
        
        if high_need_mask.any():
            # Compare participant allocations in high- and low-need intervals.
            for i in range(config.n_sites):
                beta_high = betas[high_need_mask, i].mean()
                beta_low = betas[~high_need_mask, i].mean()
                need_high = needs[high_need_mask, i].mean()
                need_low = needs[~high_need_mask, i].mean()
                print(f"    Site {i+1}: β_alta_dem={beta_high:.3f}, β_baja_dem={beta_low:.3f}, Δβ={beta_high-beta_low:+.3f}")
    
    print("\n" + "=" * 100)


# ---------------------------------------------------------------------------
# Text report export.
# ---------------------------------------------------------------------------

def export_results_txt(
    results: List[ScenarioResult],
    config: CommunitySimConfig,
    pv_ratios: np.ndarray,
    site_profiles: List[Dict[str, np.ndarray]],
    price_profile: np.ndarray,
    output_path: Path,
) -> None:
    """Export detailed simulation metrics to a text report."""
    lines: List[str] = []
    W = 120  # Report line width.

    def L(text: str = "") -> None:
        lines.append(text)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L("=" * W)
    L(f"EC_5  —  RESULTADOS COMPLETOS DE SIMULACIÓN")
    L(f"Generado: {timestamp}")
    L("=" * W)

    # Configuration.
    L("")
    L("CONFIGURACIÓN DE SIMULACIÓN")
    L("-" * W)
    L(f"  N sites:           {config.n_sites}")
    L(f"  Paso temporal:     {config.delta_t} h ({config.delta_t*60:.0f} min)")
    L(f"  Duración:          {config.sim_hours} h ({config.sim_hours/24:.1f} días)")
    L(f"  Pasos totales:     {int(config.sim_hours / config.delta_t)}")
    L(f"  Action repeat:     {config.action_repeat} pasos ({config.delta_t * config.action_repeat:.2f} h)")
    L(f"  PV scale:          {config.pv_scale}")
    L(f"  SOC rango:         [{config.soc_min}%, {config.soc_max}%]")
    L(f"  SOC inicial:       {config.initial_soc}%")
    L(f"  Ib rango:          [{config.battery_ibmin}, {config.battery_ibmax}] A")
    L(f"  Voltaje batería:   {config.battery_voltage} V")
    L(f"  Semilla aleatoria: (fijada en ejecución)")
    L("")
    L("  Baterías por site:")
    for i in range(config.n_sites):
        cap = config.battery_capacities[i] if i < len(config.battery_capacities) else 250.0
        off = config.site_load_offsets_h[i] if i < len(config.site_load_offsets_h) else 0.0
        scl = config.site_load_scales[i] if i < len(config.site_load_scales) else 1.0
        csv_name = site_profiles[i].get("csv_name", "?") if i < len(site_profiles) else "?"
        L(f"    Site {i+1}: {cap:.0f} Ah | Load offset={off:+.0f}h, scale={scl:.1f}x | CSV: {csv_name}")
    L("")
    L(f"  PV ratios (instalado): {', '.join(f'S{i+1}={r:.4f}' for i, r in enumerate(pv_ratios))}")
    L(f"  Precio medio:          {float(np.mean(price_profile)):.4f} €/kWh")
    L(f"  Precio min:            {float(np.min(price_profile)):.4f} €/kWh")
    L(f"  Precio max:            {float(np.max(price_profile)):.4f} €/kWh")

    # Summary table.
    L("")
    L("=" * W)
    L("TABLA RESUMEN COMPARATIVA")
    L("=" * W)
    uniform_cost = next((r.total_cost for r in results if "Uniform" in r.label and "baseline" in r.label), results[0].total_cost)
    hdr = f"{'Escenario':<30} | {'Coste (€)':>10} | {'Import (kWh)':>12} | {'Jain':>8} | {'Δ vs Uni':>10} | {'Tiempo':>10}"
    L(hdr)
    L("-" * W)
    for res in results:
        jain = res.fairness_metrics.get("jain_index", 0)
        delta = ((res.total_cost - uniform_cost) / uniform_cost) * 100
        delta_str = f"{delta:+.2f}%" if abs(delta) > 0.001 else "baseline"
        t_str = f"{res.elapsed_seconds:.2f}s"
        L(f"{res.label:<30} | {res.total_cost:>10.2f} | {res.grid_import_kwh:>12.2f} | {jain:>8.4f} | {delta_str:>10} | {t_str:>10}")
    L("=" * W)
    total_time = sum(r.elapsed_seconds for r in results)
    L(f"  Tiempo total de simulación: {total_time:.2f}s")

    # Participant metrics by strategy.
    L("")
    L("=" * W)
    L("DETALLE POR SITE Y ESCENARIO")
    L("=" * W)
    for res in results:
        L(f"\n  {res.label}  (t={res.elapsed_seconds:.2f}s)")
        mean_betas = np.array(res.fairness_metrics["mean_betas"])
        std_betas = np.array(res.fairness_metrics["std_betas"])
        L(f"    {'Site':<6} | {'Coste':>9} | {'Import':>11} | {'SOC ini':>8} | {'SOC fin':>8} | {'β_mean':>8} | {'β_std':>7} | {'PV ratio':>8} | {'Δβ':>8}")
        L(f"    {'-'*6}-+-{'-'*9}-+-{'-'*11}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}")
        for i, site in enumerate(res.sites):
            cost = sum(site.Cost_history)
            grid = sum(max(pg, 0) for pg in site.Pg_history) * config.delta_t / 1000.0
            soc_ini = site.battery.SOC_history[0] if site.battery.SOC_history else config.initial_soc
            soc_fin = site.battery.actual_SOC()
            delta_b = mean_betas[i] - pv_ratios[i]
            L(f"    Site {i+1} | {cost:>8.2f}€ | {grid:>10.2f}kWh | {soc_ini:>7.1f}% | {soc_fin:>7.1f}% | {mean_betas[i]:>8.4f} | {std_betas[i]:>7.4f} | {pv_ratios[i]:>8.4f} | {delta_b:>+8.4f}")
        total_dev = np.abs(mean_betas - pv_ratios).sum()
        max_dev = np.abs(mean_betas - pv_ratios).max()
        L(f"    Desv. total de PV ratio: {total_dev:.4f} | Desv. máxima: {max_dev:.4f}")

    # Paired strategy comparisons.
    L("")
    L("=" * W)
    L("COMPARACIONES PAREADAS")
    L("=" * W)
    rl_result = next((r for r in results if "RL" in r.label), None)
    for other_label in ["Uniform", "Proportional", "Rule", "MPC"]:
        other = next((r for r in results if other_label in r.label and "(baseline)" not in r.label), None)
        if rl_result and other:
            dc = rl_result.total_cost - other.total_cost
            di = rl_result.grid_import_kwh - other.grid_import_kwh
            rl_dev = rl_result.fairness_metrics.get("deviation_from_pv_ratio", 0)
            ot_dev = other.fairness_metrics.get("deviation_from_pv_ratio", 0)
            L(f"\n  RL vs {other.label}:")
            L(f"    Coste:        {rl_result.total_cost:.2f}€ vs {other.total_cost:.2f}€  ({dc:+.2f}€ / {dc/max(other.total_cost,1e-6)*100:+.1f}%)")
            L(f"    Importación:  {rl_result.grid_import_kwh:.2f} vs {other.grid_import_kwh:.2f} kWh  ({di:+.2f} kWh)")
            L(f"    Jain:         {rl_result.fairness_metrics.get('jain_index',0):.4f} vs {other.fairness_metrics.get('jain_index',0):.4f}")
            L(f"    Desv PV ratio:{rl_dev:.4f} vs {ot_dev:.4f}")
            L(f"    Tiempo:       {rl_result.elapsed_seconds:.2f}s vs {other.elapsed_seconds:.2f}s")

    # RL allocation correlation metrics.
    if rl_result and rl_result.need_history:
        L("")
        L("=" * W)
        L("ANÁLISIS DE DECISIONES RL")
        L("=" * W)
        betas = np.array(rl_result.beta_history)
        needs = np.array(rl_result.need_history)
        socs = np.array(rl_result.soc_history)
        min_len = min(len(betas), len(needs))
        betas, needs, socs = betas[:min_len], needs[:min_len], socs[:min_len]

        L(f"\n  Correlaciones (β vs indicadores de necesidad):")
        L(f"  {'Site':<8} | {'β vs Need':>12} | {'β vs (1-SOC)':>12}")
        L(f"  {'-'*8}-+-{'-'*12}-+-{'-'*12}")
        for i in range(config.n_sites):
            b_i, n_i, s_i = betas[:, i], needs[:, i], socs[:, i]
            corr_n = np.corrcoef(b_i, n_i)[0, 1] if b_i.std() > 0 and n_i.std() > 0 else 0.0
            corr_s = np.corrcoef(b_i, 100 - s_i)[0, 1] if b_i.std() > 0 and s_i.std() > 0 else 0.0
            L(f"  Site {i+1}  | {corr_n:>+12.4f} | {corr_s:>+12.4f}")

        beta_ch = np.abs(np.diff(betas, axis=0)).sum(axis=1)
        need_ch = np.abs(np.diff(needs, axis=0)).sum(axis=1)
        corr_ch = np.corrcoef(beta_ch, need_ch)[0, 1] if beta_ch.std() > 0 and need_ch.std() > 0 else 0.0
        L(f"\n  Corr(Δβ, Δnecesidades) = {corr_ch:+.4f}")

        total_need = needs.sum(axis=1)
        hn_thr = np.percentile(total_need, 75)
        hn_mask = total_need > hn_thr
        if hn_mask.any():
            L(f"\n  Momentos de alta necesidad (P75):")
            for i in range(config.n_sites):
                bh = betas[hn_mask, i].mean()
                bl = betas[~hn_mask, i].mean()
                L(f"    Site {i+1}: β_alta={bh:.4f}, β_baja={bl:.4f}, Δβ={bh-bl:+.4f}")

    # SOC statistics by strategy.
    L("")
    L("=" * W)
    L("ESTADÍSTICAS SOC POR ESCENARIO Y SITE")
    L("=" * W)
    for res in results:
        L(f"\n  {res.label}:")
        L(f"    {'Site':<6} | {'SOC mean':>9} | {'SOC std':>8} | {'SOC min':>8} | {'SOC max':>8} | {'% < 20':>7} | {'% > 80':>7}")
        L(f"    {'-'*6}-+-{'-'*9}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}")
        for i, site in enumerate(res.sites):
            soc_arr = np.array(site.battery.SOC_history, dtype=float)
            pct_low = float(np.mean(soc_arr < 20.0)) * 100
            pct_high = float(np.mean(soc_arr > 80.0)) * 100
            L(f"    Site {i+1} | {soc_arr.mean():>8.2f}% | {soc_arr.std():>7.2f}% | {soc_arr.min():>7.1f}% | {soc_arr.max():>7.1f}% | {pct_low:>6.1f}% | {pct_high:>6.1f}%")

    # Allocation statistics.
    L("")
    L("=" * W)
    L("ESTADÍSTICAS DE DISTRIBUCIÓN DE BETAS")
    L("=" * W)
    for res in results:
        b_arr = np.array(res.beta_history)
        L(f"\n  {res.label}:")
        L(f"    {'Site':<6} | {'β mean':>8} | {'β std':>7} | {'β min':>7} | {'β max':>7} | {'β P10':>7} | {'β P90':>7}")
        L(f"    {'-'*6}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
        for i in range(config.n_sites):
            bi = b_arr[:, i]
            L(f"    Site {i+1} | {bi.mean():>8.4f} | {bi.std():>7.4f} | {bi.min():>7.4f} | {bi.max():>7.4f} | {np.percentile(bi,10):>7.4f} | {np.percentile(bi,90):>7.4f}")

    L("")
    L("=" * W)
    L("FIN DEL INFORME")
    L("=" * W)

    # Write the report file.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[EC_5] Resultados exportados a: {output_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

# Policy resources: label, policy file, normalizer file, and normalizer fallback.
_KNOWN_MODELS = [
    ("PPO",  "ppo_community.zip",  "vecnorm_ppo_community.pkl",  "vecnorm_community_stats.pkl"),
]

# Supervisory and allocation-only policy directories.
_MODELS_DIR_NORMAL = ROOT / "main/models/supervisory"
_MODELS_DIR_RESTRICTED = ROOT / "main/models/allocation_only"


def _infer_vecnorm_path(policy_path: Path, preferred: Optional[Path] = None) -> Optional[Path]:
    """Infer a VecNormalize file colocated with a policy path."""
    if preferred and preferred.is_file():
        return preferred

    policy_dir = policy_path.parent
    stem = policy_path.stem.lower()
    for algo, _, vecnorm_file, vecnorm_fallback in _KNOWN_MODELS:
        if stem.startswith(algo.lower()):
            candidates = [policy_dir / vecnorm_file]
            if vecnorm_fallback:
                candidates.append(policy_dir / vecnorm_fallback)
            for candidate in candidates:
                if candidate.is_file():
                    return candidate

    fallback = policy_dir / "vecnorm_community_stats.pkl"
    return fallback if fallback.is_file() else None


def _default_model_spec(mode: str) -> RLModelSpec:
    """Return the policy and normalizer paths for the requested control mode."""
    if mode == "restricted":
        policy_path = _MODELS_DIR_RESTRICTED / "ppo_community.zip"
        vecnorm_path = _infer_vecnorm_path(policy_path)
        return RLModelSpec(
            mode=mode,
            label="RL Global (restricted)",
            policy_path=policy_path,
            vecnorm_path=vecnorm_path,
            use_hints=False,
        )

    policy_path = _MODELS_DIR_NORMAL / "ppo_community.zip"
    vecnorm_path = _infer_vecnorm_path(policy_path)
    return RLModelSpec(
        mode="full",
        label="RL Global",
        policy_path=policy_path,
        vecnorm_path=vecnorm_path,
        use_hints=True,
    )


def _resolve_model_specs(args: argparse.Namespace) -> List[RLModelSpec]:
    """Resolve configured allocation-only and supervisory policy resources."""
    if args.model_mode == "none":
        if args.policy_path:
            raise SystemExit("--policy-path cannot be combined with --model-mode none")
        return []

    if args.policy_path and args.model_mode == "both":
        raise SystemExit("--policy-path cannot be combined with --model-mode both")

    if args.policy_path:
        policy_path = Path(args.policy_path)
        is_restricted = args.restricted or args.model_mode == "restricted" or "restricted" in str(policy_path.parent).lower()
        vecnorm_hint = Path(args.vecnorm_path) if args.vecnorm_path else None
        vecnorm_path = _infer_vecnorm_path(policy_path, vecnorm_hint)
        return [
            RLModelSpec(
                mode="restricted" if is_restricted else "full",
                label="RL Global (restricted)" if is_restricted else "RL Global",
                policy_path=policy_path,
                vecnorm_path=vecnorm_path,
                use_hints=not is_restricted,
            )
        ]

    if args.model_mode == "both":
        return [_default_model_spec("full"), _default_model_spec("restricted")]
    return [_default_model_spec(args.model_mode)]




def parse_args():
    raise SystemExit("Use main/scripts/evaluate.py")


def _resolve_mpc_modes(mode: str) -> List[str]:
    if mode == "both":
        return ["beta-sequence", "supervisory"]
    if mode == "none":
        return []
    if mode not in ("beta-sequence", "supervisory"):
        raise ValueError("Only published MPC modes are supported")
    return [mode]


def _run_single(args: argparse.Namespace, seed: int, make_plots: bool) -> List[ScenarioResult]:
    config = CommunitySimConfig(
        n_sites=args.n_sites,
        delta_t=args.delta_t,
        sim_hours=args.hours,
        forecast_horizon_hours=args.forecast_horizon_hours,
        price_window_hours=args.price_window_hours,
    )
    
    Device.delta_t(config.delta_t)
    
    rng = np.random.default_rng(seed)
    
    csv_dir = Path(args.csv_dir)
    mat_path = Path(args.mat_path)
    
    csv_dir = inside(csv_dir)
    mat_path = inside(mat_path)
    if not csv_dir.is_dir() or not mat_path.is_file():
        raise FileNotFoundError("Required repository data are missing")

    steps = int(config.sim_hours / config.delta_t)
    
    print(f"[EC_5] Cargando perfiles para {config.n_sites} sites...")
    site_profiles, pv_total, price_profile, pv_ratios, base_cost_profile = load_community_profiles(
        n_sites=config.n_sites,
        csv_dir=csv_dir,
        mat_path=mat_path,
        delta_t=config.delta_t,
        steps=steps,
        pv_scale=config.pv_scale,
        rng=rng,
        site_load_offsets_h=config.site_load_offsets_h,
        site_load_scales=config.site_load_scales,
        csv_paths=getattr(args, "csv_paths", profile_files("validation")),
    )
    
    print(f"[EC_5] CSVs usados:")
    for i, p in enumerate(site_profiles):
        off = config.site_load_offsets_h[i] if i < len(config.site_load_offsets_h) else 0.0
        scl = config.site_load_scales[i] if i < len(config.site_load_scales) else 1.0
        print(f"  Site {i+1}: {p['csv_name']}  (offset={off:+.0f}h, scale={scl:.1f}x)")
    print(f"[EC_5] PV ratios: {pv_ratios}")
    
    results: List[ScenarioResult] = []

    # Evaluate the selected allocation-only and supervisory RL policies.
    
    for spec in _resolve_model_specs(args):
        if spec.label not in args.selected_labels:
            continue
        if spec.use_hints:
            print(f"[EC_5] Modelo FULL: use_hints=True (acción=betas+dr+sr)")
        else:
            print(f"[EC_5] Modelo RESTRINGIDO: use_hints=False (acción=solo betas)")

        if spec.policy_path.is_file():
            rl_strategy = create_rl_beta_strategy(
                policy_path=spec.policy_path,
                vecnorm_path=spec.vecnorm_path,
                n_sites=config.n_sites,
                pv_ratios=pv_ratios,
                min_beta_ratio=config.min_beta_ratio,
            )
            res_rl = run_scenario(
                label=spec.label,
                site_profiles=site_profiles,
                pv_total=pv_total,
                price_profile=price_profile,
                pv_ratios=pv_ratios,
                config=config,
                beta_strategy=rl_strategy,
                base_cost_profile=base_cost_profile,
                use_hints=spec.use_hints,
            )
            results.append(res_rl)
        else:
            print(f"[EC_5] No se encontró política RL en {spec.policy_path}")
    
    static_dr = float(np.clip(args.mpc_static_dr, 0.0, 1.0))
    static_sr = float(np.clip(args.mpc_static_sr, 0.0, 1.0))

    # Uniform allocation without supervisory constraints.
    if 'Uniform (baseline)' in args.selected_labels:
        res_uniform = run_scenario(
            label="Uniform (baseline)",
            site_profiles=site_profiles,
            pv_total=pv_total,
            price_profile=price_profile,
            pv_ratios=pv_ratios,
            config=config,
            beta_strategy=uniform_beta_strategy,
            base_cost_profile=base_cost_profile,
        )
        results.append(res_uniform)
    
    # Uniform allocation with fixed supervisory constraints.
    if 'Uniform + Constraints' in args.selected_labels:
        constrained_uniform = create_static_supervisory_strategy(
            uniform_beta_strategy, discharge_rate=static_dr, soc_reserve=static_sr,
        )
        res_constrained_uni = run_scenario(
            label="Uniform + Constraints",
            site_profiles=site_profiles,
            pv_total=pv_total,
            price_profile=price_profile,
            pv_ratios=pv_ratios,
            config=config,
            beta_strategy=constrained_uniform,
            base_cost_profile=base_cost_profile,
        )
        results.append(res_constrained_uni)
    
    # Proportional allocation with fixed supervisory constraints.
    if 'Proportional + Constraints' in args.selected_labels:
        constrained_prop = create_static_supervisory_strategy(
            proportional_beta_strategy, discharge_rate=static_dr, soc_reserve=static_sr,
        )
        res_constrained_prop = run_scenario(
            label="Proportional + Constraints",
            site_profiles=site_profiles,
            pv_total=pv_total,
            price_profile=price_profile,
            pv_ratios=pv_ratios,
            config=config,
            beta_strategy=constrained_prop,
            base_cost_profile=base_cost_profile,
        )
        results.append(res_constrained_prop)
    
    # Heuristic allocation with fixed supervisory constraints.
    if 'Rule-Based' in args.selected_labels:
        rule_base = create_rule_based_beta_strategy(
            n_sites=config.n_sites,
            pv_ratios=pv_ratios,
            w_need=0.4, w_soc=0.3, w_price=0.3,
        )
        constrained_rule = create_static_supervisory_strategy(
            rule_base, discharge_rate=static_dr, soc_reserve=static_sr,
        )
        res_rule = run_scenario(
            label="Rule-Based",
            site_profiles=site_profiles,
            pv_total=pv_total,
            price_profile=price_profile,
            pv_ratios=pv_ratios,
            config=config,
            beta_strategy=constrained_rule,
            base_cost_profile=base_cost_profile,
        )
        results.append(res_rule)
    
    # Published MPC modes: beta-sequence and supervisory
    label_by_mode = {
        "beta-sequence": "MPC beta-sequence",
        "supervisory": "MPC supervisory",
    }
    for mpc_mode in _resolve_mpc_modes(args.mpc_mode):
        if label_by_mode[mpc_mode] not in args.selected_labels:
            continue
        mpc_strategy = create_mpc_beta_strategy(
            site_profiles=site_profiles,
            pv_total=pv_total,
            price_profile=price_profile,
            config=config,
            pv_ratios=pv_ratios,
            horizonte_horas=args.mpc_horizon_hours,
            peso_equidad=args.mpc_equity_weight,
            discharge_rate_hint=static_dr,
            soc_reserve_hint=static_sr,
            mode=mpc_mode,
            maxiter=args.mpc_maxiter,
        )
        if mpc_mode != "supervisory":
            mpc_strategy = create_static_supervisory_strategy(
                mpc_strategy, discharge_rate=static_dr, soc_reserve=static_sr,
            )
        res_mpc = run_scenario(
            label=label_by_mode[mpc_mode],
            site_profiles=site_profiles,
            pv_total=pv_total,
            price_profile=price_profile,
            pv_ratios=pv_ratios,
            config=config,
            beta_strategy=mpc_strategy,
            base_cost_profile=base_cost_profile,
        )
        results.append(res_mpc)
    
    # Print summary metrics.
    print_summary(results, config, pv_ratios)
    
    # Plot the selected results.
    if make_plots:
        plot_comparison(results, config, pv_ratios, pv_total, price_profile)
    
    return results


def _output_path_from_args(args: argparse.Namespace) -> Optional[Path]:
    if not args.output_report or args.no_export_report:
        return None
    output_path = Path(args.output_report)
    if not output_path.is_absolute():
        output_path = inside(output_path)
    return output_path


def print_aggregate_summary(all_runs: List[List[ScenarioResult]]) -> None:
    """Print metric means and standard deviations across evaluation runs."""
    if not all_runs:
        return

    labels: List[str] = []
    for run in all_runs:
        for res in run:
            if res.label not in labels:
                labels.append(res.label)

    print("\n" + "=" * 118)
    print("RESUMEN MULTIWEEK / MULTISEED")
    print("=" * 118)
    print(f"{'Escenario':<30} | {'Coste mean±std (€)':>22} | {'Import mean±std':>22} | {'Jain mean±std':>17} | {'Tiempo mean±std':>17}")
    print("-" * 118)

    for label in labels:
        selected = [next((r for r in run if r.label == label), None) for run in all_runs]
        selected = [r for r in selected if r is not None]
        if not selected:
            continue
        costs = np.array([r.total_cost for r in selected], dtype=float)
        imports = np.array([r.grid_import_kwh for r in selected], dtype=float)
        jains = np.array([r.fairness_metrics.get("jain_index", 0.0) for r in selected], dtype=float)
        times = np.array([r.elapsed_seconds for r in selected], dtype=float)
        print(
            f"{label:<30} | "
            f"{costs.mean():>8.2f}±{costs.std(ddof=0):<8.2f} | "
            f"{imports.mean():>8.2f}±{imports.std(ddof=0):<8.2f} | "
            f"{jains.mean():>6.4f}±{jains.std(ddof=0):<6.4f} | "
            f"{times.mean():>6.2f}±{times.std(ddof=0):<6.2f}"
        )
    print("=" * 118)


def export_aggregate_results_txt(all_runs: List[List[ScenarioResult]], output_path: Path) -> None:
    """Export aggregate strategy metrics as a compact text table."""
    labels: List[str] = []
    for run in all_runs:
        for res in run:
            if res.label not in labels:
                labels.append(res.label)

    lines = [
        "EC_5 aggregate results",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Runs: {len(all_runs)}",
        "",
        "Scenario,Cost mean,Cost std,Import mean,Import std,Jain mean,Jain std,Beta deviation mean,Beta deviation std,Time mean,Time std",
    ]

    for label in labels:
        selected = [next((r for r in run if r.label == label), None) for run in all_runs]
        selected = [r for r in selected if r is not None]
        if not selected:
            continue
        costs = np.array([r.total_cost for r in selected], dtype=float)
        imports = np.array([r.grid_import_kwh for r in selected], dtype=float)
        jains = np.array([r.fairness_metrics.get("jain_index", 0.0) for r in selected], dtype=float)
        devs = np.array([r.fairness_metrics.get("deviation_from_pv_ratio", 0.0) for r in selected], dtype=float)
        times = np.array([r.elapsed_seconds for r in selected], dtype=float)
        lines.append(
            f"{label},{costs.mean():.6f},{costs.std(ddof=0):.6f},"
            f"{imports.mean():.6f},{imports.std(ddof=0):.6f},"
            f"{jains.mean():.6f},{jains.std(ddof=0):.6f},"
            f"{devs.mean():.6f},{devs.std(ddof=0):.6f},"
            f"{times.mean():.6f},{times.std(ddof=0):.6f}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[EC_5] Tabla agregada exportada a: {output_path}")


def main() -> List[ScenarioResult]:
    args = parse_args()
    if args.weeks < 1:
        raise SystemExit("--weeks debe ser >= 1")

    output_path = _output_path_from_args(args)
    all_runs: List[List[ScenarioResult]] = []
    for week_idx in range(args.weeks):
        seed = args.seed + week_idx
        if args.weeks > 1:
            print("\n" + "#" * 100)
            print(f"[EC_5] Ejecución {week_idx + 1}/{args.weeks} (seed={seed})")
            print("#" * 100)
        make_plots = (args.weeks == 1 and not args.no_plots)
        all_runs.append(_run_single(args, seed=seed, make_plots=make_plots))

    if args.weeks > 1:
        print_aggregate_summary(all_runs)
        if output_path:
            export_aggregate_results_txt(all_runs, output_path)
    elif output_path:
        # Export a report only when an explicit output path is supplied.
        # Avoid writing reports to an implicit default destination.
        export_aggregate_results_txt(all_runs, output_path)
    else:
        print("\n[EC_5] Informe no exportado: usa --output-report PATH para escribir resultados.")

    return all_runs[0]


if __name__ == "__main__":
    raise SystemExit("Use main/scripts/evaluate.py")
