"""Supervisory MPC for PV allocation, battery discharge limits, and SOC reserves."""
from __future__ import annotations
from typing import Any, Callable, Optional
import numpy as np
from decoder import project_allocation

def create_mpc_beta_strategy(site_profiles: List[Dict[str, np.ndarray]], pv_total: np.ndarray, price_profile: np.ndarray, config: 'CommunitySimConfig', pv_ratios: np.ndarray, horizonte_horas: float=8.0, peso_equidad: float=0.1, discharge_rate_hint: float=0.2, soc_reserve_hint: float=0.0, mode: str='supervisory', maxiter: int=500) -> Callable:
    """Construct full-horizon supervisory MPC with three SLSQP initializations."""
    from scipy.optimize import minimize
    mode = mode.strip().lower()
    if mode != 'supervisory':
        raise ValueError(f'Modo MPC no soportado: {mode}')
    n_sites = config.n_sites
    delta_t = config.delta_t
    action_repeat = config.action_repeat
    total_sim_steps = int(config.sim_hours / delta_t)
    horizon_sim_steps = int(horizonte_horas / delta_t)
    n_blocks = max(1, int(np.ceil(horizon_sim_steps / max(action_repeat, 1))))
    beta_blocks = n_blocks
    static_dr = float(np.clip(discharge_rate_hint, 0.0, 1.0))
    static_sr = float(np.clip(soc_reserve_hint, 0.0, 1.0))
    all_loads = [np.asarray(p['load'], dtype=np.float64) for p in site_profiles]
    pv_arr = np.asarray(pv_total, dtype=np.float64)
    price_arr = np.asarray(price_profile, dtype=np.float64)
    bat_caps = np.array([config.battery_capacities[i] if i < len(config.battery_capacities) else 250.0 for i in range(n_sites)], dtype=np.float64)
    voltage = float(config.battery_voltage)
    soc_max = config.soc_max
    soc_min_base = config.soc_min
    eff_c = 0.9
    eff_d = 0.95
    Ibmax = config.battery_ibmax
    Ibmin = config.battery_ibmin
    soc_reserve_ceil = 50.0
    price_low = float(np.min(price_arr)) * 1.1
    soc_threshold = 50.0
    _call_count = [0]
    _default_beta_seq = np.tile(pv_ratios.astype(np.float64), (n_blocks, 1))
    _default_dr_seq = np.full((n_blocks, n_sites), static_dr, dtype=np.float64)
    _default_sr_seq = np.full((n_blocks, n_sites), static_sr, dtype=np.float64)
    _last_beta_seq = [_default_beta_seq.copy()]
    _last_dr_seq = [_default_dr_seq.copy()]
    _last_sr_seq = [_default_sr_seq.copy()]

    def _normalize_beta_vector(beta: np.ndarray) -> np.ndarray:
        return project_allocation(beta, 0.01)

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

    def _pack(beta_seq: np.ndarray, dr_seq: Optional[np.ndarray]=None, sr_seq: Optional[np.ndarray]=None) -> np.ndarray:
        beta_seq = _normalize_beta_sequence(beta_seq)
        parts = [beta_seq.reshape(-1)]
        if dr_seq is None:
            dr_seq = _default_dr_seq
        if sr_seq is None:
            sr_seq = _default_sr_seq
        parts.append(np.clip(np.asarray(dr_seq, dtype=np.float64), 0.0, 1.0).reshape(-1))
        parts.append(np.clip(np.asarray(sr_seq, dtype=np.float64), 0.0, 1.0).reshape(-1))
        return np.concatenate(parts).astype(np.float64)

    def _unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0)
        beta_size = n_blocks * n_sites
        beta_seq = _normalize_beta_sequence(x[:beta_size].reshape(n_blocks, n_sites))
        dr_seq = _default_dr_seq.copy()
        sr_seq = _default_sr_seq.copy()
        dr_start = beta_size
        sr_start = dr_start + beta_size
        dr_seq = np.clip(x[dr_start:sr_start].reshape(n_blocks, n_sites), 0.0, 1.0)
        sr_seq = np.clip(x[sr_start:sr_start + beta_size].reshape(n_blocks, n_sites), 0.0, 1.0)
        return (beta_seq, dr_seq, sr_seq)

    def _simulate_horizon(beta_seq: np.ndarray, dr_seq: np.ndarray, sr_seq: np.ndarray, start_step: int, current_socs: np.ndarray) -> float:
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
                surplus = pv_i - load_i
                cap_i = bat_caps[i]
                soc_i = socs[i]
                soc_min_dyn = soc_min_base + sr_values[i] * (soc_reserve_ceil - soc_min_base)
                if surplus >= 0:
                    if soc_i < soc_max:
                        Isurplus = surplus / voltage
                        Icharge_max = (soc_max - soc_i) * cap_i / (100.0 * delta_t * eff_c)
                        Ib = min(Isurplus, Icharge_max, Ibmax)
                        DSOC = Ib * delta_t * 100.0 * eff_c / cap_i
                        realDSOC = max(0.0, min(DSOC, soc_max - soc_i))
                        socs[i] += realDSOC
                        realIb = realDSOC * cap_i / (delta_t * 100.0 * eff_c)
                        Pb = voltage * realIb / eff_c
                    else:
                        Pb = 0.0
                elif soc_i > soc_min_dyn:
                    Idischarge_max = (soc_min_dyn - soc_i) * cap_i * eff_d / (100.0 * delta_t)
                    Ib_full = max(Idischarge_max, Ibmin)
                    Ib = Ib_full * dr_values[i]
                    if price_step <= price_low and soc_i < soc_threshold:
                        Icharge_max = (soc_max - soc_i) * cap_i / (100.0 * delta_t * eff_c)
                        Ib = min(Icharge_max, Ibmax)
                        DSOC = Ib * delta_t * 100.0 * eff_c / cap_i
                        realDSOC = max(0.0, min(DSOC, soc_max - soc_i))
                        socs[i] += realDSOC
                        realIb = realDSOC * cap_i / (delta_t * 100.0 * eff_c)
                        Pb = voltage * realIb / eff_c
                    else:
                        DSOC = Ib * delta_t * 100.0 / (cap_i * eff_d)
                        realDSOC = min(0.0, max(DSOC, soc_min_dyn - soc_i))
                        socs[i] += realDSOC
                        realIb = realDSOC * cap_i * eff_d / (delta_t * 100.0)
                        Pb = voltage * realIb * eff_d
                elif price_step <= price_low:
                    Icharge_max = (soc_max - soc_i) * cap_i / (100.0 * delta_t * eff_c)
                    Ib = min(Icharge_max, Ibmax)
                    DSOC = Ib * delta_t * 100.0 * eff_c / cap_i
                    realDSOC = max(0.0, min(DSOC, soc_max - soc_i))
                    socs[i] += realDSOC
                    realIb = realDSOC * cap_i / (delta_t * 100.0 * eff_c)
                    Pb = voltage * realIb / eff_c
                else:
                    Pb = 0.0
                Pg = load_i - pv_i + Pb
                Pg = max(Pg, 0.0)
                cost += Pg / 1000.0 * price_step * delta_t
        fairness_beta = beta_seq
        mean_beta = fairness_beta.mean(axis=0)
        jain = mean_beta.sum() ** 2 / (n_sites * (mean_beta ** 2).sum() + 1e-08)
        cost += (1.0 - jain) * peso_equidad * max(cost, 0.01)
        return cost

    def _constraints() -> list[dict[str, Any]]:
        constraints = []
        for block in range(beta_blocks):
            start = block * n_sites
            constraints.append({'type': 'eq', 'fun': lambda x, s=start: np.sum(x[s:s + n_sites]) - 1.0})
        return constraints

    def _bounds() -> list[tuple[float, float]]:
        beta_bounds = [(0.01, 1.0)] * (beta_blocks * n_sites)
        return beta_bounds + [(0.0, 1.0)] * (2 * n_blocks * n_sites)
        return beta_bounds

    def _candidate_vectors() -> list[np.ndarray]:
        uniform_beta_seq = np.tile(np.ones(n_sites, dtype=np.float64) / n_sites, (n_blocks, 1))
        shifted_beta = _shift_sequence(_last_beta_seq[0], pv_ratios.astype(np.float64))
        candidates = [_pack(shifted_beta, _shift_sequence(_last_dr_seq[0], _default_dr_seq[-1]), _shift_sequence(_last_sr_seq[0], _default_sr_seq[-1])), _pack(_default_beta_seq, _default_dr_seq, _default_sr_seq), _pack(uniform_beta_seq, _default_dr_seq, _default_sr_seq)]
        return candidates

    def mpc_strategy(n_sites: int, step: int, observation: np.ndarray, **kwargs) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        _call_count[0] += 1
        if _call_count[0] % 24 == 1:
            print(f'  [MPC {mode}] Decisión {_call_count[0]}, step={step}')
        current_socs = np.array([obs[4 * i] * 100.0 for i in range(n_sites)], dtype=np.float64)

        def objective(x):
            beta_seq, dr_seq, sr_seq = _unpack(x)
            return _simulate_horizon(beta_seq, dr_seq, sr_seq, step, current_socs)
        best_cost = np.inf
        best_x = _pack(_default_beta_seq, _default_dr_seq, _default_sr_seq)
        for x0 in _candidate_vectors():
            sol = minimize(objective, x0, method='SLSQP', bounds=_bounds(), constraints=_constraints(), options={'disp': False, 'maxiter': maxiter, 'ftol': 1e-07})
            if np.isfinite(sol.fun) and sol.fun < best_cost:
                best_cost = sol.fun
                best_x = sol.x.copy()
        beta_seq, dr_seq, sr_seq = _unpack(best_x)
        _last_beta_seq[0] = beta_seq.copy()
        _last_dr_seq[0] = dr_seq.copy()
        _last_sr_seq[0] = sr_seq.copy()
        betas = beta_seq[0].astype(np.float32)
        return {'betas': betas, 'discharge_rates': dr_seq[0].astype(np.float32), 'soc_reserves': sr_seq[0].astype(np.float32)}
        return betas
    mpc_strategy.allocation_sequences = _last_beta_seq
    return mpc_strategy
