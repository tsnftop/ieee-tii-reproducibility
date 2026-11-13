"""Community environments and supervisory control for scalability assessment."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common"))
sys.path.insert(0, str(ROOT / "scalability"))
from paths import SCALE, inside, partition_files
import numpy as np
import scipy.optimize
import torch
from stable_baselines3 import PPO
from RL.community_env import CommunityEnv, CommunityEnvConfig
from lib.data_loader import load_weekly_base_profiles
from RL.train_community import CommunityTrainingConfig, _make_lr_schedule
from scripts import EC_5 as ec5
from decoder import project_allocation
from corrected_mpc import create_mpc_beta_strategy as corrected_mpc

def site_parameters(n):
    if n not in (3, 5, 10):
        raise ValueError("Supported N: 3, 5, 10")
    return dict(
        n_sites=n,
        battery_capacities=[(350., 150., 250.)[i % 3] for i in range(n)],
        site_load_offsets_h=[(0., 8., 16.)[i % 3] for i in range(n)],
        site_load_scales=[(1.2, .8, 1.)[i % 3] for i in range(n)],
    )

def env_config(n):
    return CommunityEnvConfig(**site_parameters(n),
        csv_dir="data/profiles", mat_path="data/weekly_base_data.mat")

def sim_config(n):
    return ec5.CommunitySimConfig(**site_parameters(n))

class CheckedEnv(CommunityEnv):
    """Community environment with observation, action, and constraint checks."""
    def __init__(self, config, allocation_decoder="corrected"):
        self.allocation_decoder = allocation_decoder
        if allocation_decoder != "corrected":
            raise ValueError(allocation_decoder)
        n = config.n_sites
        for key in ("battery_capacities", "site_load_offsets_h", "site_load_scales"):
            assert len(getattr(config, key)) == n, key
        super().__init__(config)
        assert self.observation_space.shape == (16*n+21,)
        assert self.action_space.shape == (3*n,)

    def _deltas_to_betas(self, deltas):
        proposal = self._pv_ratios + np.clip(np.asarray(deltas, np.float32), -1, 1)*self.config.max_beta_deviation
        return project_allocation(proposal).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        assert np.isfinite(obs).all()
        assert obs.shape == self.observation_space.shape
        return obs, info

    def step(self, action):
        assert np.asarray(action).shape == self.action_space.shape
        assert np.isfinite(action).all()
        obs, reward, terminated, truncated, info = super().step(action)
        assert np.isfinite(obs).all() and np.isfinite(reward)
        assert obs.shape == self.observation_space.shape
        beta = np.asarray(info["betas"])
        assert abs(beta.sum()-1) < 1e-5 and (beta >= 0).all()
        info["beta_min_violations"] = int(np.sum(beta < .01-1e-7))
        if self.allocation_decoder == "corrected":
            assert info["beta_min_violations"] == 0
        for key in ("discharge_rates", "soc_reserves"):
            a = np.asarray(info[key])
            assert a.shape == (self.config.n_sites,)
            assert np.isfinite(a).all() and (a >= 0).all() and (a <= 1).all()
        socs = np.array([s.battery.actual_SOC() for s in self._sites])
        assert np.isfinite(socs).all()
        assert (socs >= self.config.soc_min-1e-5).all()
        assert (socs <= self.config.soc_max+1e-5).all()
        return obs, reward, terminated, truncated, info

    def _load_all_profiles(self) -> None:
        csv_dir = self._resolve_csv_dir()
        mat_path = self._resolve_mat_path()
        csv_files = partition_files(getattr(self.config, "profile_split", "TRAIN"))
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

            # Apply participant demand time offsets and amplitude scales.
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

        # Downsample total load to hourly for hourly diagnostics
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
