"""PPO training for community PV allocation and battery supervision.

Construct vectorized environments, normalize observations and rewards,
and train the configured actor-critic policy. Callbacks record reward
components, evaluate policies, and save models and normalization statistics.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CallbackList,
    CheckpointCallback,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from repo_paths import inside

from RL.community_env import CommunityEnv, CommunityEnvConfig


# ======================================================================
# PPO training configuration.
# ======================================================================

@dataclass(slots=True)
class CommunityTrainingConfig:
    """PPO optimization, environment, and output configuration."""

    total_timesteps: int = 3_000_000
    log_dir: Optional[Path] = None
    tensorboard_log: bool = True
    tensorboard_subdir: str = "tb_community"
    policy: str = "MlpPolicy"

    # Learning-rate schedule.
    learning_rate: float = 3e-4
    lr_schedule: str = "linear"  # Constant or linearly decaying learning rate.

    # Parallel environments.
    n_envs: int = 4

    # PPO hyperparameters.
    n_steps: int = 512          # Approximately three 168-decision episodes per worker rollout.
    batch_size: int = 128       # Minibatch size for PPO updates.
    n_epochs: int = 5           # PPO optimization epochs per rollout.
    gamma: float = 0.99         # Hourly decisions over 168-step episodes.
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01      # Entropy regularization weight for action exploration.
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Actor and critic network architecture.
    policy_kwargs: dict = field(default_factory=lambda: {
        "net_arch": dict(pi=[128, 64], vf=[128, 64]),
    })

    device: str = "cpu"
    verbose: int = 1
    seed: Optional[int] = 42    # Training seed.
    save_name: str = "ppo_community"

    # Exploration through the policy distribution and optional SDE.
    use_sde: bool = False
    sde_sample_freq: int = 4    # Unused when use_sde is False.

    # PPO optimization stability.
    target_kl: float = 0.03     # Stop optimization epochs early when KL exceeds 1.5 * target_kl.

    # Policy evaluation.
    eval_freq: int = 10_000     # Evaluation interval in environment steps.
    eval_episodes: int = 5      # Number of evaluation episodes.
    checkpoint_freq: int = 50_000


# Default environment configuration.
DEFAULT_ENV_CONFIG = CommunityEnvConfig(
    n_sites=3,
    delta_t_hours=0.25,
    episode_hours=24 * 7,
    battery_capacities=[350.0, 150.0, 250.0],
    # Reward components: economic advantage, curtailment, smoothing, and fairness.
    cost_weight=0.0,
    advantage_weight=100.0,
    waste_weight=5.0,        # Penalty weight for curtailed PV.
    smoothness_weight=0.05,  # Penalty weight for allocation changes.
    fairness_weight=2.0,     # Penalty weight for deviations from PV reference ratios.
    max_beta_deviation=0.6,  # Maximum residual allocation deviation.
    # Fairness
    min_beta_ratio=0.05,     # Minimum allocation share relative to each PV reference ratio.
    # Local controller with optional policy-generated supervisory constraints.
    controller_type="hinted",
    aggressive_soc_threshold=50.0,
    use_hints=True,
    soc_reserve_ceil=50.0,
    static_discharge_rate=0.2,
    static_soc_reserve=0.0,
    # Participant demand-profile time offsets and amplitude scales.
    site_load_offsets_h=[0.0, 8.0, 16.0],
    site_load_scales=[1.2, 0.8, 1.0],       # Participant demand amplitude multipliers.
    # PV
    pv_scale=2.0,
    # Electricity-price perturbations.
    price_daily_variation=0.2,
    forecast_horizon_hours=8.0,
    price_window_hours=8.0,
)

DEFAULT_TRAINING_CONFIG = CommunityTrainingConfig()


# ======================================================================
# Callbacks
# ======================================================================

class CommunityRewardLogger(BaseCallback):
    """Record episode reward components and related control metrics."""

    def __init__(self, print_every: int = 2048, verbose: int = 1):
        super().__init__(verbose)
        self.print_every = max(1, print_every)
        self._accum: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._steps = 0
        self._episodes_completed = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, info in enumerate(infos or []):
            if not isinstance(info, dict):
                continue
            for key, value in info.items():
                if key.startswith("reward_") or key in (
                    "total_cost", "raw_cost", "waste_fraction",
                    "waste_ratio_raw", "advantage_eur", "cost_ema",
                ):
                    try:
                        v = float(value)
                    except Exception:
                        continue
                    self._accum[key] = self._accum.get(key, 0.0) + v
                    self._counts[key] = self._counts.get(key, 0) + 1

            # Count completed episodes.
            if i < len(dones) and dones[i]:
                self._episodes_completed += 1

        self._steps += 1

        if self._steps % self.print_every == 0 and self._counts:
            means = {
                k: self._accum[k] / max(1, self._counts[k])
                for k in sorted(self._counts)
            }
            # Record metrics in TensorBoard.
            for k, v in means.items():
                self.logger.record(f"community/{k}", v)
            self.logger.record("community/episodes", self._episodes_completed)

            if self.verbose >= 1:
                f = lambda x: f"{x:+.4f}"
                print(
                    f"[{self.num_timesteps:>8d}] "
                    f"R_total={f(means.get('reward_total', 0))} "
                    f"R_adv={f(means.get('reward_advantage', 0))} "
                    f"R_fair={f(means.get('reward_fairness', 0))} "
                    f"R_waste={f(means.get('reward_waste', 0))} "
                    f"R_smooth={f(means.get('reward_smooth', 0))} "
                    f"| cost€={means.get('raw_cost', 0):.3f} "
                    f"waste={means.get('waste_fraction', 0):.3f} "
                    f"adv€={means.get('advantage_eur', 0):.4f}"
                )

            self._accum.clear()
            self._counts.clear()

        return True


class VecNormSaveCallback(BaseCallback):
    """Save normalization statistics periodically and with model checkpoints.

    Keep the normalization state associated with checkpointed policies
    available throughout training.
    """

    def __init__(
        self,
        save_path: str | Path,
        save_freq: int = 50_000,
        best_model_dir: str | Path | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._save_path = str(Path(save_path))
        self._save_freq = max(1, save_freq)
        self._best_model_str = str(Path(best_model_dir) / "best_model.zip") if best_model_dir else None
        self._vecnorm_best_str = str(Path(best_model_dir) / "vecnorm_community_stats.pkl") if best_model_dir else None
        self._last_best_mtime: float | None = None

    def _save_vecnorm(self, path: str) -> None:
        """Save VecNormalize statistics from the training environment."""
        try:
            env = self.training_env
            if isinstance(env, VecNormalize):
                env.save(path)
                if self.verbose >= 1:
                    print(f"VecNormalize guardado en {path}")
        except Exception as e:
            if self.verbose >= 1:
                print(f"Aviso: no se pudo guardar VecNormalize: {e}")

    def _on_step(self) -> bool:
        import os
        # Save normalization statistics alongside checkpoints.
        if self.num_timesteps % self._save_freq == 0:
            self._save_vecnorm(self._save_path)

        # Save normalization statistics when the best evaluation reward improves.
        if self._best_model_str is not None:
            try:
                if os.path.exists(self._best_model_str):
                    mtime = os.path.getmtime(self._best_model_str)
                    if self._last_best_mtime is None:
                        self._last_best_mtime = mtime
                    elif mtime > self._last_best_mtime:
                        self._last_best_mtime = mtime
                        self._save_vecnorm(self._vecnorm_best_str)
            except Exception:
                pass

        return True


class EpisodeSummaryLogger(BaseCallback):
    """Log episode summaries at the configured episode interval."""

    def __init__(self, log_every_n_episodes: int = 50, verbose: int = 1):
        super().__init__(verbose)
        self._log_every = max(1, log_every_n_episodes)
        self._episodes = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for i, done in enumerate(dones):
            if not done:
                continue
            self._episodes += 1
            if self._episodes % self._log_every != 0:
                continue

            # Obtain the underlying environment for episode summary metrics.
            try:
                env = self.training_env.envs[i]  # type: ignore
                if hasattr(env, "get_episode_summary"):
                    summary = env.get_episode_summary()
                    if summary:
                        self.logger.record("episode/cost_eur", summary.get("total_cost_eur", 0))
                        self.logger.record("episode/baseline_eur", summary.get("baseline_cost_eur", 0))
                        self.logger.record("episode/reduction_pct", summary.get("cost_reduction_pct", 0))
                        self.logger.record("episode/grid_kwh", summary.get("grid_import_kwh", 0))
                        if self.verbose >= 1:
                            print(
                                f"  [Ep {self._episodes}] "
                                f"Cost={summary.get('total_cost_eur', 0):.2f}€ "
                                f"Baseline={summary.get('baseline_cost_eur', 0):.2f}€ "
                                f"Reduction={summary.get('cost_reduction_pct', 0):.1f}% "
                                f"Grid={summary.get('grid_import_kwh', 0):.1f}kWh"
                            )
            except Exception:
                pass

        return True


# ======================================================================
# Policy training.
# ======================================================================

def _make_lr_schedule(initial_lr: float, schedule: str):
    """Construct a constant or linearly decaying learning-rate schedule."""
    if schedule == "linear":
        def lr_fn(progress: float) -> float:
            """Map remaining training progress from 1.0 to 0.0 to a learning rate."""
            return max(initial_lr * progress, 1e-6)
        return lr_fn
    return initial_lr  # Constant learning rate.


def train_community_agent(
    training_cfg: CommunityTrainingConfig | None = None,
    env_cfg: CommunityEnvConfig | None = None,
) -> Path:
    """Train a community PPO policy and save its normalization statistics."""
    cfg = training_cfg or DEFAULT_TRAINING_CONFIG
    env_config = env_cfg or DEFAULT_ENV_CONFIG

    # Construct an environment to determine observation and action dimensions.
    probe = CommunityEnv(env_config)
    steps_ep = probe.max_episode_steps
    obs_dim = probe.observation_space.shape[0]
    act_dim = probe.action_space.shape[0]
    probe.close()

    print(f"\n{'='*60}")
    print(f"  ENTRENAMIENTO RL - COMUNIDAD ENERGÉTICA (PPO)")
    print(f"{'='*60}")
    print(f"  Sites: {env_config.n_sites}")
    print(f"  Pasos/episodio: {steps_ep}")
    print(f"  Obs dim: {obs_dim}, Act dim: {act_dim}")
    print(f"  Timesteps: {cfg.total_timesteps:,}")
    print(f"  Episodios estimados: ~{cfg.total_timesteps // (steps_ep * cfg.n_envs)}")
    print(f"{'='*60}\n")

    # Create vectorized training environments.
    vec_cls = SubprocVecEnv if cfg.n_envs > 1 else DummyVecEnv
    train_vec = make_vec_env(
        CommunityEnv, n_envs=cfg.n_envs, seed=cfg.seed,
        vec_env_cls=vec_cls, env_kwargs={"config": env_config},
    )
    train_env = VecNormalize(
        train_vec, norm_obs=True, norm_reward=True,
        clip_obs=10.0, clip_reward=10.0,
    )

    # Use a single evaluation environment without subprocess vectorization.
    eval_vec = make_vec_env(
        CommunityEnv, n_envs=1, seed=(cfg.seed or 0) + 1000,
        vec_env_cls=DummyVecEnv, env_kwargs={"config": env_config},
    )
    eval_env = VecNormalize(
        eval_vec, norm_obs=True, norm_reward=False,  # Keep evaluation rewards unnormalized.
        clip_obs=10.0, training=False,
    )

    try:
        log_dir = inside(cfg.log_dir or "main/outputs/training")
        log_dir.mkdir(parents=True, exist_ok=True)

        tb_dir = log_dir / cfg.tensorboard_subdir if cfg.tensorboard_log else None
        if tb_dir:
            tb_dir.mkdir(parents=True, exist_ok=True)

        # Check minibatch size against the rollout size.
        rollout = cfg.n_envs * cfg.n_steps
        if cfg.batch_size > rollout:
            raise ValueError(
                f"batch_size ({cfg.batch_size}) > n_envs*n_steps ({rollout})"
            )

        # Construct the learning-rate schedule.
        lr = _make_lr_schedule(cfg.learning_rate, cfg.lr_schedule)

        # Report the training hyperparameters.
        print("Hiperparámetros PPO:")
        for k in [
            "learning_rate", "lr_schedule", "n_envs", "n_steps", "batch_size",
            "n_epochs", "gamma", "gae_lambda", "clip_range", "ent_coef",
            "vf_coef", "max_grad_norm", "policy_kwargs", "use_sde", "target_kl",
        ]:
            print(f"  {k}: {getattr(cfg, k)}")

        # Construct the PPO model.
        model = PPO(
            cfg.policy,
            train_env,
            learning_rate=lr,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            clip_range=cfg.clip_range,
            ent_coef=cfg.ent_coef,
            vf_coef=cfg.vf_coef,
            max_grad_norm=cfg.max_grad_norm,
            policy_kwargs=dict(cfg.policy_kwargs),
            verbose=cfg.verbose,
            device=cfg.device,
            use_sde=cfg.use_sde,
            sde_sample_freq=cfg.sde_sample_freq,
            target_kl=cfg.target_kl,
            tensorboard_log=str(tb_dir) if tb_dir else None,
            seed=cfg.seed,
        )

        # Configure training callbacks.
        reward_logger = CommunityRewardLogger(
            print_every=max(1, cfg.n_steps), verbose=cfg.verbose,
        )
        episode_logger = EpisodeSummaryLogger(
            log_every_n_episodes=20, verbose=cfg.verbose,
        )

        # Save the model with the highest evaluation reward.
        best_model_dir = log_dir / "best_community"
        best_model_dir.mkdir(parents=True, exist_ok=True)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(best_model_dir),
            log_path=str(best_model_dir),
            eval_freq=max(cfg.eval_freq // cfg.n_envs, 1),
            n_eval_episodes=cfg.eval_episodes,
            deterministic=True,
            verbose=cfg.verbose,
        )

        # Save periodic model checkpoints.
        checkpoint_cb = CheckpointCallback(
            save_freq=max(cfg.checkpoint_freq // cfg.n_envs, 1),
            save_path=str(log_dir / "checkpoints"),
            name_prefix="ppo_community",
            verbose=0,
        )

        # Save normalization statistics periodically during training.
        vecnorm_saver = VecNormSaveCallback(
            save_path=log_dir / "vecnorm_community_stats.pkl",
            save_freq=max(cfg.checkpoint_freq // cfg.n_envs, 1),
            best_model_dir=best_model_dir,
            verbose=cfg.verbose,
        )

        callbacks = CallbackList([
            reward_logger, episode_logger, eval_callback, checkpoint_cb,
            vecnorm_saver,
        ])

        # Train the policy.
        print(f"\nEntrenando {cfg.total_timesteps:,} pasos...")
        model.learn(total_timesteps=cfg.total_timesteps, callback=callbacks)

        # Save the trained policy and normalization statistics.
        save_path = log_dir / cfg.save_name
        model.save(save_path)

        vecnorm_path = log_dir / "vecnorm_community_stats.pkl"
        try:
            train_env.save(str(vecnorm_path))
            print(f"VecNormalize guardado en {vecnorm_path}")
        except Exception as e:
            print(f"Aviso: no se pudieron guardar stats de normalización: {e}")

        metadata_path = log_dir / "training_metadata.json"
        metadata = {
            "script": "RL/train_community.py",
            "algorithm": "PPO",
            "model_path": str(save_path.with_suffix(".zip")),
            "vecnorm_path": str(vecnorm_path),
            "environment": asdict(env_config),
            "training": asdict(cfg),
            "action_space": {
                "uses_supervisory_constraints": bool(env_config.use_hints),
                "variables": "beta+dr+sr" if env_config.use_hints else "beta-only",
                "observation_dim": int(obs_dim),
                "action_dim": int(act_dim),
            },
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        print(f"Metadata de entrenamiento guardada en {metadata_path}")

        print(f"\nModelo final: {save_path.with_suffix('.zip')}")
        print(f"Mejor modelo: {best_model_dir / 'best_model.zip'}")
        if tb_dir:
            print(f"Tensorboard: {tb_dir}")

        return save_path.with_suffix(".zip")

    finally:
        train_env.close()
        eval_env.close()


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrena controlador RL de comunidad (PPO)")
    p.add_argument("--timesteps", type=int, default=3_000_000)
    p.add_argument("--n-sites", type=int, default=3)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-tensorboard", action="store_true")
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--forecast-horizon-hours", type=float, default=8.0)
    p.add_argument("--price-window-hours", type=float, default=8.0)
    p.add_argument("--static-discharge-rate", type=float, default=0.2)
    p.add_argument("--static-soc-reserve", type=float, default=0.0)
    # Reward weights
    p.add_argument("--cost-weight", type=float, default=0.0)
    p.add_argument("--advantage-weight", type=float, default=100.0)
    p.add_argument("--waste-weight", type=float, default=5.0)
    p.add_argument("--smoothness-weight", type=float, default=0.05)
    p.add_argument("--fairness-weight", type=float, default=2.0)
    p.add_argument("--max-beta-deviation", type=float, default=0.6)
    p.add_argument("--min-beta-ratio", type=float, default=0.05)
    p.add_argument("--no-hints", action="store_true",
                    help="Deprecated: usar --no-supervisory-constraints")
    p.add_argument("--no-supervisory-constraints", action="store_true",
                    help="Desactiva dr/sr: acción solo betas")
    p.add_argument("--log-dir", type=str, default=None,
                    help="Directorio donde guardar el modelo (por defecto RL/models)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    env_config = CommunityEnvConfig(
        n_sites=args.n_sites,
        delta_t_hours=0.25,
        episode_hours=24 * 7,
        battery_capacities=[350.0, 150.0, 250.0],
        cost_weight=args.cost_weight,
        advantage_weight=args.advantage_weight,
        waste_weight=args.waste_weight,
        smoothness_weight=args.smoothness_weight,
        fairness_weight=args.fairness_weight,
        max_beta_deviation=args.max_beta_deviation,
        use_hints=not (args.no_hints or args.no_supervisory_constraints),
        static_discharge_rate=args.static_discharge_rate,
        static_soc_reserve=args.static_soc_reserve,
        min_beta_ratio=args.min_beta_ratio,
        pv_scale=2.0,
        price_daily_variation=0.2,
        forecast_horizon_hours=args.forecast_horizon_hours,
        price_window_hours=args.price_window_hours,
    )

    log_dir = Path(args.log_dir) if args.log_dir else None

    training_config = CommunityTrainingConfig(
        total_timesteps=args.timesteps,
        log_dir=log_dir,
        n_envs=args.n_envs,
        learning_rate=args.lr,
        seed=args.seed,
        tensorboard_log=not args.no_tensorboard,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
    )

    save_path = train_community_agent(training_config, env_config)
    print(f"\nEntrenamiento finalizado. Modelo: {save_path}")


if __name__ == "__main__":
    main()
