"""Community-level RL controller for photovoltaic allocation.

Policy logits are transformed by softmax into allocation coefficients
whose sum is one. The controller supports minimum shares relative to
PV reference ratios and optional exponential smoothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol, Sequence

import numpy as np

from lib.controllers.BaseController import BaseController


class PredictablePolicy(Protocol):
    """Minimal prediction interface for a compatible RL policy."""
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, Optional[dict]]:
        ...


@dataclass
class CommunityRLControllerConfig:
    """Configuration for the community allocation controller.

    Args:
        n_sites: Number of community participants.
        deterministic_policy: Request deterministic predictions when True.
        pv_installed_ratios: PV reference shares, or None for uniform shares.
        min_beta_ratio: Minimum allocation as a fraction of each reference
            share. For example, 0.3 guarantees 30% of that share when feasible.
        beta_smoothing: Exponential smoothing factor, with zero disabling
            smoothing and one retaining the preceding allocation.
    """
    n_sites: int = 3
    deterministic_policy: bool = True
    pv_installed_ratios: Optional[List[float]] = None
    min_beta_ratio: float = 0.3
    beta_smoothing: float = 0.0
    

class CommunityRLController(BaseController):
    """Determine community PV allocation from a policy or external actions.

    An attached policy predicts actions from observations. Alternatively,
    set_action() supplies logits for the next allocation step.
    """

    def __init__(self, config: CommunityRLControllerConfig | None = None) -> None:
        super().__init__()
        self.config = config or CommunityRLControllerConfig()
        self.n_sites = self.config.n_sites
        
        # PV reference shares used for fairness constraints.
        if self.config.pv_installed_ratios is not None:
            ratios = np.array(self.config.pv_installed_ratios, dtype=np.float32)
            self._pv_ratios = ratios / ratios.sum()  # Normalize the reference shares.
        else:
            self._pv_ratios = np.ones(self.n_sites, dtype=np.float32) / self.n_sites
        
        # Minimum allocation shares derived from the fairness configuration.
        self._min_betas = self._pv_ratios * self.config.min_beta_ratio
        
        # Attached RL policy.
        self._policy: PredictablePolicy | None = None
        
        # Pending external action containing policy logits.
        self._pending_action: np.ndarray | None = None
        
        # Observation-vector builder.
        self._observation_builder: Callable[["CommunityRLController"], np.ndarray] | None = None
        
        # Most recent observation for diagnostics.
        self._last_observation: np.ndarray | None = None
        
        # Allocation-coefficient history.
        self.beta_history: List[np.ndarray] = []
        
        # Current allocation, initially uniform.
        self._current_betas = np.ones(self.n_sites, dtype=np.float32) / self.n_sites
        
        # Community reference supplied by the caller.
        self.community: Optional["Community"] = None  # type: ignore

    def attach_policy(self, policy: PredictablePolicy, *, deterministic: bool | None = None) -> None:
        """Attach a compatible RL prediction policy."""
        self._policy = policy
        if deterministic is not None:
            self.config.deterministic_policy = deterministic

    def set_observation_builder(self, builder: Callable[["CommunityRLController"], np.ndarray]) -> None:
        """Set the callable used to construct observations."""
        self._observation_builder = builder

    def set_action(self, action: np.ndarray | Sequence[float]) -> None:
        """Supply external action logits for the next allocation step."""
        self._pending_action = np.asarray(action, dtype=np.float32).flatten()

    def set_community(self, community: "Community") -> None:  # type: ignore
        """Set the community reference."""
        self.community = community

    def build_observation(self) -> np.ndarray:
        """Build an observation using the configured callable."""
        if self._observation_builder is not None:
            obs = self._observation_builder(self)
            self._last_observation = np.asarray(obs, dtype=np.float32)
            return self._last_observation
        # Use the default observation when no builder is supplied.
        return self._default_observation()

    def _default_observation(self) -> np.ndarray:
        """Concatenate the current allocation and PV reference ratios."""
        return np.concatenate([
            self._current_betas,
            self._pv_ratios,
        ]).astype(np.float32)

    def calculate_betas(self) -> np.ndarray:
        """Compute the current allocation with fairness constraints and smoothing.

        Returns:
            Normalized allocation coefficients with sum equal to one.
        """
        logits = self._get_action_logits()
        raw_betas = self._softmax(logits)
        
        # Apply the configured minimum-allocation constraints.
        betas = self._apply_fairness(raw_betas)
        
        # Apply exponential smoothing across control steps.
        if self.config.beta_smoothing > 0 and len(self.beta_history) > 0:
            alpha = self.config.beta_smoothing
            betas = alpha * self._current_betas + (1 - alpha) * betas
            betas = betas / betas.sum()  # Normalize the smoothed allocation.
        
        self._current_betas = betas
        self.beta_history.append(betas.copy())
        
        return betas

    def _get_action_logits(self) -> np.ndarray:
        """Obtain logits from the attached policy or pending external action."""
        # First priority: use the attached policy.
        if self._policy is not None:
            obs = self.build_observation()
            action, _ = self._policy.predict(obs, deterministic=self.config.deterministic_policy)
            return np.asarray(action, dtype=np.float32).flatten()
        
        # Second priority: consume a pending external action.
        if self._pending_action is not None:
            action = self._pending_action
            self._pending_action = None
            return action
        
        # Default to equal logits and hence uniform allocation.
        return np.zeros(self.n_sites, dtype=np.float32)

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Transform logits into normalized allocation weights using softmax."""
        # Subtract the maximum logit for numerical stability.
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum()

    def _apply_fairness(self, raw_betas: np.ndarray) -> np.ndarray:
        """Redistribute allocation weights to enforce the configured minima.

        If the requested minima are infeasible, normalize the available shares.
        """
        # Compute allocation deficits relative to the configured minima.
        deficit = np.maximum(self._min_betas - raw_betas, 0)
        total_deficit = deficit.sum()
        
        if total_deficit <= 0:
            return raw_betas
        
        # Redistribute mass from participants whose allocations exceed their minima.
        excess = np.maximum(raw_betas - self._min_betas, 0)
        total_excess = excess.sum()
        
        if total_excess < total_deficit:
            # Allocate the minima and distribute the remaining mass if feasible.
            remaining = 1.0 - self._min_betas.sum()
            if remaining > 0:
                # Distribute the remaining mass in proportion to raw_betas.
                extra = (raw_betas / raw_betas.sum()) * remaining
                return self._min_betas + extra
            else:
                return self._min_betas / self._min_betas.sum()
        
        # Reduce excess allocations proportionally to cover the deficits.
        reduction_factor = total_deficit / total_excess
        adjusted_betas = raw_betas - excess * reduction_factor + deficit
        
        return adjusted_betas / adjusted_betas.sum()

    def update_state(self) -> np.ndarray:
        """Compute and return the allocation for the current community step."""
        return self.calculate_betas()

    def get_current_betas(self) -> np.ndarray:
        """Return the current allocation without recomputing it."""
        return self._current_betas.copy()

    def get_fairness_metrics(self) -> dict:
        """Compute allocation fairness metrics for the current episode."""
        if not self.beta_history:
            return {}
        
        betas = np.array(self.beta_history)
        mean_betas = betas.mean(axis=0)
        std_betas = betas.std(axis=0)
        
        # Jain's fairness index
        jain_index = (mean_betas.sum() ** 2) / (self.n_sites * (mean_betas ** 2).sum())
        
        # Deviation from the PV reference allocation.
        deviation_from_fair = np.abs(mean_betas - self._pv_ratios).mean()
        
        return {
            "mean_betas": mean_betas.tolist(),
            "std_betas": std_betas.tolist(),
            "jain_fairness_index": float(jain_index),
            "deviation_from_fair_share": float(deviation_from_fair),
            "min_beta_violations": int(np.sum(betas < self._min_betas[None, :], axis=0).sum()),
        }

    def reset(self) -> None:
        """Reset allocation state and histories for a new episode."""
        self._current_betas = np.ones(self.n_sites, dtype=np.float32) / self.n_sites
        self.beta_history = []
        self._pending_action = None
        self._last_observation = None
