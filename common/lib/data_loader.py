from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:  # pragma: no cover - optional dependency
    from scipy.io import loadmat  # type: ignore

    HAS_SCIPY = True
except Exception:  # pragma: no cover
    HAS_SCIPY = False
    loadmat = None  # type: ignore

DEFAULT_NOISE_PCT = 0.05


def default_rng() -> np.random.Generator:
    return np.random.default_rng()

@dataclass(slots=True)
class WeeklyBaseProfiles:
    pv: np.ndarray
    load: np.ndarray
    cost_hourly: np.ndarray
    csv_name: str | None
    dt_hours: float
    steps: int


def load_weekly_base_profiles(
    csv_dir: Path,
    mat_path: Path,
    *,
    delta_minutes: float = 15.0,
    steps: int | None = None,
    rng: np.random.Generator | None = None,
    csv_path: Path | None = None,
    pv_scale: float = 1.0,
) -> WeeklyBaseProfiles:
    rng = rng or default_rng()
    csv_dir = Path(csv_dir)
    mat_path = Path(mat_path)
    if steps is None or steps <= 0:
        steps = max(1, int(round(7 * 24 * 60 / max(delta_minutes, 1e-6))))
    dt_hours = max(delta_minutes, 1e-6) / 60.0

    if csv_path is None:
        candidates = _list_csv_candidates(csv_dir)
        if not candidates:
            raise RuntimeError(f"No se encontraron CSV en {csv_dir}")
        csv_path = candidates[int(rng.integers(len(candidates)))]
    else:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise RuntimeError(f"El fichero {csv_path} no existe")

    pv_raw, load_raw = _read_pv_load(csv_path)
    if pv_scale != 1.0:
        pv_raw = np.asarray(pv_raw, dtype=float) * float(pv_scale)
    pv = _normalize_series(pv_raw, steps)
    load = _normalize_series(load_raw, steps)

    episode_hours = int(round(steps * dt_hours)) or 1
    cost_hourly = _load_cost_vector(mat_path, episode_hours)

    return WeeklyBaseProfiles(
        pv=pv,
        load=load,
        cost_hourly=cost_hourly,
        csv_name=csv_path.name,
        dt_hours=dt_hours,
        steps=steps,
    )


def perturb_cost_profile(
    cost_hourly: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
    noise_pct: float = DEFAULT_NOISE_PCT,
) -> tuple[np.ndarray, dict[str, float]]:
    rng = rng or default_rng()
    base = np.asarray(cost_hourly, dtype=float).flatten()
    if base.size == 0:
        return base, {"base_mean": 0.0, "episode_mean": 0.0, "episode_min": 0.0, "episode_max": 0.0}
    amplitude = abs(noise_pct)
    noise = rng.uniform(-amplitude, amplitude, size=base.shape)
    episode = base * (1.0 + noise)
    episode = np.clip(episode, 0.0, None)
    stats = {
        "base_mean": float(np.mean(base)),
        "episode_mean": float(np.mean(episode)),
        "episode_min": float(np.min(episode)),
        "episode_max": float(np.max(episode)),
        "noise_pct": amplitude,
    }
    return episode.astype(float), stats


def expand_hourly_cost(
    cost_hourly: np.ndarray,
    *,
    dt_hours: float,
    steps: int,
) -> np.ndarray:
    hourly = np.asarray(cost_hourly, dtype=float).flatten()
    if hourly.size == 0 or steps <= 0:
        return np.zeros(max(0, steps), dtype=float)
    factor = max(1, int(round(1.0 / max(dt_hours, 1e-6))))
    expanded = np.repeat(hourly, factor)
    if expanded.size < steps:
        reps = int(np.ceil(steps / expanded.size))
        expanded = np.tile(expanded, reps)
    return expanded[:steps]


def _list_csv_candidates(csv_dir: Path) -> list[Path]:
    if not csv_dir.exists() or not csv_dir.is_dir():
        return []
    csv_files = sorted(csv_dir.glob("*.csv"))
    preferred = [p for p in csv_files if p.stem.lower().endswith("15min")]
    return preferred or csv_files


def _read_pv_load(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        raw = np.genfromtxt(
            csv_path,
            delimiter=",",
            skip_header=1,
            usecols=(2, 3),
            dtype=float,
        )
    except Exception as exc:  # type: ignore[reportGeneralTypeIssues]
        raise RuntimeError(f"No se pudo leer el CSV {csv_path}") from exc
    raw = np.atleast_2d(raw)
    if raw.shape[1] < 2:
        raise RuntimeError(
            "Formato de CSV inesperado: se esperaban al menos 2 columnas numéricas (Power_PV y Power_Load)."
        )
    pv = np.asarray(raw[:, 0], dtype=float)
    load = np.asarray(raw[:, 1], dtype=float)
    return pv, load


def _normalize_series(values: np.ndarray, steps: int) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(values, dtype=float).flatten(), nan=0.0)
    if steps <= 0:
        return np.zeros(0, dtype=float)
    if arr.size == 0:
        return np.zeros(steps, dtype=float)
    if arr.size == steps:
        return arr.copy()
    if arr.size > steps:
        ratio = arr.size / steps
        if abs(ratio - round(ratio)) < 1e-6:
            block = int(round(ratio))
            usable = block * steps
            trimmed = arr[:usable]
            reshaped = trimmed.reshape(steps, block)
            return reshaped.mean(axis=1)
        return arr[:steps].copy()
    reps = int(np.ceil(steps / arr.size))
    tiled = np.tile(arr, reps)[:steps]
    return tiled


def _load_cost_vector(mat_path: Path, hours_needed: int) -> np.ndarray:
    hours_needed = max(1, hours_needed)
    if HAS_SCIPY and mat_path.exists():
        data = loadmat(mat_path)
        arr = np.asarray(data.get("Cost", []), dtype=float).flatten()
    else:
        arr = np.array([0.2, 0.25, 0.22, 0.21], dtype=float)
    if arr.size == 0:
        arr = np.array([0.2, 0.25, 0.22, 0.21], dtype=float)
    arr = np.nan_to_num(arr, nan=0.0)
    if arr.size >= hours_needed:
        return arr[:hours_needed].copy()
    reps = int(np.ceil(hours_needed / arr.size))
    return np.tile(arr, reps)[:hours_needed]


__all__ = [
    "WeeklyBaseProfiles",
    "load_weekly_base_profiles",
    "perturb_cost_profile",
    "expand_hourly_cost",
    "DEFAULT_NOISE_PCT",
]
