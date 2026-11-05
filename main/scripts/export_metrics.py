"""Export weekly community-control metrics and battery cycling proxies.

Provide metric rows, paired strategy comparisons, and aggregate summaries
for the main benchmark evaluation.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1] / "common"
REPO_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from EC_5 import (  # noqa: E402
    CommunitySimConfig,
    ScenarioResult,
    _run_single,
    print_aggregate_summary,
)


FULL_RL_LABEL = "RL Global"
COMPARISON_LABELS = [
    "Uniform (baseline)",
    "Uniform + Constraints",
    "Proportional + Constraints",
    "Rule-Based",
    "RL Global (restricted)",
    "MPC beta-sequence",
    "MPC supervisory",
]








def _scenario_row(run_index: int, seed: int, res: ScenarioResult) -> dict[str, Any]:
    return {
        "run_index": run_index,
        "validation_seed": seed,
        "scenario": res.label,
        "total_cost": f"{res.total_cost:.9f}",
        "grid_import_kwh": f"{res.grid_import_kwh:.9f}",
        "jain_index": f"{float(res.fairness_metrics.get('jain_index', math.nan)):.9f}",
        "mean_beta_deviation": f"{float(res.fairness_metrics.get('deviation_from_pv_ratio', math.nan)):.9f}",
        "online_eval_time_s": f"{res.elapsed_seconds:.9f}",
    }


def export_weekly_rows(
    all_runs: list[list[ScenarioResult]],
    seeds: list[int],
    output_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_index, (seed, run) in enumerate(zip(seeds, all_runs), start=1):
        for res in run:
            rows.append(_scenario_row(run_index, seed, res))

    fieldnames = [
        "run_index",
        "validation_seed",
        "scenario",
        "total_cost",
        "grid_import_kwh",
        "jain_index",
        "mean_beta_deviation",
        "online_eval_time_s",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _wilcoxon_p_value(diffs: list[float]) -> tuple[float | None, str]:
    nonzero = [d for d in diffs if abs(d) > 1e-12]
    if not nonzero:
        return None, "not computed: all paired differences are zero"
    try:
        from scipy.stats import wilcoxon  # type: ignore
    except Exception:
        return None, "not computed: scipy is not available"
    try:
        stat = wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided")
    except Exception as exc:
        return None, f"not computed: {exc}"
    return float(stat.pvalue), f"exploratory Wilcoxon signed-rank test, n={len(diffs)}"


def export_paired_summary(
    all_runs: list[list[ScenarioResult]],
    output_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for other_label in COMPARISON_LABELS:
        diffs: list[float] = []
        pct_diffs: list[float] = []
        wins = 0
        n_pairs = 0
        for run in all_runs:
            by_label = {res.label: res for res in run}
            full = by_label.get(FULL_RL_LABEL)
            other = by_label.get(other_label)
            if full is None or other is None:
                continue
            diff = float(full.total_cost - other.total_cost)
            diffs.append(diff)
            pct_diffs.append(100.0 * diff / max(float(other.total_cost), 1e-12))
            wins += int(diff < 0.0)
            n_pairs += 1

        if not diffs:
            continue
        p_value, note = _wilcoxon_p_value(diffs)
        rows.append({
            "comparison": f"{FULL_RL_LABEL} vs {other_label}",
            "n_pairs": n_pairs,
            "full_rl_lower_cost_weeks": wins,
            "mean_paired_cost_diff_full_minus_other": f"{mean(diffs):.9f}",
            "std_paired_cost_diff_full_minus_other": f"{pstdev(diffs):.9f}",
            "mean_pct_diff_full_minus_other": f"{mean(pct_diffs):.9f}",
            "wilcoxon_p_value": "" if p_value is None else f"{p_value:.9g}",
            "wilcoxon_note": note,
        })

    fieldnames = [
        "comparison",
        "n_pairs",
        "full_rl_lower_cost_weeks",
        "mean_paired_cost_diff_full_minus_other",
        "std_paired_cost_diff_full_minus_other",
        "mean_pct_diff_full_minus_other",
        "wilcoxon_p_value",
        "wilcoxon_note",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _soc_for_step_metrics(site: Any, n_steps: int) -> np.ndarray:
    soc = np.asarray(getattr(site.battery, "SOC_history", []), dtype=float)
    if soc.size == n_steps + 1:
        return soc[1:]
    if soc.size >= n_steps:
        return soc[:n_steps]
    if soc.size == 0:
        return np.zeros(n_steps, dtype=float)
    return np.pad(soc, (0, n_steps - soc.size), mode="edge")


def _count_near_boundary_entries(values: np.ndarray, threshold_mask: np.ndarray) -> int:
    if threshold_mask.size == 0:
        return 0
    entries = int(threshold_mask[0])
    if threshold_mask.size > 1:
        entries += int(np.logical_and(threshold_mask[1:], ~threshold_mask[:-1]).sum())
    return entries


def _battery_proxy_row(
    run_index: int,
    seed: int,
    res: ScenarioResult,
    delta_t: float,
    soc_min: float,
    soc_max: float,
    near_soc_margin_pct: float = 2.0,
) -> dict[str, Any]:
    total_charge_kwh = 0.0
    total_discharge_kwh = 0.0
    total_throughput_kwh = 0.0
    total_capacity_kwh = 0.0
    site_hours_near_min = 0.0
    site_hours_near_max = 0.0
    boundary_hits_min = 0
    boundary_hits_max = 0
    all_soc: list[float] = []

    for site in res.sites:
        pbat = np.asarray(getattr(site.battery, "Pbat_history", []), dtype=float)
        n_steps = int(pbat.size)
        charged = float(np.maximum(pbat, 0.0).sum() * delta_t / 1000.0)
        discharged = float(np.maximum(-pbat, 0.0).sum() * delta_t / 1000.0)
        throughput = charged + discharged

        cap_ah = float(getattr(site.battery, "capacidad_bateria", 0.0))
        voltage = float(getattr(site.battery, "voltaje_bateria", 0.0))
        capacity_kwh = cap_ah * voltage / 1000.0

        soc_step = _soc_for_step_metrics(site, n_steps)
        near_min = soc_step <= (soc_min + near_soc_margin_pct)
        near_max = soc_step >= (soc_max - near_soc_margin_pct)

        total_charge_kwh += charged
        total_discharge_kwh += discharged
        total_throughput_kwh += throughput
        total_capacity_kwh += capacity_kwh
        site_hours_near_min += float(near_min.sum() * delta_t)
        site_hours_near_max += float(near_max.sum() * delta_t)
        boundary_hits_min += _count_near_boundary_entries(soc_step, near_min)
        boundary_hits_max += _count_near_boundary_entries(soc_step, near_max)
        all_soc.extend(soc_step.tolist())

    equivalent_full_cycles = (
        total_throughput_kwh / (2.0 * total_capacity_kwh)
        if total_capacity_kwh > 1e-12 else math.nan
    )
    soc_arr = np.asarray(all_soc, dtype=float)
    return {
        "run_index": run_index,
        "validation_seed": seed,
        "scenario": res.label,
        "total_abs_battery_throughput_kwh": f"{total_throughput_kwh:.9f}",
        "total_charged_energy_kwh": f"{total_charge_kwh:.9f}",
        "total_discharged_energy_kwh": f"{total_discharge_kwh:.9f}",
        "approx_equivalent_full_cycles": f"{equivalent_full_cycles:.9f}",
        "site_hours_near_soc_min": f"{site_hours_near_min:.9f}",
        "site_hours_near_soc_max": f"{site_hours_near_max:.9f}",
        "soc_min_boundary_entries": boundary_hits_min,
        "soc_max_boundary_entries": boundary_hits_max,
        "soc_min_threshold_pct": f"{soc_min + near_soc_margin_pct:.6f}",
        "soc_max_threshold_pct": f"{soc_max - near_soc_margin_pct:.6f}",
        "mean_soc_pct": f"{float(np.mean(soc_arr)):.9f}" if soc_arr.size else "",
        "min_soc_pct": f"{float(np.min(soc_arr)):.9f}" if soc_arr.size else "",
        "max_soc_pct": f"{float(np.max(soc_arr)):.9f}" if soc_arr.size else "",
    }


def export_battery_proxy_summary(
    all_runs: list[list[ScenarioResult]],
    seeds: list[int],
    args: argparse.Namespace,
    output_path: Path,
) -> list[dict[str, Any]]:
    config = CommunitySimConfig(
        n_sites=args.n_sites,
        delta_t=args.delta_t,
        sim_hours=args.hours,
        forecast_horizon_hours=args.forecast_horizon_hours,
        price_window_hours=args.price_window_hours,
    )
    rows: list[dict[str, Any]] = []
    for run_index, (seed, run) in enumerate(zip(seeds, all_runs), start=1):
        for res in run:
            rows.append(_battery_proxy_row(
                run_index=run_index,
                seed=seed,
                res=res,
                delta_t=config.delta_t,
                soc_min=config.soc_min,
                soc_max=config.soc_max,
            ))

    fieldnames = [
        "run_index",
        "validation_seed",
        "scenario",
        "total_abs_battery_throughput_kwh",
        "total_charged_energy_kwh",
        "total_discharged_energy_kwh",
        "approx_equivalent_full_cycles",
        "site_hours_near_soc_min",
        "site_hours_near_soc_max",
        "soc_min_boundary_entries",
        "soc_max_boundary_entries",
        "soc_min_threshold_pct",
        "soc_max_threshold_pct",
        "mean_soc_pct",
        "min_soc_pct",
        "max_soc_pct",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def _aggregate_from_rows(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_scenario.setdefault(str(row["scenario"]), []).append(row)

    out: dict[str, dict[str, float]] = {}
    for scenario, scenario_rows in by_scenario.items():
        costs = np.array([_float_or_nan(r["total_cost"]) for r in scenario_rows], dtype=float)
        imports = np.array([_float_or_nan(r["grid_import_kwh"]) for r in scenario_rows], dtype=float)
        jains = np.array([_float_or_nan(r["jain_index"]) for r in scenario_rows], dtype=float)
        devs = np.array([_float_or_nan(r["mean_beta_deviation"]) for r in scenario_rows], dtype=float)
        times = np.array([_float_or_nan(r["online_eval_time_s"]) for r in scenario_rows], dtype=float)
        out[scenario] = {
            "Cost mean": float(np.mean(costs)),
            "Cost std": float(np.std(costs, ddof=0)),
            "Import mean": float(np.mean(imports)),
            "Import std": float(np.std(imports, ddof=0)),
            "Jain mean": float(np.mean(jains)),
            "Jain std": float(np.std(jains, ddof=0)),
            "Beta deviation mean": float(np.mean(devs)),
            "Beta deviation std": float(np.std(devs, ddof=0)),
            "Time mean": float(np.mean(times)),
            "Time std": float(np.std(times, ddof=0)),
        }
    return out


def _load_current_report(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    header_idx = next((i for i, line in enumerate(lines) if line.startswith("Scenario,")), None)
    if header_idx is None:
        return {}
    reader = csv.DictReader(lines[header_idx:])
    out: dict[str, dict[str, float]] = {}
    for row in reader:
        scenario = row.pop("Scenario")
        out[scenario] = {key: float(value) for key, value in row.items()}
    return out


def print_report_consistency(
    weekly_rows: list[dict[str, Any]],
    current_report: Path,
    tolerance: float = 5e-6,
) -> None:
    current = _load_current_report(current_report)
    if not current:
        print(f"[export] No current aggregate report found/readable at {current_report}")
        return
    derived = _aggregate_from_rows(weekly_rows)
    deterministic_keys = [
        "Cost mean", "Cost std", "Import mean", "Import std",
        "Jain mean", "Jain std", "Beta deviation mean", "Beta deviation std",
    ]
    mismatches: list[str] = []
    for scenario, expected in current.items():
        actual = derived.get(scenario)
        if actual is None:
            mismatches.append(f"{scenario}: missing in new rows")
            continue
        for key in deterministic_keys:
            delta = abs(actual[key] - expected[key])
            if delta > tolerance:
                mismatches.append(
                    f"{scenario} {key}: derived={actual[key]:.9f}, "
                    f"current={expected[key]:.9f}, delta={delta:.3g}"
                )
    if mismatches:
        print("[export] Aggregate deterministic metrics do NOT match current report:")
        for msg in mismatches:
            print(f"  - {msg}")
    else:
        print("[export] Aggregate deterministic metrics match current report.")
    print("[export] Timing columns are not compared because they are wall-clock measurements.")


def print_short_summaries(
    paired_rows: list[dict[str, Any]],
    battery_rows: list[dict[str, Any]],
) -> None:
    print("\n[export] Paired cost win-counts:")
    for row in paired_rows:
        print(
            f"  {row['comparison']}: "
            f"{row['full_rl_lower_cost_weeks']}/{row['n_pairs']} weeks, "
            f"mean diff={float(row['mean_paired_cost_diff_full_minus_other']):+.2f} EUR"
        )

    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in battery_rows:
        by_scenario.setdefault(str(row["scenario"]), []).append(row)
    print("\n[export] Cycling proxy summary by scenario:")
    for scenario, rows in by_scenario.items():
        throughput = [float(r["total_abs_battery_throughput_kwh"]) for r in rows]
        efc = [float(r["approx_equivalent_full_cycles"]) for r in rows]
        print(
            f"  {scenario}: throughput={mean(throughput):.2f} kWh/week, "
            f"approx EFC={mean(efc):.3f}/week"
        )




if __name__ == "__main__":
    raise SystemExit("Use evaluate.py or summarize.py in this directory")
