"""Evaluate selected published methods. Import scientific code only on execution."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common"))
from repo_paths import inside, new_output, profile_files

def main():
    config = json.loads((ROOT / "main/configs/evaluation.json").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description="Main N=3 evaluation; scenarios 1..10 use seeds 42..51.")
    parser.add_argument("--methods", nargs="+", choices=list(config["methods"]), default=list(config["methods"]))
    parser.add_argument("--scenarios", nargs="+", type=int, choices=range(1, 11), default=list(range(1, 11)))
    parser.add_argument("--output", required=True, help="New repository-relative subdirectory of main/outputs")
    cli = parser.parse_args()
    if len(set(cli.methods)) != len(cli.methods) or len(set(cli.scenarios)) != len(cli.scenarios):
        parser.error("Repeated methods/scenarios are not allowed")
    labels = [config["methods"][key] for key in cli.methods]
    for key, folder in [("supervisory_ppo", "supervisory"), ("allocation_ppo", "allocation_only")]:
        if key in cli.methods:
            for filename in ("ppo_community.zip", "vecnorm_community_stats.pkl"):
                if not inside(f"main/models/{folder}/{filename}").is_file():
                    raise FileNotFoundError(f"Missing paired policy resource: {folder}/{filename}")
    profiles = profile_files("validation")
    prices = inside("data/weekly_base_data.mat")
    if not prices.is_file():
        raise FileNotFoundError(prices)
    output = new_output(cli.output)
    args = SimpleNamespace(**{k: v for k, v in config.items() if k not in ("methods", "scenario_seeds")},
        model_mode="both", mpc_mode="both", policy_path=None, vecnorm_path=None,
        restricted=False, csv_dir=str(ROOT / "data/profiles"), mat_path=str(prices),
        csv_paths=profiles, selected_labels=labels)
    # These imports are intentionally deferred. --help does not load scientific modules.
    from scripts.EC_5 import _run_single, export_aggregate_results_txt
    from export_metrics import export_weekly_rows, export_paired_summary, export_battery_proxy_summary
    seeds = [config["scenario_seeds"][str(i)] for i in cli.scenarios]
    (output / "selection.json").write_text(json.dumps(dict(
        methods=cli.methods, scenarios=cli.scenarios, seeds=seeds,
        configuration=config, profile_files=[p.name for p in profiles]), indent=2), encoding="utf-8")
    runs = [_run_single(args, seed=seed, make_plots=False) for seed in seeds]
    export_aggregate_results_txt(runs, output / "aggregate.txt")
    export_weekly_rows(runs, seeds, output / "weekly.csv")
    export_paired_summary(runs, output / "paired.csv")
    export_battery_proxy_summary(runs, seeds, args, output / "battery.csv")

if __name__ == "__main__":
    main()
