"""Aggregate saved main-study rows only. Never imports the simulator."""
import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common"))
from repo_paths import inside, new_output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="main/reference", help="Repository-relative saved-result directory")
    parser.add_argument("--output", required=True, help="New subdirectory of main/outputs")
    cli = parser.parse_args()
    source = inside(cli.input)
    labels = set(json.loads((ROOT / "main/configs/evaluation.json").read_text(encoding="utf-8"))["methods"].values())
    files = {
        "weekly": "EC_5_weekly_validation_rows.csv",
        "paired": "EC_5_paired_summary.csv",
        "battery": "EC_5_battery_proxy_summary.csv",
    }
    datasets = {}
    provenance = {}
    for kind, reference_name in files.items():
        path = source / (reference_name if (source / reference_name).is_file() else kind + ".csv")
        path = inside(path)
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            fields = reader.fieldnames
        def keep(row):
            if kind == "paired":
                return row["comparison"] in {"RL Global vs " + label for label in labels if label != "RL Global"}
            return row["scenario"] in labels
        selected = [row for row in rows if keep(row)]
        datasets[kind] = (fields, selected)
        provenance[kind] = {"source": str(path.relative_to(ROOT)),
                            "input_rows": len(rows), "kept_rows": len(selected),
                            "excluded_rows": [row for row in rows if not keep(row)]}
    if not datasets["weekly"][1]:
        raise ValueError("No published methods found")
    output = new_output(cli.output)
    for kind, (fields, rows) in datasets.items():
        with (output / (kind + ".csv")).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)  # Retain saved numeric strings without recomputation.
    rows = datasets["weekly"][1]
    order = list(dict.fromkeys(row["scenario"] for row in rows))
    lines = ["EC_5 aggregate results", "Source: saved weekly rows (rounded source precision)", "",
        "Scenario,Cost mean,Cost std,Import mean,Import std,Jain mean,Jain std,Beta deviation mean,Beta deviation std,Time mean,Time std"]
    for label in order:
        group = [row for row in rows if row["scenario"] == label]
        values = []
        for field in ("total_cost", "grid_import_kwh", "jain_index", "mean_beta_deviation", "online_eval_time_s"):
            numbers = [float(row[field]) for row in group]
            values.extend((mean(numbers), pstdev(numbers)))  # Population standard deviation (ddof=0).
        lines.append(label + "," + ",".join(f"{v:.6f}" for v in values))
    (output / "aggregate.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Battery means from saved per-week metrics; no reconstruction of trajectories.
    fields, rows = datasets["battery"]
    numeric = [f for f in fields if f not in ("run_index", "validation_seed", "scenario")]
    with (output / "battery_means.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["scenario"] + numeric)
        writer.writeheader()
        for label in dict.fromkeys(row["scenario"] for row in rows):
            group = [row for row in rows if row["scenario"] == label]
            writer.writerow(dict(scenario=label, **{f: mean(float(row[f]) for row in group) for f in numeric}))
    # Record excluded method labels without copying their numerical metrics.
    for entry in provenance.values():
        entry["excluded_labels"] = sorted({row.get("scenario", row.get("comparison")) for row in entry.pop("excluded_rows")})
    (output / "selection.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
