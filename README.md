# Photovoltaic Allocation and Battery Supervision: Reproducibility Repository

## Overview

This repository provides data, trained policies, control implementations, and saved results for coordinated photovoltaic (PV) allocation and battery supervision in energy communities. A central supervisory policy determines PV allocation coefficients (`beta`), discharge limits (`dr`), and state-of-charge reserves (`sr`). Local rule-based controllers operate the batteries.

Two complementary studies are included:

- **Main benchmark evaluation:** a three-participant community and eight control strategies, including Allocation-Only and Supervisory PPO and MPC.
- **Scalability assessment:** communities with 3, 5, and 10 participants and five control strategies, with three independently trained supervisory PPO policies per size.

The studies use distinct experimental protocols, configurations, and trained policies. Their N=3 results are not interchangeable and must not be pooled. Reconstructing the supplied numerical summaries requires neither training nor new simulations.

## Repository structure

```text
common/                  Shared simulator, environments, controllers, and training routines
data/
  profiles/              Weekly PV and demand profiles
  data_bbaa_1week.mat     Base electricity-price data
main/
  configs/               Main-study evaluation and policy configurations
  models/                Allocation-Only and Supervisory PPO policy-normalizer pairs
  reference/             Saved weekly, paired, and battery metrics
  scripts/               Evaluation, summarization, and metric-export utilities
  outputs/               Generated results
scalability/
  code/                  Scalability environment and control implementations
  configs/               Data partitions, policies, training, and evaluation settings
  inputs/                Prepared evaluation inputs and associated metadata
  models/                PPO policy-normalizer pairs by size and training seed
  reference/episodes/    Complete saved evaluation episodes and artifacts
  reference/tables/      Saved numerical and timing tables
  evaluation/            Evaluation and summarization entry points
  outputs/               Generated results
requirements.txt         Declared Python package versions
```

Run commands from the repository root. Resource paths are resolved within the repository. Every `--output` must name a **new** repository-relative directory under the corresponding study's `outputs/` directory. Existing output directories are rejected. Keep generated outputs separate from `reference/`.

## Requirements and installation

Use **Python 3.13.13** and the package versions pinned in [requirements.txt](requirements.txt): NumPy 2.3.3, SciPy 1.16.2, pandas 2.3.3, Gymnasium 0.29.1, Stable-Baselines3 2.7.0, PyTorch 2.8.0, and Matplotlib 3.10.6. The scalability evaluator explicitly uses CPU policy inference.

Create a virtual environment and install the declared dependencies.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\Activate.ps1
```

POSIX shell:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
. .venv/bin/activate
```

In subsequent commands, `python` denotes the selected virtual-environment interpreter. It can also be invoked directly without activation. No external project checkout is required. Evaluation settings and recorded software metadata are provided in [protocol.json](scalability/configs/evaluation/protocol.json).

## Data organization

The shared `data/profiles/` directory contains weekly PV and demand CSV files. Simulation uses 15-minute physical steps and hourly control decisions. `data/data_bbaa_1week.mat` supplies base price data.

For the main benchmark evaluation, [data.json](main/configs/data.json) specifies the training and evaluation profile pools. Its `validation` key identifies the evaluation pool. Scenarios 1 through 10 correspond to scenario seeds 42 through 51, as specified in [evaluation.json](main/configs/evaluation.json). A scenario seed determines the selected profiles and their stochastic perturbations.

The scalability assessment defines three separate partitions:

- **Training set:** 32 profiles in [TRAIN.json](scalability/configs/data/TRAIN.json), used to learn the policies.
- **Tuning set:** 10 profiles in [TUNING.json](scalability/configs/data/TUNING.json), used to select the final PPO configuration.
- **Evaluation set:** 10 profiles in [FINAL_TEST.json](scalability/configs/data/FINAL_TEST.json), reserved for final assessment.

Prepared scalability inputs are stored by size and scenario in `scalability/inputs/N3/`, `N5/`, and `N10/`. Preserve the complete input directories, including NPZ files, JSON metadata, and auxiliary arrays. The evaluation checks input and policy hashes. These inputs are shared across methods and PPO training seeds for paired comparisons.

## Main benchmark evaluation

The interface is [main/scripts/evaluate.py](main/scripts/evaluate.py). Its eight method identifiers are:

- `uniform`: Uniform.
- `uniform_constraints`: Uniform + Constraints.
- `proportional_constraints`: Proportional + Constraints.
- `rule_based`: Rule-Based.
- `allocation_ppo`: Allocation-Only PPO.
- `allocation_mpc`: Allocation-Only MPC.
- `supervisory_ppo`: Supervisory PPO.
- `supervisory_mpc`: Supervisory MPC.

Uniform uses `dr=1` and `sr=0`. The constrained static strategies, Rule-Based, and Allocation-Only strategies use `dr=0.2` and `sr=0`. Supervisory strategies jointly determine allocation and battery supervisory signals. The CSV labels `RL Global (restricted)` and `RL Global` identify Allocation-Only PPO and Supervisory PPO, respectively.

`--methods` accepts one or more identifiers. `--scenarios` accepts one or more integers from 1 to 10. Defaults select all eight methods and all ten scenarios. N is fixed at 3. Every selected scenario runs for 168 hours, with 168 control decisions. This interface has no `--seeds`, `--sizes`, or `--max-decisions` option.

To run the full evaluation, including both MPC methods:

```sh
python main/scripts/evaluate.py --output main/outputs/main_evaluation
```

Outputs are `selection.json`, `weekly.csv`, `battery.csv`, `paired.csv`, and `aggregate.txt`. Paired statistics compare Supervisory PPO with other selected methods. A Uniform-only evaluation produces no paired comparison rows. `aggregate.txt` reports population standard deviations (`ddof=0`) across the selected scenarios.

## Scalability assessment

The interface is [scalability/evaluation/evaluate.py](scalability/evaluation/evaluate.py). The five method identifiers are `uniform`, `proportional`, `rule_based`, `ppo` (Supervisory PPO), and `mpc` (Supervisory MPC).

- `--sizes`: one or more of `3 5 10`.
- `--methods`: one or more of the five identifiers above.
- `--seeds`: one or more PPO **training** seeds from `42 142 242`.
- `--scenarios`: one or more scenario indices from 1 to 10.
- `--max-decisions`: an integer from 1 to 168, defaulting to a complete week of 168 decisions.
- `--output`: a new directory under `scalability/outputs/`.

Defaults select all sizes, methods, training seeds, and scenarios. `--seeds` selects stored PPO policies and does not change the scenario inputs. It does not create training replicates for deterministic benchmarks. Repeated selections are rejected by both evaluation interfaces.

To run the full scalability evaluation, including MPC:

```sh
python scalability/evaluation/evaluate.py --output scalability/outputs/scalability_evaluation
```

The full design contains 210 episodes: for each size, ten scenarios for each of four deterministic benchmarks and ten scenarios for each of three PPO policies. Results are organized by size, controller, and scenario. Each episode JSON contains metrics, completion status, a `smoke` flag, and artifact hashes. Associated NPZ files contain recorded trajectories and timing arrays.

## Reconstructing published results

Use the saved references for numerical reconstruction without loading policies, running simulators, or solving optimization problems:

```sh
python main/scripts/summarize.py --output main/outputs/published_summary
python scalability/evaluation/summarize.py --output scalability/outputs/published_summary
```

The main summarizer defaults to `main/reference/`. It retains the eight published methods and produces:

- `weekly.csv`: 80 method-scenario rows.
- `aggregate.txt`: means and population standard deviations of cost, grid import, Jain index, allocation deviation, and stored evaluation time.
- `battery.csv` and `battery_means.csv`: saved per-scenario battery metrics and their means.
- `paired.csv`: seven stored comparisons, each containing ten scenario pairs.
- `selection.json`: source and filtering information.

The main summarizer **retrieves the stored paired statistics**. It does not recalculate their differences, significance tests, or p-values. Means reconstructed from weekly rows inherit the precision of those saved rows.

The scalability summarizer defaults to `scalability/reference/episodes/`. It requires all 210 complete episodes and produces `costs_by_scenario.csv`, `paired_comparisons.csv`, `ppo_by_seed.csv`, `summary_by_N_method.csv`, `fairness_cycling.csv`, and `aggregation_status.json`. Expected counts are 15 method-size summaries, nine PPO seed summaries, and 360 paired comparison rows.

PPO metrics are averaged over ten scenarios for each policy before aggregation across the three training seeds. Columns ending in `_sd_between_seeds` report sample standard deviations (`ddof=1`) of those three policy means. They do not treat thirty PPO episodes as thirty independent trainings. For deterministic benchmarks, `cost_sd_across_scenarios` is the sample standard deviation across ten scenarios. Percentage reductions are calculated per matched scenario before averaging.

Stored timing fields remain available in the reconstructed summaries. Detailed timing tables, including decision-level distributions, are supplied in `scalability/reference/tables/timing_by_decision.csv` and `timing_scalability.csv`. The default summarization does not regenerate those tables. Do not use `--rebuild-timing` with the supplied references. The bundled solver records do not satisfy the strict artifact-hash checks required for that reconstruction, so detailed solver diagnostics cannot be reconstructed with those checks from the supplied records.

To summarize complete new evaluations instead of the saved references:

```sh
python main/scripts/summarize.py --input main/outputs/main_evaluation --output main/outputs/main_evaluation_summary
python scalability/evaluation/summarize.py --input scalability/outputs/scalability_evaluation --output scalability/outputs/scalability_evaluation_summary
```

The scalability summarizer does not accept a subset of the full design or truncated smoke-test episodes.

## Running individual evaluations

A single main-study Uniform scenario, without PPO or MPC:

```sh
python main/scripts/evaluate.py --methods uniform --scenarios 1 --output main/outputs/example_uniform_01
```

A single complete main-study scenario with Uniform and Supervisory PPO:

```sh
python main/scripts/evaluate.py --methods uniform supervisory_ppo --scenarios 1 --output main/outputs/example_supervisory_01
```

A two-decision scalability evaluation using the N=5 policy trained with seed 142:

```sh
python scalability/evaluation/evaluate.py --sizes 5 --methods ppo --seeds 142 --scenarios 1 --max-decisions 2 --output scalability/outputs/example_ppo_smoke
```

Truncated scalability episodes are marked `smoke: true`. Their costs do not reproduce weekly metrics. For a complete week, omit `--max-decisions` or set it to 168. Reducing the number of decisions does not change MPC optimizer settings. The main-study interface always retains the complete week.

Argument descriptions can be inspected without starting evaluations:

```sh
python main/scripts/evaluate.py --help
python main/scripts/summarize.py --help
python scalability/evaluation/evaluate.py --help
python scalability/evaluation/summarize.py --help
```

## Trained policies and normalization

The repository contains eleven PPO policy-normalizer pairs:

- Main Allocation-Only PPO: `main/models/allocation_only/`.
- Main Supervisory PPO: `main/models/supervisory/`.
- Scalability Supervisory PPO: `scalability/models/N{N}/seed_{seed}/` for N=3,5,10 and training seeds 42,142,242.

Each directory contains `ppo_community.zip` and its corresponding `vecnorm_community_stats.pkl`. Never exchange normalizers between policies, sizes, or studies. Evaluation uses deterministic policy predictions and does not update normalization statistics. [models.json](scalability/configs/models.json) records the nine scalability pairs, their dimensions, and hashes.

Main-policy observation/action dimensions are 63/3 for Allocation-Only PPO and 69/9 for Supervisory PPO. Scalability dimensions are `(16*N + 21)/(3*N)`, yielding 69/9, 101/15, and 181/30. Matching dimensions do not make policies from different protocols interchangeable.

**Retraining is a separate operation.** Training configurations and shared learning routines are included, but the evaluation and summarization commands above do not invoke them. Retraining produces new policies and normalizers. The generic training entry point does not automatically load the per-study JSON configurations. The commands documented here use the supplied policies.

## Computational considerations

PPO evaluation loads the selected policy and normalizer. The scalability evaluator loads policies on the CPU. The main evaluator uses the default device selection of Stable-Baselines3. MPC performs three SLSQP initializations per decision with an eight-hour horizon, `maxiter=40`, and `ftol=1e-7`. Complete MPC evaluations, particularly for N=10, can require substantial runtime. The evaluation CLIs do not implement a wall-clock timeout. Use an external process supervisor when a runtime limit is required.

The scalability evaluator sets PyTorch and numerical-library thread counts to one. Saved timing measurements distinguish policy prediction, normalization plus prediction, full controller decisions, individual solver calls, and the sum of three solver calls. Compare matching scopes and units. The main-study `online_eval_time_s` is elapsed scenario evaluation time, not per-decision latency. Timings depend on hardware and execution environment.

## Reproducibility scope and limitations

- New main-study MPC evaluations can show small numerical differences from the stored reference metrics. The package does not establish the precise cause of these differences.
- Successful process completion does not certify convergence of every MPC optimization. The main evaluator selects the candidate with the lowest finite objective without requiring `sol.success`, and does not export individual SLSQP termination statuses.
- Strict artifact-hash checking prevents detailed reconstruction of the supplied scalability solver diagnostics. Timing tables remain available as saved reference measurements.
- The supplied main-study references do not include the exact trajectory materials needed to regenerate the manuscript's main allocation-coefficient time-series figure. No regeneration command for that figure is provided.
- Stored main-study paired statistics are recovered rather than recalculated during reference summarization. New evaluations, saved-statistic reconstruction, and policy retraining are distinct reproducibility tasks.
