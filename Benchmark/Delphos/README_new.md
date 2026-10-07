# Refactored Delphos (new files)

`agent.py`, `analysis.py` and `delphos.py` remain untouched. The new implementation does not import them. `delphos_main.py` is the requested new entry point; every other added source has `new` in its name.

## Files

| File | Responsibility |
| --- | --- |
| `delphos_main.py` | Dataset, seeds, estimation budget `B`, experiment configuration and execution |
| `delphos_new.py` | DQN actions, episodes, rewards, candidate tracking and training |
| `network_new.py` | Q-network and bounded experience replay |
| `specification_new.py` | State encoding, valid actions and Apollo parameter selection |
| `apollo_new.py` | Original R/Apollo MNL likelihood, estimation and full model outputs |
| `results_new.py` | Per-run SQLite results, cached lookups and fitting-attempt ledger |
| `reporting_new.py` | Optional learning curve and best metric versus fitting attempts |
| `test_delphos_new.py` | Automated regression checks without invoking R |

## Running

Use a Python environment containing numpy, pandas, torch and rpy2, with R and Apollo installed. Matplotlib is needed only for `--plots`. The existing local `Delphos` pyenv environment was used for verification.

From this directory:

```sh
python delphos_main.py --budget 500 --seeds 101 102 103 104 105 106 107 108 109 110 --plots
```

`B = 500` in `delphos_main.py` sets the default budget and is actually passed to the learner. Dataset paths are resolved relative to the source directory, not the shell working directory. The default dataset is `Benchmark/dataset/swissmetro_process.csv`. Use `--dataset` and `--output` to override the locations.

Each execution gets a timestamped experiment directory under `experiments_new`; each seed has its own directory. An existing seed directory is refused to prevent accidental overwriting or cache contamination. There is no automatic checkpoint resume or import of legacy rewards.

## Budget and storage

The budget matches the existing VNS/SA policy: every call entering `apollo_estimate` counts, including unsuccessful fits and errors. Cached revisits do not count. Errors preparing a specification before the fitting call are recorded and cached but do not count. The budget is checked before fitting, and the result of fit B is included in learning and candidate selection before the run stops. There are no post-search refits.

`models_new.sqlite` has one primary-keyed row per specification, with its attempt number, status and complete result dictionary (metrics, coefficients, standard errors and errors). A started record is committed immediately before fitting, so an interrupted fit remains auditable. Python dictionary lookups serve repeated models without reading CSV files. SQLite is the authoritative durable result store; CSV is an export. Each run uses a single writer and independent cache.

`models_new.csv` exports all results; `best_candidates_new.csv` exports the complete stored result for each metric's best candidate. `best_history_new.csv` stores only improvements, with episode and estimation number, instead of repeating the incumbent after every episode. `training_log_new.csv` records every completed episode, cache hits, fitting count and status. `action_log_new.csv` records decisions; a duplicate buffer CSV is unnecessary. `summary_new.json` reports actual fits, unique specifications, successful fits, cache hits and the stop reason. `config_new.json` and `dqn_model_new.pth` retain configuration and network weights.

The reward convergence/patience stopping rule is removed. `--max-episodes` (default 10,000) is a safety cap for repeated candidates or preparation failures: a run that hits it reports `episode_limit_before_budget`, never budget completion. Episodes also have a 100-action horizon to prevent endless change cycles. These safeguards can affect trajectories; an incomplete run should not be compared as a completed B-fit run. Programmatic `train(stop_requested=callback)` supports external cancellation, checked between episodes and before evaluation. It does not interrupt an active R fit. Keyboard interruption saves available results and propagates the interrupt.

## Changes found during review

- The old entry point defines `B` but never passes it to the learner.
- The old `get_mnl_outcomes` usually returns `None` for a newly estimated model; the return is inside the 3,000-episode merge condition.
- The old global/job CSV cache requires repeated reads and rewrites, assumes files exist at episode zero, and shares evaluations across seeds.
- State-vector offsets overlap ASC and attribute entries; generic-only/covariate configurations also have inconsistent tuple formats. The new implementation uses four-component state tuples consistently and separate encoding blocks.
- Change actions could target missing attributes, making no change. ASC taste variants could represent the same model. Empty termination is now invalid and those redundant actions are removed.
- The old replay labels every transition terminal. The new replay marks the episode's last transition terminal and masks invalid actions when calculating target Q-values.
- The original generic interaction indices assume exactly seven covariates. Parameter selection now works for arbitrary configured covariate counts and preserves the intended coefficient sets. Non-contiguous category values are used directly rather than expanded into a numerical range. An unused `L_0` parameter and duplicate fixed names are removed.
- Apollo setup contained an entire coefficient-building block overwritten immediately afterwards, plus two input validations. These redundant operations are removed.
- Failed estimates are excluded from rewards and best candidates using value-based success checks, including numpy boolean values. Successful-model counts no longer depend on whether normalised reward happens to be positive.
- Reward extrema are maintained incrementally across successful results and include the current observation. The Apollo-returned `LL0` replaces the approximate Python row-count calculation. AIC/BIC/LL reward scaling remains adaptive; previously stored replay rewards are not retroactively rescaled. Best candidates are ranked by raw metrics.

The new code omits unused exhaustive model generators, duplicate serial/parallel outcome wrappers, legacy CSV-merging and cleanup helpers, broken candidate re-estimation and Pareto utilities, and unused imports. Those remain available in the originals. Optional reporting is intentionally smaller; the former misnamed Q-distribution plot was an action-frequency plot, not a network Q-value plot.

## Preserved modelling behaviour and limits

Apollo in R remains the estimator through rpy2. The original MNL utilities, log and Box–Cox expressions, generic/specific coefficients, ASC reference alternative, panel product, initial values and estimation settings are retained. Data still use the original R seed 123, 80/20 individual split and estimation on the training subset. The new code only exposes the five covariates selected in the original agent configuration, rather than also constructing unused luggage/who coefficients.

The original code does **not** calculate a held-out likelihood: its out-of-sample estimation block was commented out. The stored Apollo `LLout` field is preserved and must not be interpreted as a separately evaluated test-set likelihood. This refactor does not harmonise Apollo's split or model validity rules with the VNS/SA data pipeline.

Some Box–Cox specifications fail inside Apollo's gradient calculation. The original expression is deliberately retained; failures are now recorded and consume one fitting attempt. Consequently, the numerical results and DQN trajectories can differ from the old code because the encoding, replay, action validity and reward bookkeeping bugs are fixed. Old network weights are not compatible with the corrected encoding.

## Verification

```sh
python -m unittest discover -s . -p 'test_*new.py' -v
```

Verification performed during refactoring:

- Seven regression tests covering all taste/covariate encoding modes, valid actions, parameter selection, cache hits, failed fits, exact budgets, zero budget, external stopping, episode caps, terminal transitions and learning updates.
- 1,000 seeded parameter-selection comparisons against the original seven-covariate implementation: identical estimated coefficient sets.
- A real Apollo linear model `100_100_100_000_000` estimated successfully on the default dataset: AIC 8649.04186307614; a repeated request reused the stored result with the count still at one.
- A one-fit run through the new entry point recorded an Apollo gradient failure and stopped at exactly one attempt.
- Syntax checks and SHA-256 verification of the three unchanged original files.

A full ten-seed, 500-fit benchmark has not been run as part of this refactor.
