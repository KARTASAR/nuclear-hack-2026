# Experiments: Isolated Research Layer

`experiments/` is the standalone laboratory layer for research hypotheses and extended validation.
It is isolated from production paths (`src/`, `configs/`, `scripts/`, `docs/`) and uses its own runtime/output space.

## Key Documents

1. `experiments/docs/EXPERIMENTS_APPENDIX.md` — factual continuation of the main experiment story (`11.*` sections).
2. `experiments/docs/REPRODUCIBILITY.md` — reproducible runs and command patterns for this layer.
3. `experiments/docs/SOURCES.md` — source map for the experimental branch.
4. `experiments/docs/general/*` — historical detailed logs and design tickets.
5. `experiments/docs/artifacts/*` — lightweight CSV/JSON summaries referenced by docs.

## Module Map

1. `RGT/RSPSSL`: unmixing-oriented pipeline, end-to-end helpers, unmixing visualization.
2. `multitaper/MoE`: experimental spectral frontend and sector-aware mixture-of-experts.
3. `XAI`: batch explainability, validation reports, and spectral-band summaries.

## Layout

```text
experiments/
├── configs/experiment/         # experimental YAML configs
├── docs/
│   ├── EXPERIMENTS_APPENDIX.md
│   ├── REPRODUCIBILITY.md
│   ├── SOURCES.md
│   ├── general/                # historical docs + implementation snapshots
│   └── artifacts/              # lightweight tables/reports
├── runs/                       # local experimental artifacts (git-kept via .gitkeep)
├── scripts/                    # isolated CLIs
├── src/raman_hack/             # isolated package snapshot
└── tests/                      # isolated experimental tests
```

## Runtime Isolation

All experimental scripts bootstrap `experiments/src` first in `sys.path`.
Default output root is `experiments/runs`.

Override example:

```bash
EXPERIMENTS_OUTPUT_ROOT=experiments/runs_alt \
  .venv/bin/python experiments/scripts/run_experiment_v1.py \
  --config experiments/configs/experiment/v1_smoke_center1500_fast.yaml
```

## Main CLI Commands

### 1) Single run

```bash
# canonical (script-path)
.venv/bin/python experiments/scripts/run_experiment_v1.py \
  --config experiments/configs/experiment/v1_smoke_center1500_fast.yaml

# equivalent (module)
.venv/bin/python -m experiments.scripts.run_experiment_v1 \
  --config experiments/configs/experiment/v1_smoke_center1500_fast.yaml
```

### 2) Parallel sweep

```bash
# canonical (script-path)
.venv/bin/python experiments/scripts/sweep_experiments_v1_parallel.py \
  --configs experiments/configs/experiment/rgt \
  --workers 2 --continue-on-error

# equivalent (module)
.venv/bin/python -m experiments.scripts.sweep_experiments_v1_parallel \
  --configs experiments/configs/experiment/rgt \
  --workers 2 --continue-on-error
```

### 3) Full E2E (RGT pair + fusion)

```bash
# canonical (script-path)
.venv/bin/python experiments/scripts/run_full_e2e_v1.py \
  --config-1500 experiments/configs/experiment/rgt/rgt_1500_file_mean.yaml \
  --config-2900 experiments/configs/experiment/rgt/rgt_2900_file_mean.yaml

# equivalent (module)
.venv/bin/python -m experiments.scripts.run_full_e2e_v1 \
  --config-1500 experiments/configs/experiment/rgt/rgt_1500_file_mean.yaml \
  --config-2900 experiments/configs/experiment/rgt/rgt_2900_file_mean.yaml
```

### 4) Unmixing visualization

Select concrete run directories first:

```bash
RUN_1500="$(find experiments/runs -mindepth 1 -maxdepth 1 -type d -name '*1500*' | sort | tail -n 1)"
RUN_2900="$(find experiments/runs -mindepth 1 -maxdepth 1 -type d -name '*2900*' | sort | tail -n 1)"
```

Then run:

```bash
# canonical (script-path)
.venv/bin/python experiments/scripts/visualize_unmixing.py \
  --run-dir "$RUN_1500" \
  --run-dir "$RUN_2900"

# equivalent (module)
.venv/bin/python -m experiments.scripts.visualize_unmixing \
  --run-dir "$RUN_1500" \
  --run-dir "$RUN_2900"
```

### 5) Batch explainability + validation

Select a concrete run directory:

```bash
RUN_DIR="$(find experiments/runs -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
```

```bash
# canonical (script-path)
.venv/bin/python experiments/scripts/explain_batch.py \
  --run-dir "$RUN_DIR"
.venv/bin/python experiments/scripts/validate_explanations.py \
  --run-dir "$RUN_DIR"

# equivalent (module)
.venv/bin/python -m experiments.scripts.explain_batch \
  --run-dir "$RUN_DIR"
.venv/bin/python -m experiments.scripts.validate_explanations \
  --run-dir "$RUN_DIR"
```

## Legacy Ported CLIs

Ported to `raman_hack` runtime and kept as active interfaces:

1. `experiments/scripts/train_all_models.py`
2. `experiments/scripts/preprocess_data.py`
3. `experiments/scripts/run_pipeline_with_fake_data.py`
4. `experiments/scripts/explain_model.py`
