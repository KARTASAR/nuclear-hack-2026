# Reproducibility (Experiments Layer)

Этот документ фиксирует воспроизводимый запуск экспериментального слоя `experiments/`.

## 1) Environment

```bash
pip install uv
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python -e .
```

## 2) Single Experiment Run

```bash
# canonical (script-path)
.venv/bin/python experiments/scripts/run_experiment_v1.py \
  --config experiments/configs/experiment/v1_smoke_center1500_fast.yaml

# equivalent (module)
.venv/bin/python -m experiments.scripts.run_experiment_v1 \
  --config experiments/configs/experiment/v1_smoke_center1500_fast.yaml
```

## 3) Parallel Sweep

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

## 4) RGT Full E2E + Unmixing Visuals

Сначала выбрать конкретные run-директории:

```bash
RUN_1500="$(find experiments/runs -mindepth 1 -maxdepth 1 -type d -name '*1500*' | sort | tail -n 1)"
RUN_2900="$(find experiments/runs -mindepth 1 -maxdepth 1 -type d -name '*2900*' | sort | tail -n 1)"
```

```bash
# canonical (script-path)
.venv/bin/python experiments/scripts/run_full_e2e_v1.py \
  --config-1500 experiments/configs/experiment/rgt/rgt_1500_file_mean.yaml \
  --config-2900 experiments/configs/experiment/rgt/rgt_2900_file_mean.yaml

.venv/bin/python experiments/scripts/visualize_unmixing.py \
  --run-dir "$RUN_1500" \
  --run-dir "$RUN_2900"

# equivalent (module)
.venv/bin/python -m experiments.scripts.run_full_e2e_v1 \
  --config-1500 experiments/configs/experiment/rgt/rgt_1500_file_mean.yaml \
  --config-2900 experiments/configs/experiment/rgt/rgt_2900_file_mean.yaml

.venv/bin/python -m experiments.scripts.visualize_unmixing \
  --run-dir "$RUN_1500" \
  --run-dir "$RUN_2900"
```

## 5) XAI Batch + Validation

Сначала выбрать конкретный run:

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

## 6) Output Root Policy

По умолчанию все новые запуски пишут в `experiments/runs`.
Переопределение:

```bash
EXPERIMENTS_OUTPUT_ROOT=experiments/runs_alt \
  .venv/bin/python experiments/scripts/run_experiment_v1.py \
  --config experiments/configs/experiment/v1_smoke_center1500_fast.yaml
```

## 7) What To Compare

Для проверок используйте:

1. `$RUN_DIR/config_resolved.yaml`
2. `$RUN_DIR/metrics.json`
3. `$RUN_DIR/fold_metrics.csv`
4. `$RUN_DIR/predictions.parquet`
5. исторические summary из `experiments/docs/artifacts/*`
