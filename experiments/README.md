# Эксперименты: Изолированный исследовательский слой

`experiments/` — это автономный лабораторный слой для проверки исследовательских гипотез и расширенной валидации.
Он изолирован от производственных путей (`src/`, `configs/`, `scripts/`, `docs/`) и использует собственное пространство выполнения/вывода.

## Ключевые документы

1. `experiments/docs/EXPERIMENTS_APPENDIX.md` — фактическое продолжение основной истории эксперимента (разделы `11.*`).

2. `experiments/docs/REPRODUCIBILITY.md` — воспроизводимые запуски и шаблоны команд для этого слоя.

3. `experiments/docs/SOURCES.md` — карта исходного кода для экспериментальной ветки.

4. `experiments/docs/general/*` — подробные исторические журналы и проектные задания.

5. `experiments/docs/artifacts/*` — легковесные CSV/JSON-резюме, на которые ссылается документация.

## Карта модулей

1. `RGT/RSPSSL`: ориентированный на разделение смесей, сквозные вспомогательные функции, визуализация разделения смесей.

2. `multitaper/MoE`: экспериментальный спектральный интерфейс и сегментированный инструмент для анализа смесей.

3. `XAI`: пакетная объяснимость, отчеты о валидации и сводки по спектральным диапазонам.

## Структура

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

Все экспериментальные скрипты сначала загружают `experiments/src` в `sys.path`.
Корневая папка вывода по умолчанию — `experiments/runs`.

Пример переопределения:

```bash
EXPERIMENTS_OUTPUT_ROOT=experiments/runs_alt \
  .venv/bin/python experiments/scripts/run_experiment_v1.py \
  --config experiments/configs/experiment/v1_smoke_center1500_fast.yaml
```

## Основные команды CLI

### 1) Один запуск

```bash
# canonical (script-path)
.venv/bin/python experiments/scripts/run_experiment_v1.py \
  --config experiments/configs/experiment/v1_smoke_center1500_fast.yaml

# equivalent (module)
.venv/bin/python -m experiments.scripts.run_experiment_v1 \
  --config experiments/configs/experiment/v1_smoke_center1500_fast.yaml
```

### 2) Параллельный запуск

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

### 3) Полный запуск pipeline вместе с обучением GAN для очистки спектра и дальнейшей классификации

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

### 4) Визуализация разделения спектров

Сначала выберите конкретные каталоги запусков:

```bash
RUN_1500="$(find experiments/runs -mindepth 1 -maxdepth 1 -type d -name '*1500*' | sort | tail -n 1)"
RUN_2900="$(find experiments/runs -mindepth 1 -maxdepth 1 -type d -name '*2900*' | sort | tail -n 1)"
```

Затем:

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

### 5) Интерпретация и валидация

Выберите конкретный каталог запуска:

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

