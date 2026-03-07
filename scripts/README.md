# Скрипты Проекта

Эта папка содержит CLI-скрипты для запуска экспериментов, валидации и inference.

## Структура

- `inference/` — скрипты для submission/inference:
  - `build_submission_pack_v1.py`
  - `predict_submission_v1.py`
  - `predict_submission_cli_v1.sh`
- `hypotheses/h1/` — изолированные скрипты для проверки H1-гипотез.
- `hypotheses/h2/` — изолированные скрипты для проверки H2-гипотез.
- корневые `*.py` в `scripts/` — общие оркестраторы:
  - одиночный запуск (`run_experiment_v1.py`);
  - sweep (`sweep_experiments_v1*.py`);
  - сравнение/summary (`compare_runs_v1.py`, `summarize_battery_region_aware_v1.py`);
  - dev/lock и служебные проверки.

## Что запускать в типовых сценариях

- Обычный эксперимент: `run_experiment_v1.py`
- Серия экспериментов: `sweep_experiments_v1_parallel.py`
- Финальный inference: `inference/predict_submission_cli_v1.sh`

Подробные команды: `docs/REPRODUCIBILITY.md`, `docs/VALIDATION.md`, `docs/SUBMISSION.md`.
