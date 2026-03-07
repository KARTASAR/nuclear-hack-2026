# Reproducibility (Config-Driven)

Наше решение изначально спроектировано как воспроизводимое:

1. архитектуры и подходы реализованы в коде (`src/`), а не “вручную в ноутбуках”;
2. конкретный эксперимент задаётся конфигом (`configs/experiment/*.yaml`);
3. каждый запуск сохраняет фактически использованный конфиг и артефакты в `runs/<run_id>/`.

Поэтому при использовании тех же данных и тех же конфигов можно получить практически те же результаты (с возможными небольшими отклонениями из-за среды исполнения).

Ключевой принцип этого проекта:

1. подход реализован в коде,
2. конкретный эксперимент задаётся конфигом,
3. запуск воспроизводится командой.

То есть основная единица воспроизводимости здесь — не “описание словами”, а связка:
`конфиг -> команда -> run-артефакты`.

## 1) Что уже делает воспроизведение строгим

1. Все важные гипотезы заведены как конфиги в `configs/experiment/*`.
2. Каждый run сохраняет `config_resolved.yaml` (финальный конфиг фактического запуска).
3. Метрики и предсказания сохраняются в папке run в `runs/<run_id>/`.
4. Есть реестр запусков `runs/registry.csv`.

Практически это означает, что любой зафиксированный результат можно поднять через тот же конфиг и тот же тип команды.

## 2) Базовая установка

```bash
pip install uv
uv venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## 3) Воспроизведение экспериментов из конфигов

Типовой запуск одного эксперимента:

```bash
.venv/bin/python scripts/run_experiment_v1.py \
  --config <path_to_yaml_config>
```

Пример (ключевой 1500-конфиг):

```bash
.venv/bin/python scripts/run_experiment_v1.py \
  --config configs/experiment/p2_locked_holdout/p2lk_1500_ramannet_ss11.yaml
```

Пример (ключевой 2900-конфиг):

```bash
.venv/bin/python scripts/run_experiment_v1.py \
  --config configs/experiment/p2_locked_holdout/p2lk_2900_trpatch8_ss11.yaml
```

После запуска проверяются:

1. `runs/<run_id>/config_resolved.yaml`
2. `runs/<run_id>/metrics.json`
3. `runs/<run_id>/fold_metrics.csv`
4. `runs/<run_id>/predictions.parquet`

## 4) Воспроизведение “волн” (batch по наборам конфигов)

Для прогона набора конфигов:

```bash
.venv/bin/python scripts/sweep_experiments_v1_parallel.py \
  --configs configs/experiment/<folder_with_yaml_configs> \
  --workers 2
```

Примеры каталогов:

1. `configs/experiment/protocol5_core/`
2. `configs/experiment/p2_locked_holdout/`
3. `configs/experiment/dl_arch_expansion/`
4. `configs/experiment/ramanspy_stepa_screen/`

## 5) Воспроизведение fusion-результатов

Fusion воспроизводится отдельным скриптом по run_id:

```bash
.venv/bin/python scripts/dual_window_fusion_v1.py \
  --runs-root runs \
  --run-1500 <run_id_1500> \
  --run-2900 <run_id_2900> \
  --alpha-step 0.01 \
  --output <output_csv>
```

Пример:

```bash
.venv/bin/python scripts/dual_window_fusion_v1.py \
  --runs-root runs \
  --run-1500 20260306_193949_dlqs_1500_ramannet_literature_5f_ss42 \
  --run-2900 20260306_193409_dlqs_2900_spectral_transformer_literature \
  --alpha-step 0.01 \
  --output runs/fusion_sweep_dl_seed42_alpha001.csv
```

## 6) Воспроизведение строгой валидации

Region-aware summary:

```bash
.venv/bin/python scripts/summarize_battery_region_aware_v1.py \
  --battery-input <battery_long.csv> \
  --sample-meta <sample_meta.parquet> \
  --output-slices <out_slices.csv> \
  --output-summary <out_summary.csv>
```

Dev/lock:

```bash
.venv/bin/python scripts/run_dev_lock_battery_v1.py \
  --battery-input <battery_long.csv> \
  --sample-meta <sample_meta.parquet> \
  --candidate-strategies <comma_separated_strategies> \
  --output-prefix <out_prefix>
```

Multi-seed:

```bash
.venv/bin/python scripts/sweep_dev_lock_battery_v1.py \
  --battery-input <battery_long.csv> \
  --sample-meta <sample_meta.parquet> \
  --candidate-strategies <comma_separated_strategies> \
  --seed-from 1 --seed-to 20 \
  --output-prefix <out_prefix>
```

## 7) Inference воспроизводимость

Single-center:

```bash
scripts/inference/predict_submission_cli_v1.sh \
  --region cortex \
  --center 1500 \
  --input path/to/spectrum.txt
```

Dual-center:

```bash
.venv/bin/python scripts/inference/predict_submission_v1.py \
  --region cortex \
  --input-1500 path/to/spectrum_1500.txt \
  --input-2900 path/to/spectrum_2900.txt \
  --dual-policy main
```

Результат сохраняется в `artifacts/predictions/<run_folder>/`.

## 8) Что смотреть, чтобы проверить совпадение результатов

1. Сначала конфиг: `runs/<run_id>/config_resolved.yaml`
2. Потом агрегаты: `metrics.json`, `fold_metrics.csv`
3. Потом деталь: `predictions.parquet`
4. Для межэкспериментного сравнения: `runs/registry.csv`

Если сравниваются “наши зафиксированные выводы”, удобные опорные файлы:

1. `runs/stepb_select_stability_summary.csv`
2. `runs/dl_wave_summary_20260306.csv`
3. `runs/fusion_sweep_protocol5_core.csv`
4. `runs/fusion_sweep_dl_1500ramannet_2900spectraltransformer.csv`
5. `runs/analysis/hyp_v2_h1_train_aware_p2_confirm12_20260307/h1_train_aware_summary.csv`
6. `runs/analysis/hyp_v2_h2_train_aware_p2_fast4_20260307/h2_train_aware_summary.csv`

## 9) Почему могут быть небольшие расхождения

Даже при одинаковом конфиге возможны малые отличия между машинами из-за:

1. версий библиотек,
2. backend (`cpu/mps/cuda`),
3. недетерминизма низкоуровневых операций.

Правильная проверка:

1. сверять тренд и ранжирование стратегий;
2. не требовать полного совпадения до последнего знака.
