# Конфигурации Экспериментов

В этой папке лежат YAML-конфиги, через которые запускаются все эксперименты.
Идея простая: **код общий**, а различия между гипотезами задаются именно конфигами.

## Структура

- `experiment/` — основная папка с наборами конфигов по этапам и гипотезам.
- `experiment/v1_*` — базовые стартовые конфиги (smoke/baseline).
- `experiment/sanity_center1500`, `experiment/sanity_center2900` — ранние sanity-прогоны по каждому окну.
- `experiment/stepb_quickscreen`, `experiment/stepb_select` — быстрый скрининг и отбор preprocessing-гипотез.
- `experiment/wave2_file_mean` — вторая волна классических моделей на file-level признаках.
- `experiment/protocol5_core` — core-конфиги для строгого протокола сравнения кандидатов.
- `experiment/dl_literature` — DL-конфиги, основанные на архитектурах из литературы.
- `experiment/dl_arch_expansion` — расширенные DL-варианты/абляции.
- `experiment/p2_locked_holdout` — конфиги для lock-like сравнений на фиксированных split/seed.
- `experiment/p3_hpo_ramannet1500`, `experiment/p4_hpo_transformer2900` — локальный HPO для двух ключевых DL-моделей.
- `experiment/p34_stability` — проверки устойчивости лучших DL-кандидатов.
- `experiment/arch_quick_screen_4` — быстрый скрининг 4 дополнительных архитектурных вариаций.
- `experiment/ramanspy_stepa_screen` — гипотезы preprocessing, вдохновлённые RamanSPy.

## Как читать названия файлов

Обычно имя конфига содержит:

- этап (`s1`, `s2b`, `w2`, `p2lk`, `p5`, `dlqs`, `dlx` и т.д.);
- окно спектра (`1500` или `2900`);
- семейство модели (`catboost`, `logreg`, `svm`, `ramannet`, `trpatch8`...);
- иногда `ssXX` — конкретный `split_seed`.

Это помогает быстро понять, что именно меняется в гипотезе, не открывая файл.

## Быстрый запуск

Один конфиг:

```bash
.venv/bin/python scripts/run_experiment_v1.py \
  --config configs/experiment/v1_smoke_center1500_fast.yaml
```

Серия конфигов:

```bash
.venv/bin/python scripts/sweep_experiments_v1_parallel.py \
  --configs configs/experiment/stepb_select \
  --workers 2 \
  --continue-on-error
```

## Важно

- Добавляя новый эксперимент, лучше создавать **новый YAML**, а не переписывать старый.
- Для воспроизводимости фиксируйте `split_seed` и ключевые параметры модели/препроцессинга в конфиге.
- Подробные правила валидации и сравнения кандидатов: `docs/VALIDATION.md`, `docs/EXPERIMENTS.md`, `docs/REPRODUCIBILITY.md`.
