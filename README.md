# MEPhI Hack 2026: Raman Spectra Classification

Проект для хакатона по классификации рамановских спектров тканей мозга мышей.

Текущий практический фокус:

- **3-классовая классификация**: `control`, `endo`, `exo`;
- быстрые и воспроизводимые эксперименты;
- модульная архитектура под большое число гипотез;
- строгий контроль leakage через group-wise валидацию.

## Ключевые Документы Проекта

Вместо единого отчёта - основные файлы, в которых описано всё про наше решение:

1. `docs/SOLUTION.md` — что за решение выбрано и как оно работает.
2. `docs/SUBMISSION.md` — как запускать решение на своих данных (важно для организаторов).
3. `docs/VALIDATION.md` — как устроена наша валидация.
4. `docs/EXPERIMENTS.md` — какие гипотезы тестировались и что сработало.
5. `docs/REPRODUCIBILITY.md` — как воспроизвести результаты через конфиги и команды.
6. `docs/SOURCES.md` — источники, статьи и что именно мы взяли из них в решение.
7. `docs/ADDITIONAL_TASK_SPECTRAL_AREAS.md` — решение дополнительной задачи про информативные спектральные области.

## Презентация

- Финальная презентация команды: `docs/Presentation-MISIS-Deja-Vu.pdf`.

## 2. Установка И Запуск

### Clone

```bash
git clone <repo_url>
cd Mephi_hack_private
```

### Dependencies And Environment

```bash
pip install uv
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python -e .
```

### Local Run

Инференс (single-center):

```bash
scripts/inference/predict_submission_cli_v1.sh \
  --region cortex \
  --center 1500 \
  --input path/to/spectrum.txt
```

Инференс (dual-center):

```bash
.venv/bin/python scripts/inference/predict_submission_v1.py \
  --region cortex \
  --input-1500 path/to/spectrum_1500.txt \
  --input-2900 path/to/spectrum_2900.txt \
  --dual-policy main
```

Организаторам хакатона: подробнее это расписано в `docs/SUBMISSION.md`, лучше ориентироваться на него.

## 3. Основной Функционал

Ключевые возможности проекта:

1. 3-классовая классификация Raman-спектров (`control/endo/exo`).
2. Поддержка двух окон спектра (`1500`, `2900`) и их fusion.
3. Гибкий config-driven запуск экспериментов (`configs/experiment/*`).
4. Поддержка нескольких модельных семейств:
   - classical (`CatBoost`, `LogReg`, `SVM`);
   - DL (`RamanNet`, `Spectral Transformer` и др.).
5. Строгая валидация (`region-aware`, `dev/lock`, `multi-seed`).
6. Готовый inference-контур для проверки на новых спектрах.

## Команда

Команда: **MISIS Deja Vu**

Состав:

1. **Романов Никита Егорович**GitHub: https://github.com/KARTASAR
2. **Голованов Евгений Олегович**
   GitHub: https://github.com/EugGolovanov

Роли и вклад:

1. **Романов Никита Егорович**
   - разработка и интеграция основного ML-пайплайна (data -> preprocess -> training -> inference);
   - реализация и поддержка экспериментальных запусков и финального inference-контура;
   - структурирование проектной документации и финальной упаковки решения.
2. **Голованов Евгений Олегович**
   - постановка и проверка ML-гипотез по архитектурам и preprocessing;
   - анализ качества моделей, интерпретация результатов и выбор финальных стратегий;
   - участие в подготовке валидационного протокола и финальной стратегии экспериментов.

## Кратко О Задаче

Целевая задача для этого проекта:

- вход: Raman map `.txt` файлы из `data/real`;
- цель: multiclass-классификация `control` / `endo` / `exo`;
- дополнительно: интерпретация важных спектральных областей.

## Реальные Данные

Локальные данные:

- `data/real/control`
- `data/real/endo`
- `data/real/exo`

Ключевые факты:

- формат основного файла: `#X #Y #Wave #Intensity`;
- один файл = карта спектров (много точек), не один спектр;
- есть два окна: `center1500` и `center2900`;
- есть аномалии (`*_Average`, один конфликтный файл по окну), они зафиксированы в документации.

Подробно:

- [SOLUTION](docs/SOLUTION.md)

## Зафиксированный Стек (v1)

Зафиксированный стек:

- `numpy`, `scipy`
- `polars`, `pyarrow`
- `scikit-learn`
- `catboost`
- `pybaselines`
- `torch` (для DL-волны экспериментов)
- `hydra-core`, `omegaconf`
- `loguru`

Подробно:

- [REPRODUCIBILITY](docs/REPRODUCIBILITY.md)

## Эксперименты И Валидация

Полные команды и протоколы вынесены в профильные документы:

1. `docs/VALIDATION.md`
2. `docs/EXPERIMENTS.md`
3. `docs/REPRODUCIBILITY.md`
4. `docs/SUBMISSION.md`
5. `docs/SOURCES.md`
6. `docs/ADDITIONAL_TASK_SPECTRAL_AREAS.md`

## Структура Репозитория

```text
Mephi_hack_private/
├── artifacts/
│   └── submission_pack_v1/      # serialized models + manifest for inference
├── configs/
│   └── experiment/              # experiment YAML configs
├── data/
│   └── real/                    # local real Raman data
├── docs/
│   ├── SOLUTION.md
│   ├── SUBMISSION.md
│   ├── VALIDATION.md
│   ├── EXPERIMENTS.md
│   ├── REPRODUCIBILITY.md
│   ├── SOURCES.md
│   └── ADDITIONAL_TASK_SPECTRAL_AREAS.md
├── runs/                        # local run artifacts/metrics (tracked locally)
├── scripts/
│   ├── inference/               # organizer-facing prediction entrypoints
│   └── hypotheses/
│       ├── h1/
│       └── h2/
├── src/
│   └── raman_hack/
│       ├── data/                # загрузка/сборка датасета и feature tables
│       ├── preprocess/          # baseline/normalization/denoise pipeline
│       ├── models/              # CatBoost/linear/DL модели
│       ├── metrics/             # расчет метрик качества
│       ├── validation/          # split-политики и валидационные процедуры
│       ├── runner/              # train/eval orchestration
│       ├── tracking/            # логирование и run-артефакты
│       ├── interpretation/      # inverse-task и спектральная интерпретация
│       └── experimental/        # изолированные экспериментальные компоненты
├── tests/
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 6. Архитектура И Структура (Опционально)

Архитектурная идея:

1. ядро решений реализовано в `src/raman_hack/*`;
2. каждый эксперимент задается YAML-конфигом в `configs/experiment/*`;
3. запуск сохраняет артефакты в `runs/<run_id>/`;
4. inference вынесен в `scripts/inference/*`.

## 7. Демонстрация Проекта

Так как это ML-проект без веб-интерфейса, демонстрация — через inference и артефакты:

1. `prediction.json` (итог предсказания по классам),
2. `probabilities.png` (распределение вероятностей),
3. `spectrum_center1500.png` / `spectrum_center2900.png` (спектр с информативными диапазонами).

Все файлы создаются автоматически после запуска inference-команд из раздела установки.

## 8. Итоговое Заключение

Почему проект важен:

1. решает практическую задачу классификации биомедицинских спектров в условиях малого числа групп и высокой вариативности;
2. делает акцент на устойчивость, а не на единичный “удачный” score;
3. объединяет сильные стороны двух спектральных окон через fusion.

Чем отличается:

1. config-driven воспроизводимость (`код + конфиг + команда`);
2. строгий протокол валидации (`region-aware + dev/lock + multi-seed`);
3. явное разделение `main`/`fallback` режимов для надежного применения.

Что можно улучшать дальше:

1. донастройка region-aware routing в fully train-aware режиме;
2. расширение проверок на дополнительных manifests/seed-диапазонах;
3. более глубокая интерпретация спектральных диапазонов для explainability.

## Примечания

- `data/real` listed in `.gitignore` (локальные большие данные).

## Лицензия

MIT (см. файл [LICENSE](LICENSE)).
