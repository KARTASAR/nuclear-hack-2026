# Submission Guide

Этот файл описывает минимальный путь запуска финального решения для организатора.

## 0) Что должно быть установлено

- Python 3.11+

## 1) Установка зависимостей

Установить `uv`:

```bash
pip install uv
```

Создать окружение и установить зависимости:

```bash
uv venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## 2) Основной запуск (single-center)

Пример команды для запуска в терминале:

```bash
scripts/inference/predict_submission_cli_v1.sh \
  --region cortex \
  --center 1500 \
  --input path/to/spectrum.txt
```

### Параметры

- `--region`: область мозга (`cortex`, `striatum`, `cerebellum`).
- `--center`: `1500` или `2900`.
- `--input`: путь к входному `.txt` файлу спектра.

Важно: этот скрипт автоматически использует Python из `./.venv/bin/python` в корне репозитория. Поэтому перед запуском должен быть выполнен шаг 1 (создано окружение `.venv` и установлены зависимости).

## 3) Запуск с двумя файлами одного образца (рекомендуется)

Если для одного и того же образца вы можете передать сразу два файла:

- спектр с центром `1500`,
- спектр с центром `2900`,

используйте этот режим. По нашим замерам он обычно даёт более точный результат, чем запуск только с одним файлом.

```bash
.venv/bin/python scripts/inference/predict_submission_v1.py \
  --region cortex \
  --input-1500 path/to/spectrum_1500.txt \
  --input-2900 path/to/spectrum_2900.txt \
  --dual-policy main
```

## 4) Формат входа

Поддерживаются `.txt` файлы в форматах:

1. 2 колонки: `wave intensity` (одиночный спектр).
2. 4 колонки map-формата: `x y wave intensity` (будет усреднено до одного спектра).

## 5) Формат выхода

### Для `predict_submission_cli_v1.sh`

Короткий summary в терминале:

- mode,
- policy,
- predicted_class,
- probabilities по `control / endo / exo`.

Также результат сохраняется в отдельную папку запуска (по умолчанию в `artifacts/predictions/`), и путь к `prediction.json` выводится в терминал.
После сохранения JSON автоматически строятся графики:

- `probabilities.png` — вероятности классов.
- `spectrum_center1500.png` или `spectrum_center2900.png` — спектр с выделенными информативными диапазонами.

### Для `predict_submission_v1.py`

Полный JSON с полями:

- `predicted_class`,
- `probabilities` (`control`, `endo`, `exo`),
- `policy_used`, `strategy_name`,
- `model_ids_used`,
- `mode` (`single_center` или `dual_center`).

JSON автоматически сохраняется в отдельную папку запуска внутри `artifacts/predictions/`, и путь к сохранённому файлу выводится в терминал.
После сохранения JSON автоматически строятся:

- `probabilities.png`;
- `spectrum_center1500.png`;
- `spectrum_center2900.png`.
