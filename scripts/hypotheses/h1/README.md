# H1 Scripts

Скрипты в этой папке относятся только к гипотезе H1 (region-aware router), чтобы не захламлять корневой `scripts/`.

- `run_h1_region_router_battery_proxy_v2.py`  
  Proxy-проверка на уже сохранённых предсказаниях (`no-retrain`).

- `run_h1_region_router_battery_train_aware_v2.py`  
  Train-aware проверка: retrain per holdout-triplet (1500 + 2900), сбор `battery-long`.

- `check_h1_holdout_leakage_v2.py`  
  Leakage-аудит train-aware run-manifest.

- `run_h1_train_aware_protocol_v2.py`  
  Orchestrator: train-aware battery -> leakage audit -> region-aware summaries -> dev/lock sweeps.

Пример:

```bash
.venv/bin/python scripts/hypotheses/h1/run_h1_train_aware_protocol_v2.py \
  --sample-meta runs/<run_id>/sample_meta.parquet \
  --output-root runs/analysis/hyp_v2_h1_train_aware_YYYYMMDD \
  --balanced-n-total 24 --seed-from 1 --seed-to 20
```
