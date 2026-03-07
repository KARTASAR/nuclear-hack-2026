# H2 Scripts

Скрипты в этой папке относятся только к гипотезе H2 (hierarchical classification: `region -> class`).

- `run_h2_hierarchical_battery_train_aware_v2.py`  
  Train-aware проверка: retrain per holdout-triplet (1500 + 2900), далее H2 known-region class-head на `cv` и оценка на `holdout`.

- `check_h2_holdout_leakage_v2.py`  
  Leakage-аудит train-aware run-manifest.

- `run_h2_train_aware_protocol_v2.py`  
  Orchestrator: train-aware battery -> leakage audit -> region-aware summaries -> dev/lock sweeps.

Пример:

```bash
.venv/bin/python scripts/hypotheses/h2/run_h2_train_aware_protocol_v2.py \
  --sample-meta runs/<run_id>/sample_meta.parquet \
  --configs-1500 configs/experiment/p2_locked_holdout/p2lk_1500_ramannet_ss11.yaml,configs/experiment/p2_locked_holdout/p2lk_1500_ramannet_ss22.yaml,configs/experiment/p2_locked_holdout/p2lk_1500_ramannet_ss33.yaml,configs/experiment/p2_locked_holdout/p2lk_1500_ramannet_ss42.yaml \
  --configs-2900 configs/experiment/p2_locked_holdout/p2lk_2900_trpatch8_ss11.yaml,configs/experiment/p2_locked_holdout/p2lk_2900_trpatch8_ss22.yaml,configs/experiment/p2_locked_holdout/p2lk_2900_trpatch8_ss33.yaml,configs/experiment/p2_locked_holdout/p2lk_2900_trpatch8_ss42.yaml \
  --output-root runs/analysis/hyp_v2_h2_train_aware_YYYYMMDD \
  --balanced-n-total 24 --max-triplets 8 --seed-from 1 --seed-to 20
```

