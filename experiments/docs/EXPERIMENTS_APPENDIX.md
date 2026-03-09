# Experiments Appendix (Continuation, Sections 11.*)

Этот файл является фактическим продолжением основного экспериментального трека из `docs/EXPERIMENTS.md`.
Ниже зафиксированы итоговые решения experimental-ветки и карта исторических материалов.

## 11.1 Итоговые решения experimental-ветки

| Направление | Статус | Краткий вывод |
|---|---|---|
| `RGT/RSPSSL pipeline` | `EXPERIMENTAL` | Рабочий исследовательский контур, полезен для анализа, не заменяет основной production-контур. |
| `multitaper + MoE` | `EXPERIMENTAL` | Валиден как отдельное семейство, держится в лабораторном слое для дальнейшей проверки. |
| `XAI batch + validation` | `GO (research infra)` | Принят как обязательная исследовательская обвязка для объяснимости и сравнения гипотез. |
| `DL expansion A1-A5` | `NO-GO` | Архитектурное расширение не улучшило текущие winners в закреплённом протоколе. |
| `Late fusion` | `GO` | Подтверждён устойчивый прирост по сравнению с single-window режимами. |
| `Legacy raw migration artifacts` | `REMOVED` | Миграционные трассировочные артефакты удалены, оставлен только рабочий интерфейс. |

## 11.2 Карта experimental-модулей

1. `RGT/RSPSSL`
   - конфиги: `experiments/configs/experiment/rgt/*`
   - запуск: `experiments/scripts/run_full_e2e_v1.py`
   - визуализация: `experiments/scripts/visualize_unmixing.py`
2. `multitaper/MoE`
   - код: `experiments/src/raman_hack/models/multitaper.py`, `.../moe.py`
   - smoke/регрессия: `experiments/tests/test_raman_multitaper.py`, `.../test_raman_moe_model.py`
3. `XAI`
   - запуск: `experiments/scripts/explain_batch.py`
   - проверка: `experiments/scripts/validate_explanations.py`
   - артефакты: `experiments/runs/<run_id>/interpretability/*`

## 11.3 Индекс исторической документации (`experiments/docs/general/*`)

1. [1.1.PRD.md](general/1.1.PRD.md) — исходная постановка и целевые ограничения.
2. [1.2.Additional-info.md](general/1.2.Additional-info.md) — дополнения к постановке и Q&A.
3. [3.1.Real-data-guide.md](general/3.1.Real-data-guide.md) — структура и особенности `data/real`.
4. [3.2.Implementation-options.md](general/3.2.Implementation-options.md) — варианты реализации пайплайна.
5. [4.1.Internet-review-raman-preprocessing.md](general/4.1.Internet-review-raman-preprocessing.md) — обзор preprocessing-практик из литературы.
6. [4.2.Shortlist-from-literature.md](general/4.2.Shortlist-from-literature.md) — shortlist литературы под проект.
7. [5.1.Concrete-experiment-plan-and-code-architecture.md](general/5.1.Concrete-experiment-plan-and-code-architecture.md) — конкретный план экспериментов и кодовой структуры.
8. [6.1.Implementation-progress-tracker.md](general/6.1.Implementation-progress-tracker.md) — журнал реализации и ключевых шагов.
9. [7.1.Tech-stack-and-uv-setup.md](general/7.1.Tech-stack-and-uv-setup.md) — зафиксированный стек и установка.
10. [8.1.Sanity-baseline-shortlist.md](general/8.1.Sanity-baseline-shortlist.md) — sanity baseline shortlist.
11. [8.2.Wave2-and-dual-window-results.md](general/8.2.Wave2-and-dual-window-results.md) — wave2 результаты и dual-window fusion.
12. [8.3.Protocol5-core-and-production-candidate.md](general/8.3.Protocol5-core-and-production-candidate.md) — protocol5 core и production-candidate.
13. [9.1.Next-steps-roadmap-from-literature.md](general/9.1.Next-steps-roadmap-from-literature.md) — roadmap следующей волны.
14. [10.1.DL-architectures-from-literature.md](general/10.1.DL-architectures-from-literature.md) — DL-архитектуры из литературы.
15. [10.2.DL-wave-results.md](general/10.2.DL-wave-results.md) — результаты DL-волны.
16. [10.3.Prioritized-hypotheses-after-dl-breakthrough.md](general/10.3.Prioritized-hypotheses-after-dl-breakthrough.md) — приоритизация гипотез после DL-прорыва.
17. [10.4.Full-DL-architecture-expansion-plan.md](general/10.4.Full-DL-architecture-expansion-plan.md) — полный план архитектурного расширения.
18. [10.5.Physically-grounded-Raman-XAI-smart-ticket.md](general/10.5.Physically-grounded-Raman-XAI-smart-ticket.md) — ticket по физически обоснованному XAI.
19. [10.6.Domain-validation-and-physics-aware-Raman-XAI-smart-ticket.md](general/10.6.Domain-validation-and-physics-aware-Raman-XAI-smart-ticket.md) — ticket по domain-validation/XAI.
20. [10.7.Raman-integration-tickets-from-BioMamba-ideas.md](general/10.7.Raman-integration-tickets-from-BioMamba-ideas.md) — интеграционные tickets по новым идеям.
21. [10.8.Raman-XAI-pipeline-usage.md](general/10.8.Raman-XAI-pipeline-usage.md) — практический usage XAI-пайплайна.
22. [10.10.f1-core-vs-multitaper-production.md](general/10.10.f1-core-vs-multitaper-production.md) — сравнение core vs multitaper по F1.
23. [10.11.moe-multitaper-production-comparison.md](general/10.11.moe-multitaper-production-comparison.md) — production-comparison для MoE+multitaper.

## 11.4 Легковесные артефакты

Сводные CSV/JSON артефакты вынесены в `experiments/docs/artifacts/`.
Это исторические итоги запусков без тяжёлых `runs/<run_id>` директорий.

Ключевые группы артефактов:

1. fusion sweeps (`fusion_*`).
2. DL wave / expansion summaries (`dl_*`).
3. protocol/p2 lock summaries (`protocol*`, `p2_*`, `production_candidate_*`).
4. holdout and strategy comparison (`multi_holdout_*`).

Быстрые ссылки:

1. [fusion_sweep_protocol5_core.csv](artifacts/fusion_sweep_protocol5_core.csv)
2. [dl_wave_summary_20260306.csv](artifacts/dl_wave_summary_20260306.csv)
3. [p2_locked_holdout_selection_20260306.json](artifacts/p2_locked_holdout_selection_20260306.json)
4. [multi_holdout_strategy_compare_summary_20260307.csv](artifacts/multi_holdout_strategy_compare_summary_20260307.csv)

## 11.5 Стандарт запуска команд в experimental-слое

Документационный canonical-формат: script-path.

```bash
.venv/bin/python experiments/scripts/<cli>.py ...
```

Эквивалентный формат: module-path.

```bash
.venv/bin/python -m experiments.scripts.<cli> ...
```

Все примеры запуска в новых docs используют пути внутри `experiments/*`:

1. конфиги: `experiments/configs/experiment/...`
2. run-артефакты: `experiments/runs/...`
