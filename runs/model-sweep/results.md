# Experiment: model-sweep

![F1 vs cost](frontier.png)

★ = on the Pareto frontier (no other configuration is both cheaper and more accurate). ⚠ = some steps failed; the numbers describe errors, not the model, so the configuration is left off the frontier and the chart. F1 is micro-averaged over all scored items; the interval is a 90% bootstrap over patents.

|  | configuration | F1 | F1 90% CI | precision | recall | $ / patent | citations verified | patents | failed steps |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ★ | `sonnet-5__p-2026-09-25.2` | 92% | 89%–96% | 87% | 96% | $0.34 | 99% | 5 | 0 |
| ★ | `sonnet-5__p-2026-09-25.2-no-rule3` | 87% | 82%–94% | 96% | 79% | $0.32 | 99% | 5 | 0 |
| ★ | `haiku-4-5__p-2026-09-25.2` | 68% | 57%–86% | 73% | 63% | $0.13 | 84% | 5 | 0 |
| ★ | `haiku-4-5__p-2026-09-25.2-no-rule3` | 63% | 50%–78% | 65% | 61% | $0.13 | 91% | 5 | 0 |
| ⚠ | `opus-5-5__p-2026-09-25.2` | 0% | 0%–0% | – | 0% | $0.00 | 0% | 5 | 77 |
| ⚠ | `opus-5-5__p-2026-09-25.2-no-rule3` | 0% | 0%–0% | – | 0% | $0.00 | 0% | 5 | 77 |

## F1 by finding type

| configuration | rejection | allowable | amendment | allowance_reason | examiners_amendment | claim |
|---|---:|---:|---:|---:|---:|---:|
| `sonnet-5__p-2026-09-25.2` | 93% | 100% | 84% | 100% | 100% | 97% |
| `sonnet-5__p-2026-09-25.2-no-rule3` | 81% | 100% | 82% | 80% | 100% | 94% |
| `haiku-4-5__p-2026-09-25.2` | 70% | 100% | 66% | 100% | 100% | 66% |
| `haiku-4-5__p-2026-09-25.2-no-rule3` | 76% | 100% | 44% | 100% | 100% | 67% |
| `opus-5-5__p-2026-09-25.2` | 0% | 0% | 0% | 0% | 0% | 0% |
| `opus-5-5__p-2026-09-25.2-no-rule3` | 0% | 0% | 0% | 0% | 0% | 0% |

## Configurations

| configuration | model | prompt | options |
|---|---|---|---|
| `sonnet-5__p-2026-09-25.2` | claude-sonnet-5 | 2026-09-25.2 | completeness_retry=True, strict=True, judge=False, max_tokens=8000 |
| `sonnet-5__p-2026-09-25.2-no-rule3` | claude-sonnet-5 | 2026-09-25.2-no-rule3 | completeness_retry=True, strict=True, judge=False, max_tokens=8000 |
| `haiku-4-5__p-2026-09-25.2` | claude-haiku-4-5 | 2026-09-25.2 | completeness_retry=True, strict=True, judge=False, max_tokens=8000 |
| `haiku-4-5__p-2026-09-25.2-no-rule3` | claude-haiku-4-5 | 2026-09-25.2-no-rule3 | completeness_retry=True, strict=True, judge=False, max_tokens=8000 |
| `opus-5-5__p-2026-09-25.2` | claude-opus-5-5 | 2026-09-25.2 | completeness_retry=True, strict=True, judge=False, max_tokens=8000 |
| `opus-5-5__p-2026-09-25.2-no-rule3` | claude-opus-5-5 | 2026-09-25.2-no-rule3 | completeness_retry=True, strict=True, judge=False, max_tokens=8000 |
