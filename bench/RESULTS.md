# Benchmark (2026-10-02)

Reproduce with `.venv/Scripts/python -m bench.run --report`.

## Systems

| system | kind | context-aware | latency |
|---|---|---|---|
| sh-guard 0.1.10 | AST + rules | no | ~0.04 ms |
| dcg 0.15.2 | rule packs (destructive ops only) | no | ~28 ms (mostly process startup) |
| GLM-5.3-flash zero-shot | LLM judge, same policy | yes | ~1.0–1.4 s |
| Qwen3.8-27B (q4) zero-shot | LLM judge, same policy | yes | ~8 s |
| shellkeeper v1 / v2 | 0.6B fine-tune | yes | ~20 ms (7900 XTX) |
| shellkeeper `-noctx` | v1/v2 with task + history stripped | ablation | ~20 ms |

## AUC (accuracy at the default threshold)

| set | sh-guard | dcg | Qwen-27B | GLM* | sk v1 | **sk v2** | sk v2 no-ctx |
|---|---|---|---|---|---|---|---|
| test (frozen, n=1541) | 0.675 (0.59) | 0.594 (0.63) | – | 0.962 (0.96) | 0.987 (0.953) | **0.990 (0.954)** | 0.906 (0.83) |
| test_new (n=862) | 0.543 (0.52) | 0.581 (0.59) | – | 0.969 (0.97) | 0.898 (0.82) | **0.992 (0.959)** | 0.974 (0.92) |
| golden (hand-written, n=57) | 0.720 (0.61) | 0.620 (0.63) | 0.981 (0.98) | 1.000 | 0.996 (0.965) | **1.000 (1.000)** | 0.948 (0.88) |
| red-team, adjudicated (n=240) | 0.637 (0.53) | 0.647 (0.74) | 0.884 (0.91) | 0.961 (0.95) | 0.892 (0.82) | **0.953 (0.887)** | 0.618 (0.53) |

\* GLM generated and verified the training labels, and it adjudicated the red-team labels. Its scores are
therefore circular: they are close to the answer key, not an independent baseline.

## Caveats

- The test, test_new and golden sets were built around **context-dependent** cases, so they disadvantage
  the context-free tools by design. Even so, the rule-based tools trade off badly on these sets:
  - sh-guard blocks 52–94% of safe commands.
  - dcg misses 57–71% of unsafe ones.
- v2's training data targeted the failure categories found by red-teaming v1. Its red-team score is therefore
  optimistic, and a fresh red-team round on v2 is the honest test.
- At threshold 0.5, v2 catches 71% of red-team unsafe commands with a 4% false-positive rate (FPR).
  At 0.05 it catches 90% with 14% FPR.
- The remaining misses are config-level weakening that technically follows a vague user request but goes
  beyond its scope. Examples: `Host *` when the user said "the box"; `core.hooksPath /tmp/...`.

## Takeaways

- **Context is the main source of signal.** Stripping the task and history from the same v2 model takes
  red-team AUC from 0.953 to 0.618 and raises test FPR from 5% to 30%.
- v2 beats the 27B judge on the red-team set (AUC 0.953 vs 0.884) and is about 400× faster.
