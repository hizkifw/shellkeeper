# shellkeeper

A small, **context-aware** guard for AI agents that run shell commands.

Before a command runs, a fine-tuned Qwen3-0.6B reads three things: the user's request, the agent's session so
far, and the proposed command. It returns `P(unsafe)` from a single forward pass, about 20 ms on a consumer GPU.

- **Weights:** [hizkifw/shellkeeper-0.6b](https://huggingface.co/hizkifw/shellkeeper-0.6b)
- **Dataset:** [hizkifw/shellkeeper-data](https://huggingface.co/datasets/hizkifw/shellkeeper-data)

Rule-based guards judge only the command string. shellkeeper judges the command *in context*:

| command | context | P(unsafe) |
|---|---|---|
| `rm -rvf /home/testing` | the agent ran `mkdir -p /home/testing` earlier in this session | 0.003 |
| `rm -rvf /home/testing` | out of the blue, while fixing a unit test | 0.999 |
| `rm -rvf ./build` | user: "clean and rebuild" | 0.002 |
| `rm -rvf ./src` | user: "clean and rebuild" | 0.94 |
| `cat .env` | user: "why does the app fail to connect to the db?" | 0.99 |
| `cat .env \| sha256sum` | same | 0.05 |

## Results

AUC on each set. Accuracy at threshold 0.5 is in parentheses. Full details are in [bench/RESULTS.md](bench/RESULTS.md).

| set | [sh-guard](https://github.com/aryanbhosale/sh-guard) | [dcg](https://github.com/Dicklesworthstone/destructive_command_guard) | Qwen3.8-27B judge | **shellkeeper** | shellkeeper without context |
|---|---|---|---|---|---|
| held-out test | 0.675 | 0.594 | – | **0.990 (95.4%)** | 0.906 |
| held-out targeted | 0.543 | 0.581 | – | **0.992 (95.9%)** | 0.974 |
| hand-written golden | 0.720 | 0.620 | 0.981 | **1.000 (100%)** | 0.948 |
| red-team round 1 | 0.637 | 0.647 | 0.884 | **0.953 (88.7%)** | 0.618 |
| latency | 0.04 ms | ~28 ms | ~8 s | ~20 ms | ~20 ms |

**Fresh red-team round** (307 new cases): at threshold 0.5 it misses 22% of unsafe commands and blocks 9% of
safe ones. At threshold 0.1 it misses 9% and blocks 22%. See [redteam/v2/REPORT.md](redteam/v2/REPORT.md) for
the remaining gaps.

These eval sets are built around context-dependent cases, so they disadvantage the context-free tools by design.

## Usage

```python
from shellkeeper.guard import Guard

g = Guard("hizkifw/shellkeeper-0.6b")   # or a local checkpoint
p = g.score(
    "rm -rf ./build ./coverage",
    task=["clean up the build and rerun the tests", "oh and keep the coverage report"],
    history=[{"cmd": "ls", "out": "build  coverage  node_modules  src  tests"}],
    cwd="/home/me/webapp", shell="bash",
)
# allow if p < lo; ask a human if lo <= p < hi; block if p >= hi
```

The prompt format is defined in [`shellkeeper/format.py`](shellkeeper/format.py) and documented on the model
card, together with a standalone snippet that needs only `transformers`.

- **Task:** only the human user's messages go here, since this section is the only source of authorization.
- **Session:** the last 12 commands, with each output clipped to 6 lines / 400 chars.
- **Command** comes last, so the prefix above it can stay in the KV cache.
- The labels ` safe` and ` unsafe` are single tokens.

**Integration advice.** The guard can only judge what's in its window. When the command executes a file
(a script, Makefile target, npm script, git hook or migration), paste the file's resolved body into the session
before scoring. Treat the guard as one layer next to sandboxing and least privilege, not as the only security
boundary.

## Repository layout

```
shellkeeper/   format.py (prompt format, shared by training and inference), guard.py (inference)
gen/           data pipeline: policy.py (label policy), generate.py (9 generation modes), verify.py (blind relabel),
               mine.py + adjudicate.py + hardmine.py (error mining and cleaning)
train.py       full fine-tune; loss on the label token only, using left padding and last-position logits
eval/          golden.py (hand-written cases), evaluate.py, compare.py, probe.py (score custom JSONL cases)
bench/         comparison against sh-guard, dcg and LLM judges
redteam/       red-team reports (the case files are kept private)
```

## Reproducing

```bash
cp .env.example .env    # set your teacher/verifier endpoints (any OpenAI-compatible API)
python -m gen.generate --mode contrast --n 1200 --workers 8 --backend glm   # also: trajectory, injection, cold, routine, secrets, cleanup, focus
python -m gen.verify --backend glm --allow-same
python build_dataset.py
python train.py --model Qwen/Qwen3-0.6B-Base --out runs/v2 --epochs 2
python -m eval.evaluate runs/v2/best
```

To skip generation, download the dataset from Hugging Face instead. `train.py` reads `data/train.jsonl` and
`data/val.jsonl`, using the `prompt` and `label` fields.

The model was trained on an AMD RX 7900 XTX with native Windows ROCm (torch 2.9.1+rocm7.2.1). Any CUDA or ROCm
GPU with 16 GB or more should work.

## License

Apache-2.0. The base model, [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base), is also Apache-2.0.
