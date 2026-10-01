"""Compare guard checkpoints across eval sets.

Usage: .venv/Scripts/python -m eval.compare runs/v1/best runs/v2/best
Sets: frozen v1 test split, test_new (held-out targeted data), golden, adjudicated red-team set.
"""
import json, sys
from pathlib import Path

import torch

from shellkeeper.format import build_prompt
from shellkeeper.guard import Guard
from eval.golden import GOLDEN
from train import metrics

ROOT = Path(__file__).resolve().parent.parent


def load_set(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return [(r["prompt"] if "prompt" in r else
             build_prompt(r.get("task"), r.get("history"), r["cmd"], cwd=r.get("cwd"), shell=r.get("shell", "bash")),
             int(r["label"] == "unsafe")) for r in rows]


SETS = {
    "test (frozen v1)": lambda: load_set(ROOT / "data/v1/test.jsonl"),
    "test_new": lambda: load_set(ROOT / "data/test_new.jsonl"),
    "golden": lambda: [(build_prompt(g["task"], g["history"], g["cmd"], cwd=g["cwd"], shell=g["shell"]),
                        int(g["label"] == "unsafe")) for g in GOLDEN],
    "red-team (adjudicated)": lambda: load_set(ROOT / "redteam/adversarial_adj.jsonl"),
}


def main():
    data = {}
    for k, f in SETS.items():
        try:
            data[k] = f()
        except FileNotFoundError:  # e.g. private red-team set absent in a public clone
            print(f"skipping {k}: file not found")
    print(f"{'set':24} {'model':16} {'acc':>6} {'auc':>6} {'recall':>7} {'fpr':>6} {'conf_FN':>8}  n")
    for path in sys.argv[1:]:
        g = Guard(path)
        for name, rows in data.items():
            ps = []
            for i in range(0, len(rows), 32):
                ps += g.score_prompts([p for p, _ in rows[i:i + 32]])
            ys = [y for _, y in rows]
            m = metrics(ps, ys)
            cfn = sum(1 for p, y in zip(ps, ys) if y and p < 0.1)
            print(f"{name:24} {path[-16:]:16} {m['acc']:6.3f} {m['auc']:6.3f} {m['recall_unsafe']:7.3f} "
                  f"{m['fpr']:6.3f} {cfn:8d}  {len(ys)}")
        del g
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
