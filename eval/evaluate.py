"""Evaluate a trained guard on the held-out test split and the golden set.

Usage: .venv/Scripts/python -m eval.evaluate runs/v1/best [--split test]
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

from shellkeeper.guard import Guard
from shellkeeper.format import build_prompt
from eval.golden import GOLDEN
from train import metrics

D = Path(__file__).resolve().parent.parent / "data"


def score_all(g, prompts, bs=32):
    out = []
    for i in range(0, len(prompts), bs):
        out += g.score_prompts(prompts[i:i + bs])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    g = Guard(a.model)

    path = Path(a.split) if a.split.endswith(".jsonl") else D / f"{a.split}.jsonl"
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    ps = score_all(g, [r["prompt"] for r in rows])
    ys = [int(r["label"] == "unsafe") for r in rows]
    print(f"== {a.split} overall", fmt(metrics(ps, ys)))
    by = defaultdict(list)
    for r, p, y in zip(rows, ps, ys):
        by[r["mode"]].append((p, y))
    for m, v in sorted(by.items()):
        print(f"   {m:11}", fmt(metrics([p for p, _ in v], [y for _, y in v])))

    print("\n== thresholds (test): FPR at a given unsafe-recall")
    for target in (0.90, 0.95, 0.98):
        thr = sorted(p for p, y in zip(ps, ys) if y)[int((1 - target) * sum(ys))]
        m = metrics(ps, ys, thr)
        print(f"   recall>={target:.2f}: thr={thr:.3f} fpr={m['fpr']:.3f}")

    gp = score_all(g, [build_prompt(x["task"], x["history"], x["cmd"], cwd=x["cwd"], shell=x["shell"]) for x in GOLDEN])
    gy = [int(x["label"] == "unsafe") for x in GOLDEN]
    print("\n== golden", fmt(metrics(gp, gy)))
    for x, p, y in zip(GOLDEN, gp, gy):
        if (p >= 0.5) != bool(y):
            print(f"   MISS want={x['label']:6} p={p:.3f} task={x['task'][:1]} cmd={x['cmd'][:70]}")

    errs = sorted(((abs(p - y), r, p) for r, p, y in zip(rows, ps, ys)), key=lambda t: -t[0])[:15]
    print("\n== worst test errors")
    for e, r, p in errs:
        if e < 0.5:
            break
        print(f"   want={r['label']:6} p={p:.3f} [{r['mode']}] {r['cmd'][:90]}")


def fmt(m):
    return " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items())


if __name__ == "__main__":
    main()
