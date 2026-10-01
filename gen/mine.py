"""Score the training split with a trained guard and dump examples it still disagrees with.
These are either label noise (-> adjudicate & drop) or genuinely hard cases (-> seed variations).

Usage: .venv/Scripts/python -m gen.mine runs/v1/best [--split data/v1/train.jsonl]
Writes data/mined.jsonl
"""
import argparse, json
from pathlib import Path

from shellkeeper.guard import Guard

D = Path(__file__).resolve().parent.parent / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--split", default=str(D / "v1" / "train.jsonl"))
    ap.add_argument("--margin", type=float, default=0.5)
    a = ap.parse_args()
    g = Guard(a.model)
    rows = [json.loads(l) for l in open(a.split, encoding="utf-8")]
    rows.sort(key=lambda r: len(r["prompt"]))  # length-sorted batches = less padding
    out = []
    for i in range(0, len(rows), 32):
        b = rows[i:i + 32]
        for r, p in zip(b, g.score_prompts([r["prompt"] for r in b])):
            if abs(p - (r["label"] == "unsafe")) > a.margin:
                out.append(r | {"p": p})
    with open(D / "mined.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(out)}/{len(rows)} disagreements -> data/mined.jsonl")


if __name__ == "__main__":
    main()
