"""Join generated examples with verifier labels, filter, dedupe, split by scenario.

Usage: python build_dataset.py [--no-verify]
Writes data/{train,val,test}.jsonl with {"prompt", "label", "id", "mode", "cmd"}.
"""
import argparse, hashlib, json, random
from collections import Counter
from pathlib import Path

from gen.flatten import load_all
from shellkeeper.format import build_prompt

D = Path(__file__).resolve().parent / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-verify", action="store_true", help="use generator labels without agreement filter")
    ap.add_argument("--val", type=float, default=0.06)
    ap.add_argument("--test", type=float, default=0.06)
    args = ap.parse_args()

    ex = load_all()
    ver = {}
    vp = D / "verified.jsonl"
    if vp.exists():
        for l in open(vp, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                ver.setdefault(r["id"], set()).add(r["v_label"])

    kept, stats = [], Counter()
    for e in ex:
        if not args.no_verify:
            v = ver.get(e["id"])
            if v is None:
                stats["unverified"] += 1
                continue
            if v != {e["label"]}:  # every verifier must agree with the generator
                stats[f"disagree_{e['mode']}"] += 1
                continue
        stats["kept"] += 1
        kept.append(e)

    # dedupe identical (prompt) with conflicting/duplicate labels
    by_prompt = {}
    for e in kept:
        e["prompt"] = build_prompt(e["task"], e["history"], e["cmd"], cwd=e["cwd"], shell=e["shell"])
        by_prompt.setdefault(e["prompt"], []).append(e)
    final = []
    for p, es in by_prompt.items():
        if len({x["label"] for x in es}) > 1:
            stats["conflict_dropped"] += len(es)
            continue
        final.append(es[0])

    sg = D / "split_groups.json"
    if sg.exists():
        # frozen split: known groups keep their split so test numbers stay comparable across runs;
        # new groups go to train, except ~8% of new (non-hardmine) scenarios held out as test_new
        split = json.load(open(sg))
        for g in sorted({e["group"] for e in final} - split.keys()):
            mode = next(e["mode"] for e in final if e["group"] == g)
            split[g] = "test_new" if mode != "hardmine" and int(hashlib.md5(g.encode()).hexdigest(), 16) % 100 < 8 else "train"
    else:
        groups = sorted({e["group"] for e in final})
        random.Random(42).shuffle(groups)
        nv, nt = int(len(groups) * args.val), int(len(groups) * args.test)
        split = {g: "val" for g in groups[:nv]} | {g: "test" for g in groups[nv:nv + nt]}
        json.dump(split | {g: "train" for g in groups[nv + nt:]}, open(sg, "w"))
    drop = set()
    dp = D / "dropped_ids.json"
    if dp.exists():
        drop = set(json.load(open(dp)))
    outs = {"train": [], "val": [], "test": [], "test_new": []}
    for e in final:
        s = split.get(e["group"], "train")
        if s == "train" and e["id"] in drop:  # adjudicated-wrong labels: removed from train only
            stats["adjudicated_dropped"] += 1
            continue
        outs[s].append(e)

    for k, rows in outs.items():
        random.Random(k).shuffle(rows)
        with open(D / f"{k}.jsonl", "w", encoding="utf-8") as f:
            for e in rows:
                f.write(json.dumps({"prompt": e["prompt"], "label": e["label"], "id": e["id"],
                                    "mode": e["mode"], "cmd": e["cmd"]}, ensure_ascii=False) + "\n")
        c = Counter(e["label"] for e in rows)
        print(f"{k}: {len(rows)}  safe={c['safe']} unsafe={c['unsafe']}")
    print(dict(stats))


if __name__ == "__main__":
    main()
