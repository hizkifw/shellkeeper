"""Generate fresh scenarios around confirmed-hard training examples (data/hard_seeds.jsonl).
Each seed -> a new session in a different setting that tests the same distinction, with near-miss candidates.

Usage: python -m gen.hardmine [--per-seed 1] [--workers 8]
Appends to data/raw/hardmine.jsonl (mode "hardmine"; always assigned to the train split).
"""
import argparse, json, random, threading, uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from gen.generate import SYS, HIST_SPEC, chat, parse_json, valid
from gen import seeds

D = Path(__file__).resolve().parent.parent / "data"


def prompt(seed, domain, shell):
    case = seed["prompt"].removesuffix("### Verdict:").rstrip()
    return f"""A small guard model gets this case WRONG. Correct label: {seed['label']}. Why: {seed.get('reason', '')}

--- hard case ---
{case}
--- end ---

Identify the subtle distinction that makes this case {seed['label']} (what in the task/history/command decides it).
Then write a NEW scenario that tests the same distinction in a different setting: {domain}, shell: {shell}.
Do not copy names, paths or wording. Give 6 candidate next commands: the analogue of the hard case, plus near-misses
on both sides of the line (small edits that flip or keep the label), labeled by the policy for THIS new context.

JSON schema:
{{"distinction": "<=20 words",
  "task": ["1-2 user messages"], "cwd": "/abs/path",
  "history": [{HIST_SPEC}, ... 1 to 6],
  "candidates": [{{"cmd": "...", "label": "safe|unsafe", "why": "<=15 words"}}, ... exactly 6]}}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    seeds_ = [json.loads(l) for l in open(D / "hard_seeds.jsonl", encoding="utf-8")]
    jobs = [s for s in seeds_ for _ in range(a.per_seed)]
    rng = random.Random(7)
    out = D / "raw" / "hardmine.jsonl"
    lock, ok = threading.Lock(), [0]

    def job(s):
        dom, sh = rng.choice(seeds.DOMAINS), rng.choice(seeds.SHELLS)
        text, _ = chat([{"role": "system", "content": SYS}, {"role": "user", "content": prompt(s, dom, sh)}],
                       max_tokens=2500, effort="medium")
        d = parse_json(text)
        if d is None or not valid("hardmine", d):
            return
        rec = {"id": uuid.uuid4().hex[:12], "mode": "hardmine", "teacher": "glm",
               "seed": {"domain": dom, "shell": sh, "from": s["id"]}, "data": d}
        with lock, open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ok[0] += 1

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(job, jobs))
    print(f"hardmine: {ok[0]}/{len(jobs)} scenarios written")


if __name__ == "__main__":
    main()
