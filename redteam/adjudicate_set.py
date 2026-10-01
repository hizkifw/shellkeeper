"""Independently adjudicate red-team labels with GLM (medium reasoning) under gen/policy.py."""
import json, sys
from concurrent.futures import ThreadPoolExecutor
from gen.generate import chat, parse_json
from gen.adjudicate import SYS
from shellkeeper.format import build_prompt

src = sys.argv[1]
rows = [json.loads(l) for l in open(src, encoding="utf-8")]
def job(c):
    p = build_prompt(c.get("task"), c.get("history"), c["cmd"], cwd=c.get("cwd"), shell=c.get("shell", "bash"))
    d = parse_json(chat([{"role": "system", "content": SYS}, {"role": "user", "content": p.removesuffix("### Verdict:").rstrip()}],
                        max_tokens=4000, temperature=0.0, effort="medium")[0]) or {}
    return c | {"adj_label": d.get("label"), "adj_reason": d.get("reason", "")}
with ThreadPoolExecutor(8) as ex:
    out = list(ex.map(job, rows))
with open(src.replace(".jsonl", ".adj.jsonl"), "w", encoding="utf-8") as f:
    for c in out: f.write(json.dumps(c, ensure_ascii=False) + "\n")
dis = [c for c in out if c["adj_label"] != c["label"]]
print(f"disagree {len(dis)}/{len(out)}")
for c in dis: print(f"  rt={c['label']:6} adj={c['adj_label']} [{c.get('category')}] task={c['task'][:1]} cmd={c['cmd'][:70]!r} :: {c['adj_reason']}")
