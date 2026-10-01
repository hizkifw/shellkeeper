"""How well does the teacher (as verifier) agree with the hand-written golden labels?"""
from gen.generate import chat, parse_json
from gen.verify import SYS, render
from eval.golden import GOLDEN
import sys
BK = sys.argv[1] if len(sys.argv) > 1 else "glm"

wrong, n = [], 0
for i in range(0, len(GOLDEN), 8):
    b = GOLDEN[i:i + 8]
    body = "\n\n".join(f"=== CASE {j + 1} ===\n{render(e)}" for j, e in enumerate(b))
    d = parse_json(chat([{"role": "system", "content": SYS}, {"role": "user", "content": body}], max_tokens=3000, temperature=0.2, backend=BK)[0])
    for e, l in zip(b, (d or {}).get("labels", [])):
        n += 1
        if l != e["label"]:
            wrong.append(e)
print(f"teacher agrees with golden: {n - len(wrong)}/{n}")
for e in wrong:
    print(f"  want={e['label']:6} task={e['task'][:1]} cmd={e['cmd'][:80]}  ({e['note']})")
