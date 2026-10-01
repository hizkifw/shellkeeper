"""Benchmark shellkeeper against other command guards on the same eval sets.

Usage (from project root, .venv python):
  .venv/Scripts/python -m bench.run --systems shguard,sk:runs/v1/best,sk-noctx:runs/v1/best,llm:glm,llm:qwen
  .venv/Scripts/python -m bench.run --report

Each system's scores are cached in bench/results/<system>.jsonl as {"key", "p", "ms"}; "p" is a risk score
in [0, 1] (higher = more dangerous). Re-running only scores missing keys.

Systems:
  shguard        - sh-guard 0.1.10 (rule/AST based, command only). p = score / 100.
  dcg            - destructive_command_guard 0.15.2 (rule packs, command only; scope = destructive ops). deny=1.
  sk:<ckpt>      - shellkeeper with full context.
  sk-noctx:<ckpt>- shellkeeper with task + history stripped (ablation: what does context buy?).
  llm:<backend>  - zero-shot LLM judge given the same policy + full context (glm = GLM-5.3-flash, qwen = Qwen3.8-27B).
                   NOTE: GLM generated and verified the training labels, so its agreement is inflated.
"""
import argparse, json, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "bench" / "results"
sys.path.insert(0, str(ROOT))

from shellkeeper.format import build_prompt  # noqa: E402
from eval.golden import GOLDEN  # noqa: E402


def load_sets():
    sets = {}

    def from_split(path, name):
        rows = []
        for l in open(path, encoding="utf-8"):
            r = json.loads(l)
            # recover the raw command/context from the prompt (single source of truth = prompt)
            rows.append({"key": f"{name}:{r['id']}", "prompt": r["prompt"], "cmd": r["cmd"],
                         "y": int(r["label"] == "unsafe"), "mode": r["mode"]})
        return rows

    def from_cases(cases, name):
        rows = []
        for i, c in enumerate(cases):
            rows.append({"key": f"{name}:{c.get('id', i)}", "cmd": c["cmd"], "y": int(c["label"] == "unsafe"),
                         "prompt": build_prompt(c.get("task"), c.get("history"), c["cmd"], cwd=c.get("cwd"),
                                                shell=c.get("shell", "bash")),
                         "prompt_noctx": build_prompt(None, None, c["cmd"], cwd=c.get("cwd"), shell=c.get("shell", "bash")),
                         "mode": c.get("category", "golden")})
        return rows

    sets["test"] = from_split(ROOT / "data/v1/test.jsonl", "test")
    sets["test_new"] = from_split(ROOT / "data/test_new.jsonl", "test_new")
    sets["golden"] = from_cases(GOLDEN, "golden")
    rt = ROOT / "redteam/adversarial_adj.jsonl"  # private (not in the public repo)
    if rt.exists():
        sets["redteam"] = from_cases([json.loads(l) for l in open(rt, encoding="utf-8")], "redteam")
    return sets


def noctx_prompt(r):
    if "prompt_noctx" in r:
        return r["prompt_noctx"]
    # strip task + history from a rendered prompt, keep shell/cwd line and command
    p = r["prompt"]
    sess = p.split("### Session\n", 1)[1]
    env = sess.split("\n", 1)[0] if sess.startswith(("shell:", "cwd:")) else None
    return "\n".join(["### Task", "(none given)", "### Session"] + ([env] if env else []) +
                     ["(no previous commands)", "### Command", r["cmd"].strip(), "### Verdict:"])


def cached(system):
    p = RES / f"{system.replace(':', '_').replace('/', '_')}.jsonl"
    have = {}
    if p.exists():
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            have[r["key"]] = r
    return p, have


def run_shguard(rows, path, have):
    todo = [r for r in rows if r["key"] not in have]
    if not todo:
        return
    tmp_in, tmp_out = RES / "_sg_in.jsonl", RES / "_sg_out.jsonl"
    with open(tmp_in, "w", encoding="utf-8") as f:
        for r in todo:
            f.write(json.dumps({"key": r["key"], "cmd": r["cmd"]}) + "\n")
    subprocess.run([str(ROOT / "bench/.venv-shguard/Scripts/python"), str(ROOT / "bench/shguard_score.py"),
                    str(tmp_in), str(tmp_out)], check=True)
    with open(path, "a", encoding="utf-8") as f:
        for l in open(tmp_out, encoding="utf-8"):
            r = json.loads(l)
            f.write(json.dumps({"key": r["key"], "p": r["score"] / 100, "ms": r["ms"], "level": r["level"]}) + "\n")


def run_dcg(rows, path, have):
    """dcg v0.15.2 (rule packs, default config). deny -> 1, warn/ask -> 0.5, allow -> 0.
    Dialect follows the session shell (posix for bash/zsh/sh = the Bash hook path, ps for powershell)."""
    exe = str(ROOT / "bench/dcg/dcg.exe")
    todo = [r for r in rows if r["key"] not in have]

    def job(r):
        dialect = "ps" if "shell: powershell" in r["prompt"] else "posix"
        t = time.perf_counter()
        cp = subprocess.run([exe, "test", "--stdin", "--format", "json", "--dialect", dialect, "--no-suggestions"],
                            input=r["cmd"].encode("utf-8"), capture_output=True)
        ms = (time.perf_counter() - t) * 1000  # includes process startup; dcg's in-hook cost is lower
        try:
            dec = json.loads(cp.stdout.decode("utf-8", "replace"))["decision"]
        except (json.JSONDecodeError, KeyError):
            dec = "error"
        return r, {"deny": 1.0, "allow": 0.0}.get(dec, 0.5), ms, dec

    with ThreadPoolExecutor(8) as ex, open(path, "a", encoding="utf-8") as f:
        for r, p, ms, dec in ex.map(job, todo):
            f.write(json.dumps({"key": r["key"], "p": p, "ms": ms, "decision": dec}) + "\n")


def run_sk(rows, path, have, ckpt, noctx):
    import torch
    from shellkeeper.guard import Guard
    todo = [r for r in rows if r["key"] not in have]
    if not todo:
        return
    g = Guard(ckpt)
    g.score("ls")
    with open(path, "a", encoding="utf-8") as f:
        for r in todo:  # one at a time to measure per-call latency honestly
            pr = noctx_prompt(r) if noctx else r["prompt"]
            torch.cuda.synchronize(); t = time.perf_counter()
            p = g.score_prompts([pr])[0]
            torch.cuda.synchronize(); ms = (time.perf_counter() - t) * 1000
            f.write(json.dumps({"key": r["key"], "p": p, "ms": ms}) + "\n")


def run_llm(rows, path, have, backend):
    from gen.generate import chat, parse_json
    from gen.adjudicate import SYS
    todo = [r for r in rows if r["key"] not in have]

    def job(r):
        t = time.perf_counter()
        text, _ = chat([{"role": "system", "content": SYS},
                        {"role": "user", "content": r["prompt"].removesuffix("### Verdict:").rstrip()}],
                       max_tokens=600 if backend == "glm" else 2000, temperature=0.0, backend=backend, effort="low")
        d = parse_json(text) or {}
        lab = d.get("label")
        return r, ({"safe": 0.0, "unsafe": 1.0}.get(lab)), (time.perf_counter() - t) * 1000

    with ThreadPoolExecutor(12 if backend == "glm" else 4) as ex, open(path, "a", encoding="utf-8") as f:
        for r, p, ms in ex.map(job, todo):
            if p is not None:
                f.write(json.dumps({"key": r["key"], "p": p, "ms": ms}) + "\n")
                f.flush()


def auc(ps, ys):
    pos = [p for p, y in zip(ps, ys) if y]; neg = [p for p, y in zip(ps, ys) if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def report(sets, systems, thresholds):
    print(f"\n{'set':10} {'system':28} {'n':>5} {'AUC':>6} {'thr':>5} {'acc':>6} {'recall':>7} {'FPR':>6} {'p50 ms':>8}")
    for sname, rows in sets.items():
        for s in systems:
            _, have = cached(s)
            got = [(have[r["key"]]["p"], r["y"], have[r["key"]]["ms"]) for r in rows if r["key"] in have]
            if not got:
                continue
            ps, ys, ms = zip(*got)
            for thr in thresholds.get(s.split(":")[0], [0.5]):
                tp = sum(p >= thr and y for p, y in zip(ps, ys)); fp = sum(p >= thr and not y for p, y in zip(ps, ys))
                npos = sum(ys); nneg = len(ys) - npos
                acc = (tp + (nneg - fp)) / len(ys)
                print(f"{sname:10} {s[-28:]:28} {len(ys):5d} {auc(ps, ys):6.3f} {thr:5.2f} {acc:6.3f} "
                      f"{tp / max(1, npos):7.3f} {fp / max(1, nneg):6.3f} {sorted(ms)[len(ms) // 2]:8.2f}")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="")
    ap.add_argument("--sets", default="test,test_new,golden,redteam")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--report-systems", default="shguard,dcg,sk:runs/v1/best,sk-noctx:runs/v1/best,sk:runs/v2/best,llm:glm,llm:qwen")
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    sets = {k: v for k, v in load_sets().items() if k in a.sets.split(",")}
    rows = [r for v in sets.values() for r in v]
    for s in filter(None, a.systems.split(",")):
        path, have = cached(s)
        kind, _, arg = s.partition(":")
        print(f"scoring {s} ({sum(r['key'] not in have for r in rows)} new)", flush=True)
        if kind == "shguard":
            run_shguard(rows, path, have)
        elif kind == "dcg":
            run_dcg(rows, path, have)
        elif kind == "sk":
            run_sk(rows, path, have, arg, noctx=False)
        elif kind == "sk-noctx":
            run_sk(rows, path, have, arg, noctx=True)
        elif kind == "llm":
            run_llm(rows, path, have, arg)
    if a.report:
        # sh-guard: >=0.21 = caution+ (ask), >=0.51 = danger+ ; LLM judges are binary
        report(sets, a.report_systems.split(","), {"shguard": [0.21, 0.51]})


if __name__ == "__main__":
    main()
