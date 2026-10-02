"""Evaluate GGUF quants of the guard via llama-server.

Usage: .venv/Scripts/python -m bench.gguf_eval gguf/*.gguf [--backend vulkan|cpu] [--latency]
Scores are cached in gguf/results/<name>.jsonl as {"key", "p"}; --report prints the table.
P(unsafe) = softmax over the two label-token logprobs (read from the top-n list of the first generated token).
"""
import argparse, json, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bench.run import load_sets, auc  # noqa: E402
from shellkeeper.format import build_prompt  # noqa: E402

BIN = {"vulkan": ROOT / "tools/llama/vulkan/llama-server.exe", "cpu": ROOT / "tools/llama/cpu/llama-server.exe"}
RES = ROOT / "gguf/results"
PORT = 8137
SAFE_ID, UNSAFE_ID = None, None


def all_sets():
    sets = load_sets()
    rt2 = ROOT / "redteam/v2/adversarial_set.adj.jsonl"
    if rt2.exists():
        rows = []
        for l in open(rt2, encoding="utf-8"):
            c = json.loads(l)
            if c.get("adj_label") not in ("safe", "unsafe"):
                continue
            rows.append({"key": f"redteam2:{c['id']}", "cmd": c["cmd"], "y": int(c["adj_label"] == "unsafe"),
                         "prompt": build_prompt(c.get("task"), c.get("history"), c["cmd"], cwd=c.get("cwd"),
                                                shell=c.get("shell", "bash"))})
        sets["redteam2"] = rows
    return sets


class Server:
    def __init__(self, gguf, backend, parallel=8):
        try:  # refuse to share a port: a second server would fail to bind and we'd silently score the wrong model
            requests.get(f"http://127.0.0.1:{PORT}/health", timeout=1)
            raise RuntimeError(f"port {PORT} already in use; another llama-server is running (use --port)")
        except requests.ConnectionError:
            pass
        args = [str(BIN[backend]), "-m", str(gguf), "--port", str(PORT), "-c", str(2048 * parallel), "-np", str(parallel),
                "--no-webui", "-b", "2048", "-ub", "1024"]
        args += ["-ngl", "99"] if backend == "vulkan" else ["-ngl", "0", "-t", "6"]
        self.p = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(240):
            try:
                if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=1).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.5)
        raise RuntimeError("llama-server did not start")

    def close(self):
        self.p.terminate()
        self.p.wait(timeout=30)


def label_ids():
    global SAFE_ID, UNSAFE_ID
    r = requests.post(f"http://127.0.0.1:{PORT}/tokenize", json={"content": " safe unsafe"}).json()["tokens"]
    SAFE_ID, UNSAFE_ID = r
    return r


def score(prompt):
    r = requests.post(f"http://127.0.0.1:{PORT}/completion", timeout=120, json={
        "prompt": prompt, "n_predict": 1, "n_probs": 50, "temperature": 0, "post_sampling_probs": False,
        "cache_prompt": False, "samplers": [], "ignore_eos": True,
        # equal bias on both labels: forces a label token to be *sampled* (a byte-fragment token would make the
        # server withhold probs) without changing the reported pre-sampling probs or the safe/unsafe ratio
        "logit_bias": [[SAFE_ID, 100], [UNSAFE_ID, 100]]}).json()
    if "completion_probabilities" not in r:  # should not happen with ignore_eos; surface it instead of guessing
        raise RuntimeError(f"no probabilities in response: {str(r)[:200]}")
    top = {t["id"]: t["logprob"] for t in r["completion_probabilities"][0]["top_logprobs"]}
    ls, lu = top.get(SAFE_ID, -50.0), top.get(UNSAFE_ID, -50.0)  # absent from top-50 => negligible
    import math
    return 1 / (1 + math.exp(ls - lu))


def run(gguf, backend, sets, latency):
    name = Path(gguf).stem
    out = RES / f"{name}.jsonl"
    have = set()
    if out.exists():
        have = {json.loads(l)["key"] for l in open(out, encoding="utf-8")}
    rows = [r for v in sets.values() for r in v if r["key"] not in have]
    if not rows and not latency:
        return
    srv = Server(gguf, backend)
    try:
        label_ids()
        with ThreadPoolExecutor(8) as ex, open(out, "a", encoding="utf-8") as f:
            for r, p in zip(rows, ex.map(lambda r: score(r["prompt"]), rows)):
                f.write(json.dumps({"key": r["key"], "p": p}) + "\n")
        if latency:
            ps = [r["prompt"] for r in sets["golden"]] * 2
            score(ps[0])
            t = []
            for p in ps:
                s = time.perf_counter(); score(p); t.append((time.perf_counter() - s) * 1000)
            t.sort()
            with open(RES / "latency.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"model": name, "backend": backend, "p50_ms": t[len(t) // 2], "p90_ms": t[int(len(t) * .9)]}) + "\n")
            print(f"{name} [{backend}] latency p50 {t[len(t) // 2]:.1f} ms  p90 {t[int(len(t) * .9)]:.1f} ms")
    finally:
        srv.close()


def report(sets, names, ref="shellkeeper-0.6b-BF16"):
    def load(n):
        p = RES / f"{n}.jsonl"
        return {json.loads(l)["key"]: json.loads(l)["p"] for l in open(p, encoding="utf-8")} if p.exists() else {}
    refs = load(ref)
    hf = {}
    hp = ROOT / "bench/results/sk_runs_v2_best.jsonl"
    if hp.exists():
        hf = {json.loads(l)["key"]: json.loads(l)["p"] for l in open(hp, encoding="utf-8")}
    size = {n: (ROOT / "gguf" / f"{n}.gguf").stat().st_size / 1e6 for n in names if (ROOT / "gguf" / f"{n}.gguf").exists()}
    print(f"{'model':38} {'MB':>6} " + " ".join(f"{s + ' AUC/acc':>19}" for s in sets) + f" {'flips vs BF16':>14} {'mean|dp|':>9}")
    if hf:
        cells = []
        for s, rows in sets.items():
            g = [(hf[r["key"]], r["y"]) for r in rows if r["key"] in hf]
            cells.append(f"{auc(*zip(*g)):8.4f}/{sum((p >= .5) == y for p, y in g) / len(g):.3f}" if g else f"{'-':>19}")
        print(f"{'(HF transformers bf16)':38} {'':>6} " + " ".join(f"{c:>19}" for c in cells))
    for n in names:
        sc = load(n)
        if not sc:
            continue
        cells = []
        for s, rows in sets.items():
            g = [(sc[r["key"]], r["y"]) for r in rows if r["key"] in sc]
            cells.append(f"{auc(*zip(*g)):8.4f}/{sum((p >= .5) == y for p, y in g) / len(g):.3f}" if g else f"{'-':>19}")
        common = [k for k in sc if k in refs]
        flips = sum((sc[k] >= .5) != (refs[k] >= .5) for k in common)
        dp = sum(abs(sc[k] - refs[k]) for k in common) / max(1, len(common))
        print(f"{n:38} {size.get(n, 0):6.0f} " + " ".join(f"{c:>19}" for c in cells) + f" {flips:>9}/{len(common):<4} {dp:9.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ggufs", nargs="*")
    ap.add_argument("--backend", default="vulkan", choices=BIN)
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--report", nargs="*")
    ap.add_argument("--port", type=int, default=PORT)
    a = ap.parse_args()
    globals()["PORT"] = a.port
    RES.mkdir(parents=True, exist_ok=True)
    sets = all_sets()
    for g in a.ggufs:
        print(f"scoring {g}", flush=True)
        run(g, a.backend, sets, a.latency)
    if a.report is not None:
        names = a.report or sorted(p.stem for p in (ROOT / "gguf").glob("*.gguf"))
        report(sets, names)


if __name__ == "__main__":
    main()
