"""Flatten raw scenarios (data/raw/*.jsonl) into one example per (context, command)."""
import json
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def _hist(h):
    out = []
    for x in h or []:
        if isinstance(x, dict) and x.get("cmd"):
            out.append({"cmd": str(x["cmd"]), "out": str(x.get("out") or "")})
    return out


def _task(t):
    if isinstance(t, str):
        return [t] if t.strip() else []
    return [str(x) for x in (t or []) if str(x).strip()]


def flatten_record(rec):
    d, mode, gid = rec["data"], rec["mode"], rec["id"]
    shell = rec["seed"].get("shell")
    teacher = rec.get("teacher", "qwen")
    ex = []

    def add(i, task, cwd, hist, cmd, label, why):
        ex.append({"id": f"{gid}-{i}", "group": gid, "mode": mode, "teacher": teacher, "shell": shell, "task": _task(task),
                   "cwd": cwd or None, "history": _hist(hist), "cmd": str(cmd), "label": label, "why": why or ""})

    if mode in ("trajectory", "injection", "routine", "secrets", "cleanup", "hardmine", "focus"):
        for i, c in enumerate(d["candidates"]):
            add(i, d["task"], d.get("cwd"), d.get("history"), c["cmd"], c["label"], c.get("why"))
    elif mode == "contrast":
        ctxs = d["contexts"]
        for i, c in enumerate(ctxs):
            add(f"c{i}", c.get("task"), c.get("cwd"), c.get("history"), d["command"], c["label"], c.get("why"))
        c0 = ctxs[0]
        for i, v in enumerate(d.get("variants") or []):
            if v["cmd"].strip() == d["command"].strip():
                continue  # duplicate of context 0
            add(f"v{i}", c0.get("task"), c0.get("cwd"), c0.get("history"), v["cmd"], v["label"], v.get("why"))
    elif mode == "cold":
        for i, it in enumerate(d["items"]):
            add(i, it.get("task"), it.get("cwd"), [], it["cmd"], it["label"], it.get("why"))
    return ex


def load_all():
    ex = []
    for p in sorted(RAW.glob("*.jsonl")):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line:
                ex.extend(flatten_record(json.loads(line)))
    return ex
