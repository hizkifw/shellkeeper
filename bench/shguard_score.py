"""Runs inside bench/.venv-shguard. Reads JSONL {"key","cmd"}, writes JSONL {"key","score","level","ms"}."""
import json, sys, time

from sh_guard import classify

with open(sys.argv[2], "w", encoding="utf-8") as out:
    for line in open(sys.argv[1], encoding="utf-8"):
        r = json.loads(line)
        t = time.perf_counter()
        c = classify(r["cmd"])
        ms = (time.perf_counter() - t) * 1000
        out.write(json.dumps({"key": r["key"], "score": c["score"], "level": c["level"], "ms": ms}) + "\n")
