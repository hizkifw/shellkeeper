"""Prompt format shared by data building, training and inference.

The classifier sees: the user's request(s), a compact session transcript
(previous commands + truncated outputs), and the candidate command last.
The candidate goes at the very end so that, in an agent loop, everything
before it is a stable prefix that can be KV-cached between calls.
"""

LABELS = ("safe", "unsafe")
# Leading space matters for BPE tokenizers: " safe" / " unsafe" are what we score.
LABEL_TEXT = {"safe": " safe", "unsafe": " unsafe"}

MAX_OUT_LINES = 6
MAX_OUT_CHARS = 400
MAX_HISTORY = 12


def _clip_output(out: str) -> str:
    out = (out or "").rstrip()
    if not out:
        return ""
    lines = out.splitlines()
    if len(lines) > MAX_OUT_LINES:
        keep = MAX_OUT_LINES - 1
        lines = lines[: keep - 1] + [f"... ({len(lines) - keep} more lines)"] + lines[-1:]
    out = "\n".join(lines)
    if len(out) > MAX_OUT_CHARS:
        out = out[: MAX_OUT_CHARS - 20] + " ...[truncated]"
    return out


def label_token_ids(tokenizer) -> dict:
    """Map label -> single token id. Fails if a label isn't exactly one token
    (verified single-token for Qwen3/3.5, SmolLM2, LFM2, Gemma3 tokenizers)."""
    ids = {}
    for lab, text in LABEL_TEXT.items():
        toks = tokenizer.encode(text, add_special_tokens=False)
        assert len(toks) == 1, f"label {text!r} is {len(toks)} tokens for this tokenizer: {toks}"
        ids[lab] = toks[0]
    # the prompt must not merge with the label's leading space at the boundary
    p = tokenizer.encode("### Verdict:", add_special_tokens=False)
    pl = tokenizer.encode("### Verdict:" + LABEL_TEXT["unsafe"], add_special_tokens=False)
    assert pl[: len(p)] == p and pl[len(p):] == [ids["unsafe"]], "prompt/label tokenization boundary mismatch"
    return ids


def build_prompt(task, history, command, cwd=None, shell=None) -> str:
    """task: str or list[str] (user messages, oldest first)
    history: list of {"cmd": str, "out": str} (oldest first)
    command: the candidate command to classify
    """
    if isinstance(task, (list, tuple)):
        task = "\n".join(f"- {t}" for t in task if t)
    task = (task or "").strip() or "(none given)"

    parts = ["### Task", task, "### Session"]
    env = []
    if shell:
        env.append(f"shell: {shell}")
    if cwd:
        env.append(f"cwd: {cwd}")
    if env:
        parts.append(" | ".join(env))
    hist = list(history or [])[-MAX_HISTORY:]
    if not hist:
        parts.append("(no previous commands)")
    for h in hist:
        parts.append(f"$ {h['cmd'].strip()}")
        o = _clip_output(h.get("out", ""))
        if o:
            parts.append(o)
    parts += ["### Command", command.strip(), "### Verdict:"]
    return "\n".join(parts)
