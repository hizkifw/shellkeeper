"""Inference: one forward pass, read the next-token logits for " safe" / " unsafe".

    from shellkeeper.guard import Guard
    g = Guard("runs/v1/best")
    p = g.score(task="clean and rebuild", history=[{"cmd": "ls", "out": "build src"}], command="rm -rf ./src")
    # p = P(unsafe) renormalised over the two label tokens

Policy suggestion: p < lo -> auto-run, lo <= p < hi -> ask the human, p >= hi -> block.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from shellkeeper.format import build_prompt, label_token_ids


class Guard:
    def __init__(self, path, device=None, dtype=torch.bfloat16):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(path)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype).to(self.device).eval()
        ids = label_token_ids(self.tok)
        self.pair = [ids["safe"], ids["unsafe"]]

    @torch.no_grad()
    def score_prompts(self, prompts):
        enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        pos = (enc.attention_mask.cumsum(-1) - 1).clamp(min=0)
        out = self.model(**enc, position_ids=pos, logits_to_keep=1)
        lg = out.logits[:, -1, self.pair].float()
        return torch.softmax(lg, -1)[:, 1].tolist()

    def score(self, command, task=None, history=None, cwd=None, shell=None):
        return self.score_prompts([build_prompt(task, history, command, cwd=cwd, shell=shell)])[0]


if __name__ == "__main__":
    import argparse, json, time
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("command")
    ap.add_argument("--task", default="")
    ap.add_argument("--history", default="[]", help='JSON list of {"cmd","out"}')
    ap.add_argument("--cwd")
    a = ap.parse_args()
    g = Guard(a.model)
    g.score("ls")  # warmup
    t = time.perf_counter()
    p = g.score(a.command, task=a.task, history=json.loads(a.history), cwd=a.cwd)
    print(f"P(unsafe)={p:.4f}  ({(time.perf_counter()-t)*1000:.1f} ms)")
