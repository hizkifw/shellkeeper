"""Quantization-aware fine-tuning of the guard for a specific GGUF weight format.

All transformer linear weights are fake-quantized in the forward pass (straight-through estimator), so the model
learns weights that survive llama.cpp quantization. Embeddings (tied with the output head) are left in full
precision here; llama-quantize keeps token_embd at a higher-precision type anyway.

Formats (blocks run along the input dimension, matching ggml's row layout):
  q4_0 - EXACT replica of ggml's reference Q4_0: 32-weight blocks, d = (signed absmax) / -8 stored as fp16,
         q = clamp(floor(x/d + 8.5), 0, 15) - 8. Export with llama-quantize Q4_0 *without* --imatrix to match.
  q3_k - approximation of Q3_K: 16-weight sub-blocks, symmetric, q in [-4, 3], d = absmax / 4.
  tq   - EXACT replica of ggml TQ2_0 / TQ1_0 ternary {-1,0,1} x fp16 absmax per 256 weights (imatrix-independent).
  q2_k - approximation of Q2_K: 16-weight sub-blocks, affine with non-positive min, q in [0, 3].
         (llama.cpp searches scales per sub-block; its error is <= this min/max scheme, so training against this
          is a conservative proxy.)

Loss = CE(label token) + alpha * KL(teacher 2-way || student 2-way), teacher = the full-precision starting model.

Usage: .venv/Scripts/python qat.py --init runs/v2/best --fmt q4_0 --out runs/v2-qat-q4_0 --epochs 0.5
"""
import argparse, math, random, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from shellkeeper.format import label_token_ids
from train import load, encode, batches, collate, last_logits, evaluate


def _blocks(w, bs):
    o, i = w.shape
    assert i % bs == 0, (w.shape, bs)
    return w.reshape(o, i // bs, bs)


def fq_q4_0(w):
    b = _blocks(w.float(), 32)
    idx = b.abs().argmax(-1, keepdim=True)
    mx = torch.gather(b, -1, idx)                       # signed value with largest magnitude
    d = (mx / -8).half().float()                        # stored as fp16
    inv = torch.where(d != 0, 1 / d, torch.zeros_like(d))
    q = torch.clamp(torch.floor(b * inv + 8.5), 0, 15) - 8
    return (q * d).reshape(w.shape)


def fq_q3_k(w):
    b = _blocks(w.float(), 16)
    d = b.abs().amax(-1, keepdim=True) / 4
    d = torch.where(d == 0, torch.ones_like(d), d)
    q = torch.clamp(torch.round(b / d), -4, 3)
    return (q * d).reshape(w.shape)


def fq_q2_k(w):
    b = _blocks(w.float(), 16)
    lo = torch.clamp(b.amin(-1, keepdim=True), max=0)
    hi = b.amax(-1, keepdim=True)
    s = (hi - lo) / 3
    s = torch.where(s == 0, torch.ones_like(s), s)
    q = torch.clamp(torch.round((b - lo) / s), 0, 3)
    return (q * s + lo).reshape(w.shape)


def fq_tq(w):
    """EXACT replica of ggml TQ2_0/TQ1_0 (ternary): 256-weight blocks, d = absmax (stored fp16, inverse taken
    from the fp32 value), q = lroundf(x/d) in {-1,0,1}. TQ quantization ignores --imatrix, so export is exact."""
    b = _blocks(w.float(), 256)
    d = b.abs().amax(-1, keepdim=True)
    inv = torch.where(d != 0, 1 / d, torch.zeros_like(d))  # fp32 id, fp32 product, as in ggml
    y = b * inv
    # lroundf without the fp32 trap: floor(a + 0.5) rounds 0.49999997 up to 1. Use floor(a) + floor(2*frac).
    a = y.abs(); fl = torch.floor(a)
    q = torch.clamp(torch.sign(y) * (fl + torch.floor(2 * (a - fl))), -1, 1)
    return (q.float() * d.half().float()).reshape(w.shape)


FQ = {"q4_0": fq_q4_0, "q3_k": fq_q3_k, "q2_k": fq_q2_k, "tq": fq_tq}


class FQLinear(nn.Module):
    def __init__(self, lin: nn.Linear, fq):
        super().__init__()
        self.lin, self.fq = lin, fq

    def forward(self, x):
        w = self.lin.weight
        wq = w + (self.fq(w).to(w.dtype) - w).detach()  # STE
        return F.linear(x, wq, self.lin.bias)


@torch.no_grad()
def ternary_init(model):
    """Replace each 256-block of every linear weight with its least-squares-optimal ternary approximation
    {-s, 0, +s} (threshold t chosen from a grid around 0.7*mean|w|). Since the block's absmax then equals s,
    ggml's absmax TQ quantizer reproduces this optimum exactly, so QAT starts from the best ternary point."""
    for mod in model.model.modules():
        if not isinstance(mod, nn.Linear):
            continue
        w = mod.weight.data
        b = _blocks(w.float(), 256)
        a = b.abs()
        best_err, best = None, None
        for f in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2):
            t = f * a.mean(-1, keepdim=True)
            m = (a > t).float()
            sc = (a * m).sum(-1, keepdim=True) / m.sum(-1, keepdim=True).clamp(min=1)
            q = torch.sign(b) * m * sc
            err = ((q - b) ** 2).sum(-1, keepdim=True)
            if best is None:
                best_err, best = err, q
            else:
                better = err < best_err
                best = torch.where(better, q, best); best_err = torch.minimum(err, best_err)
        mod.weight.data.copy_(best.reshape(w.shape).to(w.dtype))


def wrap(model, fq):
    n = 0
    for name, mod in list(model.model.named_modules()):
        for cn, child in list(mod.named_children()):
            if isinstance(child, nn.Linear):
                setattr(mod, cn, FQLinear(child, fq)); n += 1
    return n


def unwrap(model):
    for name, mod in list(model.model.named_modules()):
        for cn, child in list(mod.named_children()):
            if isinstance(child, FQLinear):
                setattr(mod, cn, child.lin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="runs/v2/best")
    ap.add_argument("--fmt", required=True, choices=FQ)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--alpha", type=float, default=1.0, help="weight of distillation KL")
    ap.add_argument("--tq-init", action="store_true", help="init linears at optimal ternary (for --fmt tq)")
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--grad_accum", type=int, default=2)
    a = ap.parse_args()

    torch.manual_seed(0); rng = random.Random(0); dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.init)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    lab = label_token_ids(tok); pair = [lab["safe"], lab["unsafe"]]
    teacher = AutoModelForCausalLM.from_pretrained(a.init, dtype=torch.bfloat16).to(dev).eval()
    model = AutoModelForCausalLM.from_pretrained(a.init, dtype=torch.float32, attn_implementation="sdpa").to(dev)
    if a.tq_init:
        ternary_init(model)
    print(f"fake-quantized linears: {wrap(model, FQ[a.fmt])} ({a.fmt})")
    model.gradient_checkpointing_enable(); model.config.use_cache = False

    train, val = encode(tok, load("train"), a.max_len), encode(tok, load("val"), a.max_len)
    ev0 = evaluate(model, val, lab, pad_id, dev, a.max_tokens)
    print(f"before QAT (fake-quant {a.fmt}) VAL", {k: round(v, 4) for k, v in ev0.items()}, flush=True)

    spe = math.ceil(len(batches(train, a.max_tokens, False, None)) / a.grad_accum)
    total = max(1, int(spe * a.epochs)); warm = max(1, total // 20)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0, betas=(0.9, 0.95), fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / total))))
    step, t0, run = 0, time.time(), []
    model.train()
    while step < total:
        for k, b in enumerate(batches(train, a.max_tokens, True, rng)):
            ids, mask, pos = collate(train, b, pad_id, dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                tp = torch.softmax(last_logits(teacher, ids, mask, pos)[:, pair], -1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = last_logits(model, ids, mask, pos)
            y = torch.tensor([lab[train[i][1]] for i in b], device=dev)
            kl = F.kl_div(torch.log_softmax(lg[:, pair], -1), tp, reduction="batchmean")
            loss = F.cross_entropy(lg, y) + a.alpha * kl
            (loss / a.grad_accum).backward(); run.append(loss.item())
            if (k + 1) % a.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % 20 == 0:
                print(f"step {step}/{total} loss {sum(run)/len(run):.4f} {time.time()-t0:.0f}s", flush=True); run = []
            if step >= total:
                break
    ev = evaluate(model, val, lab, pad_id, dev, a.max_tokens)
    print(f"after QAT (fake-quant {a.fmt}) VAL", {k: round(v, 4) for k, v in ev.items()}, flush=True)
    unwrap(model)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model.to(torch.bfloat16).save_pretrained(out, safe_serialization=True); tok.save_pretrained(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
