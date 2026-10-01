"""Full fine-tune of a small causal LM to emit " safe" / " unsafe" after the prompt.

Loss: cross-entropy over the full vocab at the label position only (so greedy decoding
also yields the label), computed from the last hidden state only -> no seq x vocab logits.
Left padding keeps the label position at index -1 for every row.

Usage: .venv/Scripts/python train.py --model Qwen/Qwen3-0.6B-Base --out runs/v1
"""
import argparse, json, math, random, time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from shellkeeper.format import label_token_ids, build_prompt
from eval.golden import GOLDEN

D = Path(__file__).resolve().parent / "data"


def load(name):
    return [json.loads(l) for l in open(D / f"{name}.jsonl", encoding="utf-8") if l.strip()]


def encode(tok, rows, max_len):
    out = []
    for r in rows:
        ids = tok.encode(r["prompt"], add_special_tokens=False)
        if len(ids) > max_len:  # keep the head (task) and tail (command); drop the middle of the session
            ids = ids[: max_len // 4] + ids[-(max_len - max_len // 4):]
        out.append((ids, r["label"]))
    return out


def batches(data, max_tokens, shuffle, rng):
    """Length-bucketed batches under a padded-token budget."""
    idx = sorted(range(len(data)), key=lambda i: len(data[i][0]))
    bs, cur, cur_max = [], [], 0
    for i in idx:
        L = len(data[i][0])
        if cur and max(cur_max, L) * (len(cur) + 1) > max_tokens:
            bs.append(cur); cur, cur_max = [], 0
        cur.append(i); cur_max = max(cur_max, L)
    if cur:
        bs.append(cur)
    if shuffle:
        rng.shuffle(bs)
    return bs


def collate(data, b, pad_id, device):
    L = max(len(data[i][0]) for i in b)
    ids = torch.full((len(b), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(b), L), dtype=torch.long)
    for j, i in enumerate(b):
        s = data[i][0]
        ids[j, L - len(s):] = torch.tensor(s)
        mask[j, L - len(s):] = 1
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    return ids.to(device), mask.to(device), pos.to(device)


def last_logits(model, ids, mask, pos):
    h = model.model(input_ids=ids, attention_mask=mask, position_ids=pos).last_hidden_state[:, -1]
    return model.lm_head(h).float()


@torch.no_grad()
def evaluate(model, data, lab_ids, pad_id, device, max_tokens):
    model.eval()
    ps, ys, loss = [], [], 0.0
    s, u = lab_ids["safe"], lab_ids["unsafe"]
    for b in batches(data, max_tokens, False, None):
        ids, mask, pos = collate(data, b, pad_id, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = last_logits(model, ids, mask, pos)
        y = torch.tensor([lab_ids[data[i][1]] for i in b], device=device)
        loss += F.cross_entropy(lg, y, reduction="sum").item()
        ps += torch.softmax(lg[:, [s, u]], -1)[:, 1].tolist()
        ys += [int(data[i][1] == "unsafe") for i in b]
    model.train()
    return metrics(ps, ys) | {"loss": loss / max(1, len(data))}


def metrics(ps, ys, thr=0.5):
    tp = sum(p >= thr and y for p, y in zip(ps, ys)); fp = sum(p >= thr and not y for p, y in zip(ps, ys))
    fn = sum(p < thr and y for p, y in zip(ps, ys)); tn = len(ys) - tp - fp - fn
    # AUROC via rank statistic
    pos = [p for p, y in zip(ps, ys) if y]; neg = [p for p, y in zip(ps, ys) if not y]
    auc = None
    if pos and neg:
        allp = sorted([(p, 1) for p in pos] + [(p, 0) for p in neg])
        rank_sum, r = 0.0, 1
        i = 0
        while i < len(allp):
            j = i
            while j < len(allp) and allp[j][0] == allp[i][0]:
                j += 1
            avg = (r + r + (j - i) - 1) / 2
            rank_sum += avg * sum(1 for k in range(i, j) if allp[k][1])
            r += j - i; i = j
        auc = (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return {"acc": (tp + tn) / max(1, len(ys)), "recall_unsafe": tp / max(1, tp + fn),
            "fpr": fp / max(1, fp + tn), "auc": auc, "n": len(ys)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--out", default="runs/v1")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--max_tokens", type=int, default=8192, help="padded tokens per micro-batch")
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); rng = random.Random(args.seed)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    lab_ids = label_token_ids(tok)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, attn_implementation="sdpa").to(dev)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    train, val = encode(tok, load("train"), args.max_len), encode(tok, load("val"), args.max_len)
    golden = encode(tok, [{"prompt": build_prompt(g["task"], g["history"], g["cmd"], cwd=g["cwd"], shell=g["shell"]),
                           "label": g["label"]} for g in GOLDEN], args.max_len)
    n_unsafe = sum(l == "unsafe" for _, l in train)
    print(f"train={len(train)} (unsafe {n_unsafe}) val={len(val)} golden={len(golden)} "
          f"max_len_seen={max(len(x) for x, _ in train)}")

    steps_per_epoch = math.ceil(len(batches(train, args.max_tokens, False, None)) / args.grad_accum)
    total = int(steps_per_epoch * args.epochs)
    warm = max(1, int(0.05 * total))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95), fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / total))))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    best, step, t0, run_loss = -1, 0, time.time(), []
    model.train()
    ep = 0
    while step < total:
        bl = batches(train, args.max_tokens, True, rng)
        for k, b in enumerate(bl):
            ids, mask, pos = collate(train, b, pad_id, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = last_logits(model, ids, mask, pos)
            y = torch.tensor([lab_ids[train[i][1]] for i in b], device=dev)
            loss = F.cross_entropy(lg, y)
            (loss / args.grad_accum).backward()
            run_loss.append(loss.item())
            if (k + 1) % args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 20 == 0:
                print(f"ep {ep} step {step}/{total} loss {sum(run_loss)/len(run_loss):.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s", flush=True)
                run_loss = []
            if step % max(1, steps_per_epoch // 2) == 0 or step == total:
                vm = evaluate(model, val, lab_ids, pad_id, dev, args.max_tokens)
                gm = evaluate(model, golden, lab_ids, pad_id, dev, args.max_tokens)
                print(f"  VAL {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in vm.items()})}")
                print(f"  GOLDEN {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in gm.items()})}", flush=True)
                score = vm["auc"] or 0
                if score > best:
                    best = score
                    model.save_pretrained(out / "best", safe_serialization=True)
                    tok.save_pretrained(out / "best")
                    json.dump({"step": step, "val": vm, "golden": gm, "args": vars(args)},
                              open(out / "best" / "metrics.json", "w"), indent=1)
            if step >= total:
                break
        ep += 1
    print(f"done. best val auc {best:.4f} -> {out/'best'}")


if __name__ == "__main__":
    main()
