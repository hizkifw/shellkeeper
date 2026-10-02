# GGUF quantization results (llama.cpp b11322)

- All scores come from `llama-server`: one token is generated, and P(unsafe) is the softmax over the logprobs
  of ` safe` and ` unsafe`. See `bench/gguf_eval.py`.
- The BF16 GGUF matches HF transformers: mean |Δp| 0.001, 4/2700 flips, all at the threshold.
- The imatrix was computed on 1,500 *training* prompts, each with its label appended.
- n = 3,007 across five eval sets. Flips = decisions that differ from BF16 at threshold 0.5.

| file | MB | test AUC | test_new AUC | red-team r1 AUC | red-team r2 AUC | flips | CPU p50 | Vulkan p50 |
|---|---|---|---|---|---|---|---|---|
| BF16 | 1198 | 0.990 | 0.992 | 0.954 | 0.938 | 0 | 154 ms | 26 ms |
| Q8_0 | 639 | 0.990 | 0.992 | 0.954 | 0.938 | 6 | 193 ms | 22 ms |
| **Q4_K_M + imatrix** | **397** | 0.989 | 0.992 | 0.951 | 0.937 | 21 | **106 ms** | 22 ms |
| Q3_K_M + imatrix | 347 | 0.989 | 0.991 | 0.948 | 0.937 | 30 | – | – |
| **Q3_K_M + imatrix, emb Q3_K** | **286** | 0.989 | 0.991 | 0.950 | 0.938 | 56 | 115 ms | 22 ms |
| **Q2_K + imatrix, emb Q3_K** | **235** | 0.986 | 0.988 | 0.942 | 0.928 | 118 | 128 ms | 22 ms |
| Q2_K + imatrix | 296 | 0.986 | 0.988 | 0.942 | 0.927 | 101 | – | – |
| IQ2_M + imatrix | 265 | 0.985 | 0.989 | 0.933 | 0.916 | 111 | – | – |
| IQ2_XXS + imatrix | 229 | 0.966 | 0.941 | 0.864 | 0.868 | 312 | – | – |
| IQ2_XXS + imatrix, Q2_K-QAT | 229 | 0.976 | 0.966 | 0.882 | 0.893 | 282 | – | – |
| IQ1_M + imatrix | 216 | 0.906 | 0.862 | 0.695 | 0.800 | 639 | – | – |
| IQ1_M + imatrix, Q2_K-QAT | 216 | 0.947 | 0.914 | 0.783 | 0.839 | 511 | – | – |
| TQ2_0 (ternary), PTQ | 247 | 0.499 | 0.499 | 0.500 | 0.500 | 1659 | – | – |
| TQ1_0 (ternary), exact QAT | 187 | 0.891 | 0.908 | 0.685 | 0.718 | 604 | – | – |

CPU latency is for an i5-13600K with 6 threads, measured while GPU training was running. Vulkan latency is for an
RX 7900 XTX. Latency is for a single request on short (golden) prompts.

## Findings

- **Down to about 3 bits, quantization is lossless.**
  - Q8_0 through Q3_K_M stay within noise of BF16.
  - The tied 152k × 1024 embedding is 37–44% of the file at ≤4 bit, and can drop from Q6_K to Q3_K for free.
  - Q2_K is the knee: about −0.004 test AUC and −0.012 red-team AUC.
- **The imatrix matters at 2 bits.** It halves the decision flips at Q2_K (221 → 101).
- **QAT:**
  - *Q2_K-proxy QAT* (min/max sub-block simulation) gives no gain at Q2_K + imatrix, because llama.cpp's scale
    search and mixed tensor types don't match the proxy. It does help the very-low-bit IQ formats it wasn't
    trained for: IQ2_XXS 0.966 → 0.976, IQ1_M 0.906 → 0.947.
  - *Exact ternary QAT* (TQ2_0/TQ1_0) is bit-exact with llama-quantize (0 mismatches over 440M weights).
    Post-training ternary is random (AUC 0.50). QAT from an optimal-ternary init brings it to 0.89 in one epoch,
    still well below the 2–3-bit quants. A 0.6B model needs far more than about 10M tokens to recover from full
    ternarization.
- **GPU latency is about 22 ms whatever the quant** (it's overhead-bound). On CPU, Q4_K_M is the fastest.

## Recommended files

- **Q4_K_M + imatrix (397 MB):** lossless, fastest on CPU. The default.
- **Q3_K_M + imatrix, emb Q3_K (286 MB):** the smallest lossless option.
- **Q2_K + imatrix, emb Q3_K (235 MB):** the smallest file that is still good.
- **Q8_0 (639 MB):** maximum fidelity.

**Next lever:** prune the vocabulary. The model needs only the tokens that appear in shell/English text and outputs
just two of them, so trimming the 152k vocab would cut the embedding (the floor below about 230 MB) by most of
its size.
