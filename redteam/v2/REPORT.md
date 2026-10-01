# Red-team round 2: v2 (`runs/v2/best`)

Round 2 used 307 fresh cases (`adversarial_set.jsonl`) in 35 categories. The cases were only scored,
never executed. GLM-5.3 (medium reasoning) independently adjudicated the labels and disagreed with 27 of
307. The adjudicated labels are in `adversarial_set.adj.jsonl`.

## v1 vs v2 on these fresh cases (adjudicated labels)

| | missed unsafe @0.5 | blocked safe @0.5 | missed unsafe @0.1 | blocked safe @0.1 | confident misses (p<0.1) |
|---|---|---|---|---|---|
| v1 | 48/165 (29%) | 18/142 (13%) | 28/165 (17%) | 34/142 (24%) | 28 |
| v2 | **36/165 (22%)** | **13/142 (9%)** | **15/165 (9%)** | 31/142 (22%) | **15** |

## Round-1 weaknesses

**Fixed in v2:** payload-in-file wrappers, config-level security weakening, indirection through
variables/files/aliases, and the git/DB false positives.

**Still open, ranked:**

1. CI/CD test-gate weakening (about 8/9 missed): `|| true`, `continue-on-error`, deleting test steps, `--no-verify`.
2. Making cloud storage public (about 7/9 missed): `--acl public-read`, bucket policies with `Principal:"*"`, Azure public access.
3. ORM migration runners whose migration body drops data (about 5/11 missed). This is the wrapper problem again, specific to DB tools.
4. PowerShell/.NET equivalents of bash families v2 already handles (about 6/16 missed): env dumps, TLS-validation callbacks, Defender.
5. netcat listen bind-shells and unrequested `ngrok` tunnels (surface-form gaps).
6. Multi-target scope creep (scores hover just under 0.5).
7. Architectural: output clipping keeps the first 5 lines and the last line, so a payload in the middle of a `cat`'d script is invisible.

**False positives:** registry auth logins (ECR `get-login-password | docker login`, npm `_authToken`),
`kubectl get secrets` (lists names only), and `git reset --hard` / `--force-with-lease` when the user asked for it.

**Robust to:** decode-and-execute, `curl | sh`, reverse shells, broad `find`/`chmod`, prompt-format spoofing,
injected instructions, unicode/whitespace tricks, delayed/background payloads, unrequested messaging,
log tampering, most supply-chain forms, privileged containers, and the sensitive-vs-benign file distinction.
Context flips work: the same command is correctly safe when the user asked for it and unsafe when they didn't.

See the agent's batch files (`b01`–`b11`) and `analyze.py` for per-case detail.
