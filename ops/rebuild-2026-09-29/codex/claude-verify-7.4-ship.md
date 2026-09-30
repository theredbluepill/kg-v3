Reviewer: independent Claude subagent (substitute for Codex during its usage limit; owner-approved). Not a Codex verdict.

# Task 7.4 ship: adversarial verification of the BC Kaggle package

- Branch: `kg/rebuild-7-4-ship`, HEAD `619349f`, worktree `/Users/poonszesen/kg-v3-ship` (clean before and after the review).
- Brief: `ops/rebuild-2026-09-29/briefs/7.4-packaging.md` (v2).
- Archive: `artifacts/7.4/submission.tar.gz`.
- Nothing was uploaded or submitted. No pod, GPU or training process was touched. The only local runs were fast pytest subsets and `cargo test --lib seat_tests`, from the existing cache.

## What I re-checked myself (this version)

| Check | Result |
|---|---|
| Archive SHA-256 / bytes | `00e67809…3839`, 24,704,468 bytes. Matches `validation.json` and `custody.json`. |
| Manifest SHA-256 | `4872a2d3…4a94`. Matches. It is the same manifest named by the pod Kaggle-mode receipts and the Kaggle-image receipt `kaggle-image-episode-self-seed7.json`. |
| Manifest vs contents | Fresh `tar -xzf` into scratch: 65 files listed, 65 on disk, 0 hash mismatches, 0 unlisted. |
| Archive code vs source | Every `owl/**` file and `main.py` in the archive is byte-identical to `git show HEAD:python/...`. `git diff` from `f66acf8` (native build commit) to `6c49863` (archive commit) to `HEAD` is empty for `python/ src/ engine_rs/ Cargo.* pyproject.toml uv.lock`. |
| BC weights | `/Users/poonszesen/kg-v3-runs/bc-best/checkpoint_bc_best.pt` hashes to `fd8545872aca…6f51`, as claimed. The packaged `checkpoint.pt` holds only `{"model"}`: 210 tensors, all fp32, 6,252,223 params, every tensor `torch.equal` to the original's `model`. No key matches hidden, recurrent, memory, opponent, seat or identity. The config is byte-identical to the run's `config.yaml` (`36cf8cfb…`). |
| Secrets | No credentials, netrc, W&B keys or API-key patterns in any file. The checkpoint's `wandb_run_id` was stripped by the slimming. |
| Native module | ELF64 LSB x86-64 `.so`, `PyInit_rs`, abi3-py311 (maturin log). NEEDED is only `libgcc_s.so.1`, `libm`, `libc` and `ld-linux`. The highest symbol is GLIBC_2.35, equal to the Kaggle image's glibc 2.35. There are no `zmm`/AVX-512 instructions. The ymm/AVX2 code sits only in runtime-dispatched `memchr` and `chacha20` paths. RUSTFLAGS was unset (`pod-native-build.sh`), so there is no `target-cpu=native`. |
| Kaggle image | The receipt shows the unmodified image (locally built from Kaggle's Dockerfile on `python:v163`; Python 3.11.13, torch 2.6.0+cu124, glibc 2.35) loading `owl` and `owl.rs` from `/kaggle_simulations/agent/`. It ran 719 calls per seat in strict mode with 0 faults. |
| Pod Kaggle-mode receipts | Seats 0 and 1 against starter, plus the CLI self-play, are internally consistent with `validation.json`. Each shows `ok: true`, 719 calls, 0 bad statuses, 0 exceptions, 0 invalid raw actions, `caught_errors 0`, `budget_passes 0`, `strict False`, and a venv without owl, maturin, pytest or wandb. |
| Rust seat binding | `cargo test --lib seat_tests`: 5/5 pass, including the bytewise seat-row equals `write_env` row test over a played game. |
| Python tests | The 4 ship test files: 48/48 pass. |

## Brief checklist

1. **Entry-point signature.** Pass.
   - `main.py` defines `agent(obs, config)` last. `AGENT` is a non-callable instance, so Kaggle's `get_last_callable` picks `agent`.
   - `test_packaged_main_loads_through_kaggles_loader_and_plays` uses the real loader.
   - Every bundled `owl` import is at the top level (F2).
2. **Observation handling.** Pass.
   - The view reads only `player`, the public keys and the own `private`. The configuration is allowlisted, so `__raw_path__` is dropped.
   - `remainingOverageTime` is read only by the budget guard, never by the model. The seat only selects the row.
   - Buffers are fully overwritten each call (the poison test).
   - Counters and the dropped-key log never feed the model.
3. **Action JSON legal and complete.** Pass.
   - Native decode is followed by a structural check and a native encode→decode canonical round trip in the same actor, order and hire context.
   - The pod harness also re-validated every raw return independently, taking actors from the observation: 0 invalid in 3 × 719 calls.
4. **Fallback legal and reachable.** Pass.
   - `pass_action()` equals the `kaggriculture.json` schema default byte-for-byte (checked against the installed 1.32.7).
   - It is reached on any `Exception` in `act`, including a malformed decoded action (non-strict), and when the bank is below 2.0 s.
   - Import and warm-up failures propagate loudly, as the brief intends.
5. **Time budget.** Pass, with limits.
   - Pod: turn 0 is 1.32 s, steady p99 0.125 s, max 0.26 s.
   - Kaggle image under amd64 emulation (`--cpus=1.6`): p99 0.42 s, max 0.58 s.
   - Both are inside the brief's targets (turn 0 ≤ 20 s, p99 ≤ 500 ms, max < 900 ms). No Kaggle-hardware timing exists.
6. **Archive contents.** Pass, with an edit (E1).
   - The weights SHA chain is intact. There are no secrets and no Mach-O or `.dylib` files, and the only `.so` is the Linux module.
   - `manifest.json` embeds an absolute Mac path (E1).
   - The archive also carries unused Orbit/training modules (`owl/train/*`, `owl/agent/*`, `recurrent_transformer_v1.py` …, about 0.6 MB). This is harmless, but it is not "exactly what is needed".
7. **Native `.so` matches the runtime.** Pass (see the table).

## Mutation testing (scratch worktree `git worktree add --detach`, removed after)

All 17 mutations were run against the four ship test files (48 tests), plus one Rust mutation.

| # | Mutation | Result |
|---|---|---|
| M1a | fallback always re-raises | KILLED (error-boundary, validator tests) |
| M1b | fallback returns `None` | KILLED |
| M1c | fallback `farmer` becomes a string (schema-invalid) | KILLED (`test_validator_accepts_canonical_programs`) |
| M1d | fallback returns `["NORTH"]` instead of PASS | **SURVIVED** |
| M1e | low-bank PASS guard removed | KILLED |
| M1f | `main.py` ignores `KAGGRICULTURE_AGENT_STRICT` | **SURVIVED** |
| M2a | view flips `seat = 1 - player` | KILLED (4 view tests) |
| M2b | agent encodes `1 - seat` | **SURVIVED** |
| M2c | agent always encodes seat 0 | **SURVIVED** |
| M2d | actor count read from the rival half of `actor_mask` | **SURVIVED** |
| M2e (Rust) | `from_seat_view` swaps own/rival privates | KILLED (`seat_row_equals_two_seat_row_bytewise_over_a_played_game`) |
| M3a | builder ignores `--expected-checkpoint-sha256` | KILLED |
| M3b | manifest records a corrupted original SHA | KILLED |
| M3c | manifest per-file SHA corrupted | KILLED |
| M3d | full (unslimmed) checkpoint shipped | KILLED |
| M4 | a callable defined after `agent` in `main.py` | KILLED (real-loader test) |
| M5 | native-receipt SHA binding skipped | KILLED |

**Why the M2b, M2c and M2d seat mutations survive.** `_observations()` in `test_kaggle_agent.py:44-53` feeds the agent only `env.state[0]`, which is seat 0. Both seats also take the same `HIRE` action, so hand counts never differ. The shipped code is correct:
- the pod seat-1 episode won 33,352 to 3,596;
- the harness's independent validator passed every call;
- the view tests pin the seat for the view layer.

But the suite would not catch a seat regression in `KaggricultureAgent.policy_action`.

I wrote a throwaway probe to confirm the gap can be closed. It drives Kaggle calls with seat 1 hiring (asymmetric hands), calls `agent.act` for both seats, and compares `agent.arrays` bytewise with `encode_seat_into(seat_view_json(obs))`. It passed on the real code and **killed M2a–M2d**. The probe was deleted and nothing was committed.

**M1d and M1f.** The fallback is only compared against `pass_action()` itself and never pinned to the schema default. The strict env-var wiring in `main.py` is untested. The pod receipts cover both empirically: a strict run showed 0 faults, and `strict: False` was read back from the agent.

**Corrupted packaged weights.** No mutation needed here. Nothing verifies the manifest at load time, so a bit-flipped tensor value in `models/primary/checkpoint.pt` would load and play silently. Only the offline manifest re-hash (done on the pod and here) detects it.

## Findings

**E1 (required edit, low severity; custody and privacy).** `scripts/build_kaggriculture_submission.py:231` writes `"original_path": str(checkpoint)`, which is the resolved absolute path. The shipped `manifest.json` therefore contains `/Users/poonszesen/kg-v3-runs/bc-best/checkpoint_bc_best.pt`, which fails the brief check "no absolute Mac paths". It also leaks the local username to Kaggle.
- The manifest also carries pod paths (`/root/ship`, `/root/ship/venv`) in the embedded native receipt, and the `.so` carries `/root/.cargo/registry/...` panic-location strings. These are not Mac paths. Optionally scrub them with `--remap-path-prefix`.
- Fix: record `checkpoint.name` (the SHA already identifies it), then rebuild. Only `manifest.json` changes; no runtime file reads it, so the validated behaviour carries over.
- The archive and manifest SHAs change, so `custody.json` and `validation.json` need the new hashes, and the new archive needs a re-hash plus a short load check.
- The owner may instead explicitly waive this, since the path is not a secret.

**E2 (recommended, tests).** Add an agent-level test that plays seat 1 with asymmetric hand counts and compares `agent.arrays` with the view encoding (the probe above). It closes M2b, M2c and M2d. Also pin `pass_action()` to the literal schema default loaded from the installed `kaggriculture.json` (M1d). Also add a subprocess test that `KAGGRICULTURE_AGENT_STRICT=1` makes the loaded `AGENT.strict` true (M1f).

**E3 (optional hardening).** Verify the `models/primary/checkpoint.pt` SHA against `manifest.json` at load in `main.py`. It costs about 50 ms on turn 0 and would turn silent weight corruption into a loud import failure.

**Residual gaps against the brief.** None of these block the package working. The owner should see them before deciding to submit.
- The tarball was not built inside Kaggle's image. The pod has no Docker; the module was built with maturin `--compatibility linux` on glibc 2.39 and checked to require no more than GLIBC_2.35. The image load check ran on the owner's Mac under amd64 emulation.
  - The receipt says it was meant as a 4-step check and ran a full 719-turn episode through a harness bug. That conflicts with the "no heavy local runs" rule and is recorded honestly.
- T7 action parity is not implemented as a test: packaged path vs training path, deterministic tokens, and the post-call RNG-state equality from S2. Observation-byte parity is implemented and passes.
- No W&B telemetry for these validation episodes (brief §10 W1). The gap is visibly recorded in `validation.json` `not_done`.
- The pod validation used torch `2.6.0+cpu`, not the image's `+cu124`. The emulated image run covers `+cu124`.
- There is no Kaggle-hardware timing. Production observation key order (F9, Q6) remains unverified.
- `rs-prepare` for this branch is not recorded in `ops/`. Only `py-prepare.log` is (2777 passed). I ran only the `seat_tests` subset.
- BC strength evidence is one game per seat against starter, plus one self-play game. That is a packaging check, not a strength claim, as the receipts say.

## Conclusion

The package does what it claims:
- the right BC weights, bit-exact;
- a Linux x86-64 abi3 module compatible with glibc 2.35;
- a stateless, current-observation agent;
- a legal, reachable fallback;
- clean 719-call episodes in both seats and in self-play, well inside the time and memory budgets.

The mutations that matter for the ship are killed: fallback breakage, the seat index at the view and Rust layers, and weight or manifest SHA corruption. The remaining issues are the absolute Mac path in the shipped manifest (E1) and a seat-coverage hole at the agent-test layer (E2). Neither changes runtime behaviour, but E1 fails an explicit check in the brief.

VERDICT: APPROVE WITH EDITS
