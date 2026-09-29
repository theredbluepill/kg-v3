# GPU checks bundle — evidence (2026-09-29, pod `w7ia3zvxqsvs3g`, GPUs 0 and 1)

Run statement: `../run-statements/gpu-checks-bundle.md`. It was committed with the scripts in `5f2ee2d` (09:14:26Z), before any launch. Amendment 1 was committed in `a4c75e0` (09:20:45Z), after attempt 1 stopped and before attempt 2. The results section is "GPU checks bundle (component)" in `../results.md`.

## Identity

- **Source:** `kg/rebuild-trainer-model` @ `8fde43cd7408c9c4f9147eeb8f916bd08f8266ed`, tree `70d50fa37e66aa322e8db00c9359d3f9e3031499`.
- **Pod checkout:** `/workspace/kg-v3-rebuild` went from `e1458d2` (tree `16b79ba5…`, porcelain empty) to detached `8fde43c` via a git bundle (sha256 `ecb953dc…9e4e`; `git bundle verify` ok) plus fetch and checkout at 09:15:04Z. Receipt: `pod/attempt1/receipts/git_update.txt`.
  - `origin` is unchanged (the earlier bundle path), and nothing was pushed.
  - Porcelain was empty after the update, at both launches (`git_pre.txt`), after both runs (`git_post.txt`) and at 09:29:44Z (`post-run/idle_after_copy.txt`).
- **Rust extension:** not rebuilt. No `*.rs`, `Cargo.*`, `build.rs`, `pyproject.toml` or `uv.lock` change between `e1458d2` and `8fde43c`. `python/owl/rs.abi3.so` sha256 was `35239d1b…93b8` before and after.
- **Stack** (`receipts/versions.txt`): torch 2.9.0+cu128 (git `0fabc3ba…`), triton 3.5.0, flash-attn 2.8.3, CUDA runtime 12.8, driver 595.91.07, 2× RTX PRO 6000 Blackwell Server Edition (97,887 MiB each). No `TORCHINDUCTOR_*`/`TRITON_*` env vars were set. `which nsys` exits 1.
- **Backend records:** every `*.backend.json` has `value_at_start == value_at_end ==` the requested setting (`ATEN` or `ATEN,TRITON,CPP`). Every process record shows HEAD `8fde43c`, an empty porcelain, `kg_file` under `/workspace/kg-v3-rebuild/python`, and its `CUDA_VISIBLE_DEVICES`.

## Attempts

| attempt | driver wall | exit | what ran |
|---|---|---|---|
| 1 (`pod/attempt1/`) | 09:15:32–09:16:28Z, 55.4 s | 3 | `c3_mid_aten` pass. `c2_aten_bwd` **FAIL_stop**, only on `blocks.0.attn.k.bias`: gradient rel_max 1.155 / 0.807 / 0.858 at the warm point / 4,194,305 / 4,198,400. The running `c3_dense_aten` was terminated (rc −15) and nothing else started. |
| 2 (`pod/attempt2/`) | 09:21:13–09:29:13Z, 480.0 s | 0 | Every stage passed. The control reproduced its expected failure. |

The aggregate driver wall was 535 s, against the 60-min limit. Attempt 1 stopped on a degenerate criterion. The key-bias gradient is analytically zero (softmax shift invariance): fp64 CPU check 3.9e-14 vs 30.8 for the query bias, in `post-run/kbias_analytic_zero_cpu_check.txt`. Amendment 1 changed only that parameter's criterion, and attempt 2 reran the whole bundle. Every compiled-vs-eager number in c2 is identical in the two attempts; only the eager-vs-eager floor differs (flash backward is nondeterministic).

## Placement and timeline, attempt 2 (`pod/attempt2/driver.jsonl`)

- **GPU 1:** `c2_aten_bwd` 09:21:13–09:22:10 pass. Then `c2_ctl_default_bwd` until 09:22:47, control_reproduced: rc 1, illegal memory access during default-backend autotuning at 4,194,305. Then `c1_aten` until 09:23:19 pass, and `c1_default` until 09:23:51 pass.
- **GPU 0:** `c3_mid_aten` until 09:22:11, `c3_dense_aten` until 09:23:07, `c3_mid_default` until 09:24:16 and `c3_dense_default` until 09:25:27, all pass.
- **Phase 2, GPU 0 alone, GPU 1 idle:** `c4_mid_aten` 09:25:27–09:27:10 pass, then `c4_dense_aten` until 09:29:13 pass.
- **Idle:** GPUs 0 and 1 showed 0 MiB, 0 % and no compute processes at pre and prelaunch for both attempts, and at post and 09:29:44Z (`post-run/idle_after_copy.txt`). The pod was left running and idle.

## Files

- `scripts/`: the committed scripts (attempt 2 versions). `pod/attempt{1,2}/` hold the scripts as run (sha256 in `receipts/scripts.sha256`), the logs, JSONL records, backend records, `template_census.*` and receipts.
- `pod/attempt2/c1_*.outliers.json`: every outlier coordinate (row, token, channel) per density and path.
- `summarize.py` → `summary.json`: every number quoted in `results.md`, computed from the retained records.
- `post-run/`: the key-bias CPU check and the post-copy idle/git capture.
- Not copied: the Inductor/Triton caches (the pod run dir is 977 MB) and attempt 2's `receipts/compile_caches.sha256`, which is 4,123,148 B and over the 1 MB rule (sha256 `52b8e71ad24197ccb3d0b8b4fc55b2dd7efcfd8d6443c3a4cb2c30e7a5323eb9`, 22,125 lines). Both remain on the pod under `/workspace/kg-v3-rebuild/runs/gpu-checks-2026-09-29/`. Attempt 1's cache hash list (865 kB) is included.
- `MANIFEST.sha256`: sha256 of every file here.
