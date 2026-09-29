# DMA fence test and mutation check (Task 1.5, Claude handoff C): receipt

Status: **pending Codex review** (Codex is at its usage limit until Oct 6). Not
independently verified. The run statement is `run-statement.md`, committed in
`648ff82` before launch.

## Identity

- Pod `aki4vy8kpfldpa`, 2x RTX PRO 6000 Blackwell, driver 595.91.07, with only GPU 0
  used (`CUDA_VISIBLE_DEVICES=0`). `/root/kg-v3` is at `994818b`, and it had 0
  porcelain lines before and after (`pod/git_state*.txt`).
- torch 2.9.0+cu128, pytest 9.0.3, Python 3.12.3, and the `.venv` from the env receipt (`9ae0e05`).
- Hashes (`pod/hashes_original.sha256`): `env.py` `cbd3fe16…368f`,
  `test_env_cuda_fence.py` `ca05047e…5929`, `test_env.py` `63d920f8…7091`, and
  `owl/rs.abi3.so` `5ad74acd…de69` (the env receipt's release build).
- Both GPUs showed 0 MiB with no compute apps before and after (`pod/idle_*.csv`).
- Wall time: the driver ran 16:03:37Z to 16:03:50Z and the clean repeat ran
  16:05:05Z to 16:05:11Z, about 20 s of GPU work in total. The fence work's
  share of pod time is under $0.10 at $4.18/h.

## Results

| Step | Code | Outcome | Log |
|---|---|---|---|
| 1 | original `/root/kg-v3`, 3 runs | **3 passed** (0.61 s, 0.39 s, 0.43 s) | `pod/step1_original_run{1,2,3}.log` |
| 2 | scratch copy, fence body replaced by `pass` | **1 failed**: `AssertionError: market_float` at the fenced assertion (`test_env_cuda_fence.py:34`) | `pod/step2_mutated.log` |
| 3 | scratch copy restored | sha256 `cbd3fe16…368f` on both files, `cmp` identical; **1 passed** | `pod/step3_restored.{sha256,log}` |
| 2b | clean repeat of 2, with no copied bytecode | **1 failed**: `AssertionError: market_float` at `tests/kaggriculture/test_env_cuda_fence.py:34` (scratch rootdir) | `pod/step2b_mutated.log` |
| 3b | clean repeat of 3 | restored sha256 equal, `cmp` identical; **1 passed** | `pod/step3b_restored.{sha256,log}` |

- The mutation is a single-line diff (`pod/step2_mutation.diff`, line 336):
  `torch.cuda.current_stream(self.transfer_device).synchronize()` is replaced by
  `pass  # MUTATION: fence removed`. The pattern matched exactly once.
- The step 1 pass also covers the test's own in-process control, which
  monkeypatches `_fence` to a no-op. It asserts that an overwrite is then
  visible, and that control passed.
- Under mutation, the pinned `market_float` host buffer changed after `step`
  while its non-blocking H2D copy was still queued behind `torch.cuda._sleep`.
  The device copy then held the post-step prices (for example 0.1000 became
  0.1040). The assertion stops at the first buffer that differs, so later
  buffers were not checked in that run.

### Why step 2 was repeated (2b/3b)

The step 2 traceback printed the test path as `../kg-v3/tests/...` although
rootdir was `/root/fence-scratch`. The first copy was taken with `tar`, which
kept the `__pycache__` files and their mtimes. So the test module's cached
bytecode was valid for the scratch copy, and it still carried the original
`co_filename`. The source bytes were identical (the same test sha256). The
mutated `env.py` changed size and mtime, so it was recompiled, and
`pod/step2_import_paths.txt` shows `owl`, `owl.kaggriculture.env` and
`owl.rs` importing from `/root/fence-scratch/python`. To remove the
ambiguity, 2b/3b were repeated with `__pycache__` excluded and
`PYTHONDONTWRITEBYTECODE=1` (`pod/dma_fence_2b.sh`, `pod/driver_2b.log`: 0
`__pycache__` dirs before and after). The traceback then named the scratch
file, and the outcomes were the same. The scratch copy and the saved
original were deleted afterwards.

## Conclusion (pending review)

On this pod the test is discriminating. It passes with the fence and fails
when the fence is removed from source. The copy was restored byte for byte
and confirmed by sha256 and `cmp`. This is the L6 evidence for plan 1.5 D/F.

## Limits

- The test runs one env configuration (`make_env` defaults) with a single
  200M-cycle `_sleep`. The trainer's own copy schedule is not exercised, and
  only one GPU was used.
- Step 2b was a repeat added after the first mutation run. It was not in the
  pre-launch statement.
