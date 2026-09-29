# Run statement: DMA fence test and mutation check (Task 1.5, Claude handoff C)

Written before launch. Status of results: pending Codex review.

- **Question:** does `KaggricultureEnv._fence()` (`python/owl/kaggriculture/env.py`,
  a `current_stream().synchronize()` on CUDA with pinned buffers) stop `step` from
  overwriting host buffers while a non-blocking H2D copy of them is still queued
  behind a GPU sleep? And does the test detect a fence that has been removed
  from source?
- **Inputs and code path:** pod `aki4vy8kpfldpa`, `/root/kg-v3` at `994818b`
  (0 porcelain lines; the env receipt at `9ae0e05`). The `.venv` is from the env
  receipt and is not modified. Test
  `tests/kaggriculture/test_env_cuda_fence.py::test_step_does_not_overwrite_pending_dma`
  (sha256 `ca05047e…5929`) uses `make_env(pin_memory=True, transfer_device=cuda)` from
  `tests/kaggriculture/test_env.py` (`63d920f8…7091`) and the real `owl.rs` release
  binding. `env.py` sha256 is `cbd3fe16…368f`. Only GPU 0 is used (`CUDA_VISIBLE_DEVICES=0`).
- **Steps:**
  1. Run the test 3 times in `/root/kg-v3`. Expect a pass each time. The test
     includes its own in-process control, a monkeypatched no-op `_fence`, which
     must expose an overwrite.
  2. Mutation: copy the tree without `.venv`/`target` to `/root/fence-scratch`
     and save the original `env.py` bytes. Replace the body line
     `torch.cuda.current_stream(self.transfer_device).synchronize()` with `pass`.
     Run the test with `PYTHONPATH=/root/fence-scratch/python` from
     `/root/fence-scratch`, and record `owl.__file__` to prove the scratch
     module was imported. Expect a **FAIL** at the fenced assertion.
  3. Restore the saved bytes and confirm that the restored sha256 equals
     `/root/kg-v3`'s `env.py` and `cbd3fe16…368f`. Rerun the test on the
     restored scratch copy and expect a pass.
- **Expected discriminating observation:** step 1 passes and step 2 fails with
  a buffer name in the assertion. If step 2 passes, the fence is not what
  protects the buffers, or the delay is too short. That makes the test
  non-discriminating on this pod, and it is reported as such.
- **Stopping condition:** steps 1 to 3 complete, or on the first unexpected
  outcome (pass under mutation, fail on the original, or a CUDA error). The
  wall cap is 10 min, enforced on the pod by `timeout 600` for each pytest call.
- **Budget:** 10 min or less on the running pod ($4.18/h, so $0.70 or less). No new resource.
- **Safety:** before launch, both GPUs are idle with no compute apps. `/root/kg-v3`
  is not edited, and the scratch copy is deleted after the hash check. No
  installs, driver changes or security changes. No W&B run: this is a unit test,
  not a training run.
