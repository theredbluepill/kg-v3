# 8-GPU sprint kit: Mac-side runbook

**Status.** This is preparation only. No pod was created, started or touched, and nothing here has run on a GPU. The owner is still looking for the pod ("I still looking for 8pods for last sprint, pls wait for my news"). The owner has accepted that 8 GPUs change the per-update batch ("is ok for 1:1"), so this run is not 1:1 with the 4-GPU run. Do not touch the running 4-GPU pod `abl4mvr5w1mmn4` or its run `pcy5knet`.

## What runs

The recipe is the one the live run `pcy5knet` uses (W&B `spoon/kg-v3`, code `0f70773`, which is in integration main `07c8fc99`). Only the world size changes:

- `configs/kaggriculture_4rank_margin.yaml` with `-o env.n_envs=12 rl.horizon=720 rl.segments_per_minibatch=1 rl.gae_lambda=1.0`.
- Reward: `econ_bank` 0.25 / 150000 / 0.25 and `econ_margin` 0.25 / 100000 / 0.25. `econ_shaping` is 0, from the preset.
- From the preset: Muon 1e-4 / AdamW 5e-6, `native_threads` 4, and a checkpoint plus last_best evaluation every 10M env steps.
- Start with `--load-model-weights <promoted last_best> --load-model-weights-mode model_only`. This gives a fresh optimizer, so the 1,000-step warm-up runs again.
- W&B is online and the wrapper refuses to run otherwise.

Per iteration, 4 GPUs (pcy5knet) against 8 GPUs (sprint):

| | 4 GPUs (pcy5knet) | 8 GPUs (sprint) |
|---|---|---|
| Games per iteration (`train/bank_games`) | 48 | **96** |
| Env steps per iteration | 34,560 | 69,120 |
| Optimizer steps per iteration | 12 | 12 |
| Rows per optimizer step | 4 × 1,440 | **8 × 1,440** |
| Iterations per 10M checkpoint/eval | ~289 | ~145 |
| Optimizer steps per 10M env steps | ~3,470 | ~1,740 |

The 8-rank run reaches each checkpoint after half as many optimizer steps. Each step averages twice the data at the same learning rate. The LR schedule counts optimizer steps, so per env step it advances half as fast.

**Reconciling `configs/kaggriculture_8rank.yaml` (on main).** That preset is the Isaiah-division base recipe for 8 ranks, not the current recipe:

| | `kaggriculture_8rank.yaml` | sprint |
|---|---|---|
| n_envs / horizon | 32 / 64 | 12 / 720 |
| segments_per_minibatch | 2 | 1 |
| gae_lambda | 0.9 | 1.0 |
| Muon / AdamW LR | 2e-3 / 1e-4 | 1e-4 / 5e-6 |
| econ_shaping | 0.2 | 0 |
| econ_bank, econ_margin | off | on (as above) |
| native_threads | 2 | 4 |

The sprint does **not** use it. `run_ppo` reads the world size from torchrun, and the 4-rank preset holds nothing that depends on 4 ranks; the name "4rank" only records where it came from. `launch.sh` pins the preset and the overrides. Anything added with `--extra` changes the recipe and needs the owner's yes.

**Code identity.** Main `07c8fc99` adds `rl.initial_stagger` and `model.critic_offset` on top of `0f70773`. Both are off by default, and neither the preset nor the overrides turn them on. The launch receipt records the commit it ran and the file hashes.

## Kit files (copied to `/root/sprint-kit` on the pod)

| File | Role |
|---|---|
| `bootstrap.sh` | Sets up the pod. It is idempotent and stops at the first failure: GPU count and driver gate, repo from the bundle, rustup, uv, `uv sync`, `maturin develop --release`, `assert_release_build`, the exact compile-stack check, `/root/sweep-cache`, topology, W&B check. |
| `launch.sh` | Pre-launch checks, then freezes the kit and generates `/root/sprint/NAME/run.sh`. It runs `setsid nohup`, reads the **real** pgid from the file that `run.sh` writes, starts the watchdog on that pgid, and prints how to stop. `--dry-run` only prints the plan. |
| `main_probe_auto.py` | The `main_probe.py` wrapper (nt-probe records, W&B online gate, `KG_NT_NUMA=cpu` affinity), with the topology discovered at runtime. `--topology N` prints the planned binding. |
| `watchdog.py` | The nonfinite-only watchdog of the live earn runs. Its per-rank counts are no longer fixed to 4 ranks. |
| `stop.sh` | Sends SIGTERM to the run's group and all its descendants, then SIGKILL after 60 s, then stops the watchdog. Writes `receipts/NAME/stop.txt`. |
| `first_iters.py` | Checks the first iterations from the log. It runs on the pod or on the Mac copy. |
| `copyoff.sh` | The Mac copy-off loop (`HOST PORT NAME [DEST]`). |
| `test_main_probe_auto.py` | Parser tests that run on the Mac. Not part of the repo test suite. |

**Topology semantics.** `main_probe.py` hardcoded `GPU_NODE {0:0,1:0,2:1,3:1}` and the two node CPU lists of `abl4mvr5w1mmn4`. The pod's `numa_bind` records show that each rank got its node's **whole** 128-CPU set, shared by the 2 ranks on that node, with no split.

The auto wrapper keeps that behaviour and finds the mapping at runtime: GPU index → `nvidia-smi` `pci.bus_id` (or the sorted `/proc/driver/nvidia/gpus` names as a fallback) → `/sys/bus/pci/devices/<bdf>/numa_node` → `node<N>/cpulist`, intersected with the allowed CPU set.

- `numa_node=-1`, or a device with no `numa_node` file (a kernel without NUMA): the affinity is left at the allowed set.
- A bus id that has no sysfs device is an error.
- `LOCAL_RANK` beyond the GPUs: it raises before torch loads.
- `launch.sh` exports `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so CUDA's device order is nvidia-smi's by definition. The 4-GPU runs did not set it; with identical GPUs it changes nothing else.
- A node is marked `oversubscribed` only when it has fewer CPUs than `ranks_on_node × (4 + 2)`. The flag is reported and does not change the binding. With 8 ranks on a typical 2-socket box (4 per node, 96 to 128 CPUs per node) it stays false. If it is true, look at the CPU count before launching; do not change the binding blind.

## Memory and `n_envs`

- Measured on 96 GB RTX PRO 6000 at horizon 720: 12 envs used 68.5 GB per GPU (27 GB headroom). 16 envs used 90.96 GB, which was judged too tight for rank 0's 10M evaluation.
- Rule of thumb: **about 5 GB per env plus about 10 GB base**. Keep at least 25 GB of headroom, so `n_envs ≤ floor((GPU_GiB − 35) / 5)`, capped at 12.
- By GPU size: 96 GB → 12 (the recipe). 80 GB (H100/A100-80) → 8 or 9. 141 GB (H200) → 12 (do not raise it).
- `launch.sh` prints the estimate. It refuses an obvious OOM, and it refuses `n_envs` above the fit unless `--allow-tight-memory` is given. A smaller `n_envs` also changes the optimizer steps and games per iteration (`n_envs` × ranks), so tell the owner.

## Driver gate (what happens on another driver)

`python/owl/model/compile_gemm.py` pins `KAGGRICULTURE_PROBED_COMPILE_STACK` to torch 2.9.0, triton 3.5.0 and NVIDIA driver `595.91.07`. `run_ppo` calls `check_compile_stack()` at startup for the compiled trunk, which the recipe uses. On any other driver it raises `RuntimeError: unprobed NVIDIA driver X for compiled Kaggriculture regions (probed: 595.91.07); repeat the ATEN-only GEMM A/B ...`. There is no override flag.

The 4-GPU pod passed only because its driver **is** 595.91.07 (`../pod4-2026-09-30/env/pod/hardware.txt`). `bootstrap.sh` checks this first and exits 3 with three options. Driver changes are ruled out: the repo rules forbid upgrading drivers.

- **A.** Get a pod with driver 595.91.07, for example the same GPU type or data center as `abl4mvr5w1mmn4`. This keeps the recipe identical.
- **B.** Repeat the ATEN-only GEMM A/B on the new stack, add the driver to the list on a branch with a cookbook record, and re-bundle. This costs the time of the A/B.
- **C.** Run eager with `--extra 'rl.model_compile=none'`. This skips the gate but changes the recipe (throughput, and eager versus compiled numerics), so the owner decides.

With `--allow-unprobed-driver`, bootstrap still finishes the build for B or C.

## Steps (Mac, zsh; run in order)

**0. Variables.** Take `HOST` and `PORT` from the owner's pod (the RunPod "SSH over exposed TCP" line). Take `CKPT` from the choice below.

```bash
HOST=root@<ip>; PORT=<port>
KIT=/Users/poonszesen/kg-v3-pod4/ops/rebuild-2026-09-29/sprint-8gpu
STAGE=/Users/poonszesen/kg-v3-runs/sprint-8gpu-stage; mkdir -p $STAGE
NAME=earn720-lr1e4-8gpu-from-promoted-$(date -u +%Y%m%d)      # new, never reused
KEY_OPTS=(-i $HOME/.ssh/id_ed25519 -o IdentitiesOnly=yes -o UserKnownHostsFile=$HOME/.ssh/known_hosts.runpod -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=30)
pod()   { ssh "${KEY_OPTS[@]}" -p $PORT $HOST "$@"; }
podcp() { scp "${KEY_OPTS[@]}" -P $PORT "$@"; }
pod 'nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv; nproc'   # first look
```

**Checkpoint choice.** Use the latest promoted `checkpoint_last_best.pt` of `pcy5knet`, copied off to the Mac and checked against the pod's sha256. When this kit was written, the newest was the 30M promotion, sha256 `60f23fd4eda7c529df114e89c552fc4a56714ec569a3d84f2249ede344b9036c`, in `/Users/poonszesen/kg-v3-runs/earn720-lr1e4-from-promoted2-4rank-20260930/promoted-30M/checkpoint_last_best.pt`.

That run's copy-off compares **size only**, so a later promotion of the same size would not reach the Mac. Before choosing, check W&B or the log for a newer promotion. Get it without touching the live run: a read-only `sha256sum` or `scp` on `abl4mvr5w1mmn4` needs the main agent's or the owner's OK.

```bash
CKPT=/Users/poonszesen/kg-v3-runs/earn720-lr1e4-from-promoted2-4rank-20260930/promoted-30M/checkpoint_last_best.pt
shasum -a 256 $CKPT          # write this into the launch record
```

**1. Bundle main.**

```bash
cd /Users/poonszesen/kg-v3-int && git fetch origin
SHA=$(git rev-parse origin/main); echo $SHA
git merge-base --is-ancestor 07c8fc9972297f28e0414da752700194a90d4fc3 $SHA && echo "contains 07c8fc99"
git bundle create $STAGE/sprint.bundle refs/remotes/origin/main && git bundle verify $STAGE/sprint.bundle
shasum -a 256 $STAGE/sprint.bundle
```

**2. Copy the bundle, the checkpoint and the kit.**

```bash
pod 'mkdir -p /root/sprint-kit /root/start'
podcp $STAGE/sprint.bundle $HOST:/root/sprint.bundle
podcp $CKPT $HOST:/root/start/checkpoint_last_best.pt
podcp $KIT/{bootstrap.sh,launch.sh,stop.sh,main_probe_auto.py,watchdog.py,first_iters.py} $HOST:/root/sprint-kit/
pod 'sha256sum /root/sprint.bundle /root/start/checkpoint_last_best.pt'   # must equal the Mac hashes
```

**3. Install the W&B key over stdin only.** It is never an argument, never printed and never in the kit. This follows `cookbook/workflows/install-the-wandb-credential-before-any-pod-launch.md`.

```bash
cd /Users/poonszesen/kg-v3-int && set -o pipefail
uv run python scripts/export_wandb_netrc_entry.py | ssh "${KEY_OPTS[@]}" -p $PORT $HOST 'set -eu; umask 077
  if [ -e "$HOME/.netrc" ]; then echo "pod ~/.netrc exists; merge the api.wandb.ai entry by hand" >&2; exit 1; fi
  tmp="$HOME/.netrc.wandb.$$"; cat > "$tmp"
  if [ ! -s "$tmp" ]; then rm -f "$tmp"; echo "empty credential; nothing installed" >&2; exit 1; fi
  chmod 600 "$tmp"; mv "$tmp" "$HOME/.netrc"'
pod 'stat -c %a ~/.netrc'     # expect 600; bootstrap step 11 checks wandb.Api().default_entity
```

**4. Bootstrap.** This takes about 3 to 5 minutes. It is idempotent, so if SSH drops, rerun it.

```bash
pod "bash /root/sprint-kit/bootstrap.sh --bundle /root/sprint.bundle --sha $SHA --expect-gpus 8"
```

Expect `BOOTSTRAP DONE ... gate_ok=1`, `compile stack accepted`, `wandb.Api ok, entity: spoon`, and a topology JSON with 8 ranks and `"oversubscribed": false`. On exit 3 (the driver gate), stop and bring options A/B/C to the owner. Receipts go to `/root/receipts/sprint-bootstrap/<stamp>/`.

**5. Launch.** Do a dry run first, then the real launch.

```bash
pod "bash /root/sprint-kit/launch.sh --checkpoint /root/start/checkpoint_last_best.pt --name $NAME --dry-run"
pod "bash /root/sprint-kit/launch.sh --checkpoint /root/start/checkpoint_last_best.pt --name $NAME"
```

It prints `LAUNCHED`, the pgid, the watchdog pid and the stop command. If the GPUs are not 96 GB, add `--n-envs <fit>` (see the memory section) and tell the owner.

**6. Check the first iterations.** Iteration 1 includes compile time; at 4 ranks it took 38.7 s against 14.7 s for later ones.

```bash
pod "python3 /root/sprint-kit/first_iters.py /root/runs/$NAME.log 8 12"          # after ~3 minutes
pod 'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv'
pod "tail -3 /root/runs/$NAME-watchdog.log"
```

- `first_iters.py` should show:
  - 8 `numa_bind` and 8 `start` records;
  - the W&B URL;
  - `bank_games=96.0` in every iteration (the 4-GPU run showed 48);
  - optimizer steps `+12` per iteration;
  - `env steps per iteration = 69,120`;
  - `OK first iterations match the expected shape`.
- GPU memory should be about 68.5 GB per GPU on 96 GB GPUs, since the per-rank work is unchanged; rank 0 rises during evaluation.
- Throughput: the 4-GPU run did about 2,350 env steps/s. Linear scaling would give about 4,700. That is an expectation to measure, not a pass mark.
- On the Mac, confirm on W&B that the run is `running` in `spoon/kg-v3`.

**7. Start the copy-off.**

```bash
mkdir -p /Users/poonszesen/kg-v3-runs/$NAME
nohup $KIT/copyoff.sh $HOST $PORT $NAME >> /Users/poonszesen/kg-v3-runs/$NAME/copyoff.log 2>&1 &
echo "copyoff pid $!"
```

**8. Stop (when the owner says so).**

```bash
pod "bash /root/sprint-kit/stop.sh $NAME 'owner stop'"
ONCE=1 $KIT/copyoff.sh $HOST $PORT $NAME >> /Users/poonszesen/kg-v3-runs/$NAME/copyoff.log 2>&1   # final pass
kill <copyoff pid>
```

Manual fallback: `pod "kill -TERM -\$(cat /root/receipts/$NAME/pgid)"` stops the launcher group only. The ranks are in their own sessions, which is why `stop.sh` walks the descendants.

**After launch**, record `launch.md` in a run folder beside this kit, as for `earnB/launch.md`. Include the owner quote, W&B URL, checkpoint sha, head, pgid, watchdog log, copy-off pid, the first-iteration check and the not-1:1 note.

## Validated before any pod (2026-09-30)

**On the Mac:**

- `bash -n` passed on every `.sh` under bash 3.2 and 5.3. shellcheck is not installed, so it did not run.
- `python3 -m py_compile` passed on every `.py`.
- `test_main_probe_auto.py` (15 tests) passes on Python 3.9 and 3.12. It reproduces the hardcoded 4-GPU table.
- `first_iters.py` on the live `pcy5knet` log (4 ranks, 12 envs) printed `OK`: 48 games and +12 optimizer steps per iteration.
- `launch.sh --dry-run` and each argument refusal were checked.
- `copyoff.sh` argument checks were run.

**In a throwaway Linux container** (`python:3.12-slim` plus procps and git; no network, no GPU, no training). The fakes were `nvidia-smi` and a torchrun that starts 8 setsid "ranks" printing `[nt-probe]` records.

`launch.sh` refused, as intended:

- topology failure with `--numa cpu`, writing nothing;
- an unprobed driver;
- busy GPUs;
- an OOM `n_envs`;
- 12 envs on 80 GB (it suggested 8);
- a reused NAME.

The launch path worked:

- The real pgid was read from `receipts/NAME/pgid`.
- The watchdog on that pgid stopped the run on a NaN `loss/policy` from rank 3. It listed and terminated the group and all 8 rank sessions; 0 were left afterwards.
- `stop.sh` ended a healthy run the same way. A second `stop.sh` reported nothing running.
- The generated `run.sh` wrote `SIGTERM`, `exit=143`, `end` and `DONE` to `times.txt`.
- `first_iters.py` printed 96 games and +12 steps per iteration.

`bootstrap.sh` on a real bundle of `origin/main` (`07c8fc99`):

- It rejected a bad `--sha` and the wrong GPU count.
- It checked out `07c8fc99` detached. The ancestry checks passed.
- With an unprobed driver it exited 3 and printed options A/B/C.
- A rerun with the probed driver was idempotent up to step 4, where rustup's curl failed with no network (`BOOTSTRAP FAILED at line …`).
- It refused to switch commits over a tracked edit.

`main_probe_auto.py` on the container's real sysfs (a kernel without NUMA): `--topology` and the per-rank binding gave node -1 with the allowed set. `LOCAL_RANK` beyond the GPUs raised a clear error.

**Not validated:**

- Nothing has run on a GPU pod. That includes real NUMA sysfs on a 2-socket host, `nvidia-smi` output, torchrun and NCCL with 8 ranks, and memory and throughput at 8 ranks.
- bootstrap steps 4 to 11: rustup, uv, `uv sync`, maturin, the version checks and W&B.
- `copyoff.sh` against a real host.
