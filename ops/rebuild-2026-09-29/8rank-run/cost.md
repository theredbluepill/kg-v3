# 8x RTX PRO 6000: price and stock (read-only catalog read, DRAFT)

**Nothing was created.** Creating the 8-GPU pod needs the owner's explicit approval after seeing the live hourly price. That price must be re-read right before creation: stock is live, and the catalog price below is not a booking quote.

## Live read, 2026-09-30 ≈00:08Z (RunPod MCP, read-only)

Calls: `get-capacity` with `gpuCount` 8 and the RTX PRO 6000 family, deep-probed on CUDA 12.8, 12.9, 13.0, 13.1, 13.2, 13.3 and 13.4, plus the all-version matrix; `get-gpu-type` (count 8, POD, availability) for the Server and Workstation editions; and `get-network-volume 4llk4uaf20`.

### Stock for 8 GPUs per pod, by host CUDA version

| GPU type | 12.8 | 12.9 | 13.0 | 13.1 | 13.2 | 13.3 | 13.4 | Overall |
|---|---|---|---|---|---|---|---|---|
| RTX PRO 6000 Blackwell Server Edition (96 GB) | Out | Out | Out | Out | Out | Out | Out | `availability: NONE` |
| RTX PRO 6000 Blackwell Workstation Edition (96 GB) | Out | Out | Out | Out | Out | Out | Out | `availability: NONE` |
| RTX PRO 6000 Blackwell Max-Q Workstation (96 GB, Community only) | Out | Out | Out | Out | Out | Out | Out | unavailable |

This agrees with the owner's earlier live check (8x RTX PRO 6000 "Out" on CUDA 12.8 and 13.0) and extends it to 12.9 and 13.1–13.4. The only 8-count PRO 6000 rows with stock were MIG slices: 1g.24gb at "Medium" and 2g.48gb at "Low" ($4 and $8 per hour), both on 13.0. They are not full GPUs and are not substitutes.

For comparison, the same read at 2 GPUs per pod showed the Server Edition "Low" on 13.0 and 13.2 at $3.38/h (the lowest offer, 2 × $1.69).

### Catalog price (per GPU-hour; `pricePerHr` for 8 GPUs was null because nothing is in stock)

| GPU type | Community | Secure | Max per pod (Community / Secure) |
|---|---|---|---|
| Server Edition | $1.69 | $2.09 | 9 / 9 |
| Workstation Edition | $1.69 | $2.19 | 4 / 8 |

So an 8x pod lists at **$13.52/h** (Server, Community), **$16.72/h** (Server, Secure) or **$17.52/h** (Workstation, Secure only at 8 GPUs). The earlier 2x Secure pod billed $4.18/h (2 × $2.09).

## Totals for the planned hours (catalog price; re-read before approval)

| Block | Hours | Community Server ($13.52/h) | Secure Server ($16.72/h) | Secure Workstation ($17.52/h) |
|---|---|---|---|---|
| 6.3b setup and qualification (setup, seeds, memory smoke, threads sweep, all-reduce, 30-min complete-work) | 2 | $27.04 | $33.44 | $35.04 |
| Main run | 24 | $324.48 | $401.28 | $420.48 |
| Main run | 48 | $648.96 | $802.56 | $840.96 |
| Main run | 72 | $973.44 | $1,203.84 | $1,261.44 |

- The main-run length is the owner's decision (it is `launch.sh`'s `RUNTIME_HOURS`). The rows above are examples, not a plan.
- No 8-rank SPS has been measured, so this does not convert hours into env steps. After 6.3b's complete-work run, the hours per 10M-step interval (one checkpoint and one promotion check) is `10,000,000 / (global game SPS × 3,600)`. For scale only: the 2-rank 6.2 run measured 2,297 game SPS (iterations 2–235), which is about 73 minutes per 10M steps on 2 GPUs.
- A network volume adds its own storage charge; none is planned.

## The driver and the CUDA filter

`run_ppo`'s compile gate accepts only NVIDIA driver 595.91.07 (`python/owl/model/compile_gemm.py`). RunPod filters hosts by the CUDA version the driver reports, not by driver version. The 595 driver branch is inferred, not checked, to report CUDA 13.2, which would make `allowedCudaVersions: ["13.2"]` the filter most likely to land on the probed driver. `setup.sh` checks the driver on every GPU and stops on any other version. Another driver needs the ATEN A/B re-probe first (`ops/rebuild-2026-09-29/run-statements/aten-gemm-ab.md`).

## Region and the BC checkpoint

Network volume `4llk4uaf20` ("kggriv2-shared-1tb", 1,000 GB, STANDARD) is in **EU-RO-1**. It holds the original BC best at `/workspace/kg-v3-bc-2026-09-29/run/bc-20260929-142216/checkpoint_bc_best.pt`. Only a pod created in EU-RO-1 with that volume attached can read it. Any other pod gets the Mac's durable copy through `copy_bc_best.sh`.
