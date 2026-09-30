The pairing is correct, but the proposed replay check needs revision.

1. **P1 — The placeholder seed causes valid turns to fail.** The [new check](/Users/poonszesen/kg-v3-bcbrief/ops/rebuild-2026-09-29/briefs/5.1-bc-data.md:93) steps headers that still specify `seed: 0`, while the brief treats `info.seed` as custody only. Both Kaggle and the [merged Rust engine](/Users/poonszesen/kg-v3-bcbrief/engine_rs/src/lib.rs:4441) use the resolved episode seed for daily weeds and shop unlocks. Observation encoding needs no real seed; transition verification does.

   I ran an in-memory check against the cached Python engine whose SHA-256 matches the Rust pin, using local episode `114406062`:

   | Pairing and seed | Matching transitions |
   |---|---:|
   | `steps[t+1].action`, recorded seed `863363664` | 719/719 |
   | `steps[t+1].action`, placeholder seed `0` | 704/719 |
   | Shifted `steps[t].action`, recorded seed | 0/719 |

   Require the recorded seed in the replay-check header, keeping it outside model inputs. Otherwise line 95’s rejection rule drops correctly paired demonstrations.

2. **P2 — Specify the executable path for the new check.** Rust already supplies `Game::from_header → step → snapshot`, so the check is implementable. However, the merged Python header binding only encodes observations; [Task 1.4 explicitly excludes a production load-state API](/Users/poonszesen/kg-v3-bcbrief/ops/rebuild-2026-09-29/briefs/1.4.md:580), and this brief prohibits ad hoc bindings. Name an offline Rust verification harness, its invocation from preparation, and its per-turn results. Include it in the implementation checklist and source custody.

3. **P2 — Make the hard-link safeguard explicit.** A separate checkout/venv at [line 343](/Users/poonszesen/kg-v3-bcbrief/ops/rebuild-2026-09-29/briefs/5.1-bc-data.md:343) does not isolate installed files sharing inodes. Require either no in-place edits of installed files in either venv, or a fresh environment created with `uv --link-mode=copy`. That is the actual constraint recorded in the [pod environment Reference](/Users/poonszesen/kg-v3/cookbook/references/pod-v3-environment-runs-flash-attn-2-8-3-forward-on-sm120.md:40).

4. **P3 — Reconcile custody and cookbook records.** The brief still says volume access is required, the ZIP destination/retention are unknown, and the GPU is the fallback (lines 19, 49–57, 269). Update these and the required cookbook note/index/log together. The archive hash matches the reference’s `source.json`; cite that receipt rather than `data-manifest.json`.

The episode semantics are independently supported by [Kaggle core](/Users/poonszesen/.cache/uv/archive-v0/px2dKviBRRYjAZ0BRXW1A/kaggle_environments/core.py:275): it attaches submitted actions, runs the interpreter, then appends the resulting state. The pinned reference `prepare.py` uses exactly this pairing.

The primary-pod choice follows the owner direction. No-learner, capped-worker, compressed-output and measured-free-disk conditions are retained. I verified historical hash consistency, not live pod state. No files changed; the executed transition probe used Python, not Rust.

VERDICT: REVISE