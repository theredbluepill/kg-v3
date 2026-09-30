# Stop: vs-cha22-4rank-20260930 (W&B kifqbyx5)

- **Why:** owner, 2026-09-30: "let's switch back to self play no matter what, and think about how do we get the agent to earn moneny for real?" then, on waiting for the 10M evaluation, "i think it will just lose all games."
- **When/how:** stopped by PID (`kill -TERM -47845`, ranks 47885-47888, watchdog 48195) at iteration 286, before the first 10M checkpoint, so no checkpoint and no fixed-bot evaluation exist. GPUs free afterwards (0 compute apps).
- **Custody:** final log copied to `/Users/poonszesen/kg-v3-runs/vs-cha22-4rank-20260930/vs-cha22-4rank-20260930.log`, sha256 `17b336c9d437e747b0dabbd86c814fb216b5a9c309b72095ae3143a81a442d22` (matches the pod). Mac copy loop (pid 80077) stopped.
- **Outcome (rank-0 game ends, 256 games each, train metrics):** win rate 0 at every game end. Own bank 62.8k (iteration 12, ~BC) -> 57.3k (45) -> 52.3k (68) -> 43.1k (90) -> 39.6k (102); margin vs cha22 -80.6k -> -112.1k; teacher KL 0.02 -> 1.92. BC greedy vs cha22 on CPU: 0/16, 61.9k vs 144.4k.
- **Reading:** with the term M margin scale at 50k every game was past the clamp, so the per-step reward was ~0.0006 (`train/reward_margin_abs_mean`), a near-zero-signal control; the policy still drifted downhill. One seed, stopped early; not a verdict on fixed-opponent training.
