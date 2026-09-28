# Two-GPU SPS verification

Question: can the current 8,294,450-parameter Kaggriculture port complete finite
PPO updates on two full RTX PRO6000 GPUs, and what is steady-state complete-work SPS?
Use the canonical run_ppo/PPOTrainer via benchmark_kaggriculture, eight native
environments per rank, horizon64, BF16, compiled trunk, native_threads2.
Exclude three warmup updates and measure five (5120 game transitions, normally
10240 seat turns). Report startup separately. Stop at the bounded result, an
explained execution error, or a profile-backed fix plus remeasurement.

Pod:6mlzh6v4c89w2x, created for this verification, requested2PRO6000 Server,
catalog and actual pod GPU cost$4.18/hour plus disk. No previous pod is changed.
Copy source/config, logs, summary and optional nsys trace locally before stopping
the created pod; no long learner or selected checkpoint is established by this run.
Prior-best evaluation/promotion is disabled only inside the timing harness.

Live inventory/price/capacity read before creation. Source-image provenance:
runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 from the existing account pod
inventory; the project lockfile still governs Torch2.9.0.
