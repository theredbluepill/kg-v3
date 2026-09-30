# Native rollout preparation and publication

Target: reduce the serial part of `KaggricultureVectorizedEnv.step` without
changing actions, rules, rewards, observations, caller buffer storage, seed
allocation, or transactional failure behavior. The owner's native brief is the
scope. This is a CPU implementation episode, not a new training option; the
standing board's lead recipe remains unchanged and no training run is opened.

Base: `b2276bc5b70073b58a27f9e5fbd52473dbd48569`, branch `kg/sps-rollout`.
Inputs: default 720-step game envelope, fixed-seed legal randomized action
programs, own-bank/margin reward, current PyO3 adapter and native lifecycle.
Expected discriminator: the new binary reproduces the old binary's per-field
bitwise trajectory hashes at 1, 4 and 8 native workers, including two complete
games per environment, auto-reset, snapshots/seed state, and selected truncation.
Existing injected failures and new cache/error-priority checks must preserve
all committed state and output bytes.

Microbenchmark: release extension, 20 environments, four native threads,
720 `step` calls per replicate, actions generated outside the timed region.
Compare identical workload/action/output digests and report Mac CPU native PyO3
step time separately from initialization and action generation. No GPU, PPO,
complete-update SPS, H200 or live-run speed claim follows from this benchmark.
Benchmark processes run without concurrent builds/tests when collecting final
samples; source, binary, platform and toolchain identities accompany results.

Stop when parity and required repository checks pass and the report/cookbook
record and requested local commit are complete. Any parity divergence must be
localized and repaired before the new path qualifies. No push or remote-machine
access is part of this episode.
