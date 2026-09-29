# Kaggriculture evaluation opponents

This standalone edition-2024 crate imports Starter, R04, EcoBot and E776 plus
E776's executable policy tape byte-exactly from reference commit
65f0eac5bb00b18a9d3acce319c2a231cbd5dff0. The root training crate has no dependency
on it. The dedicated OPPONENT_MANIFEST.json records source, authored files and
original-Python oracle custody; run scripts/check_opponent_import.py from the
repository root.

Use OpponentKind's exact string keys starter, r04, ecobot and e776. Construct
a SeatController for each environment, seat and episode. Its action method must
be called once per step. It rejects a wrong seat, environment/episode,
repeated/skipped turn and completed game. Call reset with a fresh step-zero
Game at each new episode. Restoring a controller mid-episode requires replaying
the prefix from step zero, since the scripted bots retain their own state.
This memory and opponent labels stay in evaluator bookkeeping, never learned
actor/critic inputs, rewards, normalization or checkpoint selection.

Game owns the frozen engine through an opaque holder and refreshes a full
StepSnapshot after construction and every successful step. Failed steps retain
the prior snapshot. Its API exposes public state, both seats' private states and
statuses/rewards, plus Starter's integer hire-cost multiplier. The opaque holder
exposes no engine reference, RNG, seed or hidden-counter getters to controller
modules. Because both private states remain accessible, tests perturb the rival
state at seven checkpoints in both seats for all four bots, with own-state
positive controls. Engine RNG, seed and hidden counters cannot be perturbed
through this view. The boundary prevents reads rather than testing arbitrary
mutations of those inaccessible fields.

Game::from_engine adopts an existing native engine; its caller must supply that
engine's original Config. The frozen engine has no config getter, so this
precondition cannot be independently verified. The snapshot clone is an
evaluation-path cost; no training-throughput or Mac performance claim is made.

play_match accepts exactly Config::default(), runs at most 719 transitions and
passes official farmer/hands/market JSON unchanged. It records each submitted
action, controller errors and joint engine transaction acceptance for both seats.
A successful transaction does not mean every unit command or market order had
an effect. The engine's public joint market metrics preserve submitted,
committed and zero-commit counts; its per-seat metrics are private and are not
fabricated here. Failed transactions stop the match with an explicit engine
error. Only completed raw banks determine the winner; None denotes either a
draw or an incomplete result, distinguished by completed. Starter's inline
tests exercise their own custom configs; this does not broaden match support.

Original-Python parity is currently blocked: the first trace (seed 20260929,
Starter seat 0 versus R04 seat 1) disagrees at step 12, R04 hands[2][0], native
WEST versus Python NORTH. Both seats have 13 actions compared, with 13/12
matches, and the preceding 12 complete state transitions agree. The parity
test remains failing; imported bytes and strict action comparisons are unchanged.
Corpus expansion stopped. Other bot/seat combinations are not qualified.
See ops/rebuild-2026-09-29/7.1/results.md for diagnosis, counts and coverage.

## Notices and redistribution limits

The engine's MIT license and provenance remain at engine_rs/LICENSE and
engine_rs/VENDORED_FROM.md. That license does not resolve original controller
license custody. Python oracle sources are read-only Git blobs from
/Users/poonszesen/kaggriculture at e8884aae82eddeb7a1aeae99ecceeca7c830d67e;
they are not copied into this repository. Starter is Kaggle's builtin starter
from kaggle-environments 1.32.7. R04 has no agent-level PROVENANCE.md.
EcoBot and E776 provenance declares no software license and warns against
redistribution. Do not redistribute these imported controllers or their data
while that gap remains unresolved. Exact notice text and hashes are preserved
in ops/rebuild-2026-09-29/7.1/python-oracle-source-audit.json.
