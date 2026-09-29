# Kaggriculture evaluation opponents

This standalone edition-2024 crate imports Starter, R04, EcoBot and E776 plus
E776's executable policy tape byte-exactly from reference commit
65f0eac5bb00b18a9d3acce319c2a231cbd5dff0. The root training crate has no dependency
on it. The dedicated OPPONENT_MANIFEST.json records source, authored files and
original-Python oracle custody; run scripts/check_opponent_import.py from the
repository root. That default check (also run by `just prepare`) needs no
sibling repository and pins the original entry hashes structurally; add
--original-sources on the owner's machine to re-read every original file.

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

Original-Python parity: on eight default-config oracle games generated on
CPython 3.11.15 (the Kaggle simulation image's interpreter), seeds
20260929-20260936, every bot plays both seats twice and all 11,504 recorded
original-submission actions match, with full state agreement after every
transition. Oracles from other interpreters are refused: CPython 3.12's
compensated float sum() changes R04's decisions (see
ops/rebuild-2026-09-29/7.1/run2/r04-mismatch.md). Denominators, coverage and
gaps are in docs/rules-parity-coverage.md and
ops/rebuild-2026-09-29/7.1/review/results.md. Mid-episode replay is also checked
against Python: fresh original controllers rebuilt from each oracle's prefix at
steps 37, 360 and 695 resume for 24 steps (fixtures/replay/REPLAY.json.gz), and
fresh native controllers rebuilt the same way match all 1,152 resumed actions
and the 24 final states (ops/rebuild-2026-09-29/7.1/verify-r1/).

## Notices and redistribution limits

The engine's Apache-2.0 license and provenance remain at engine_rs/LICENSE and
engine_rs/VENDORED_FROM.md. That license does not resolve original controller
license custody. Python oracle sources are read-only Git blobs from
/Users/poonszesen/kaggriculture at e8884aae82eddeb7a1aeae99ecceeca7c830d67e;
they are not copied into this repository. Starter is Kaggle's builtin starter
from kaggle-environments 1.32.7. R04 has no agent-level PROVENANCE.md.
EcoBot and E776 provenance declares no software license and warns against
redistribution. Do not redistribute these imported controllers or their data
while that gap remains unresolved. Exact notice text and hashes are preserved
in ops/rebuild-2026-09-29/7.1/python-oracle-source-audit.json.
