# Kaggriculture evaluation opponents

This standalone edition-2024 crate imports Starter, R04, EcoBot and E776 plus
E776's executable policy tape byte-exactly from reference commit
65f0eac5bb00b18a9d3acce319c2a231cbd5dff0. Cha22's full ig_agent port comes from
the same commit with its dependency closure (V43, V47, V48, Farm2945, Metav4,
Pipe16): 27 Rust files plus 3 embedded data fixtures are byte-exact, and 8 files
differ only in the three Game-view accessor lines (public snapshot reference,
privates() and configuration()), which the checker re-derives from the pinned
blobs. The root training crate depends on it only for fixed-opponent collection
(env.opponent_mix, through HostedSeat below); every registry key, Cha22
included, can be hosted there. The dedicated OPPONENT_MANIFEST.json records source, authored files and
original-Python oracle custody; run scripts/check_opponent_import.py from the
repository root. That default check (also run by `just prepare`) needs no
sibling repository and pins the original entry hashes structurally; add
--original-sources on the owner's machine to re-read every original file.

Use OpponentKind's exact string keys starter, r04, ecobot, e776 and cha22. Construct
a SeatController for each environment, seat and episode. Its action method must
be called once per step. It rejects a wrong seat, environment/episode,
repeated/skipped turn and completed game. Call reset with a fresh step-zero
Game at each new episode. Restoring a controller mid-episode requires replaying
the prefix from step zero, since the scripted bots retain their own state.
This memory and opponent labels stay in evaluator bookkeeping, never learned
actor/critic inputs, rewards, normalization or checkpoint selection.

Cha22 embeds three data fixtures through include_str!/include_bytes! in
byte-exact files. v43-routes.json (4.9 MB, the decompressed V43 route tapes) is
read on every Cha22 turn. farm2945-sell-library.bin and metav4-sell-library.bin
(2.7 MB together) are needed only to compile farm2945/race.rs: Cha22's path
through Metav4 does not call the Farm2945 predict layer, and a load probe saw
no library read across four full Cha22 matches. Dropping them would mean editing
imported code.

Game owns the frozen engine through an opaque holder and refreshes a full
StepSnapshot after construction and every successful step. Failed steps retain
the prior snapshot. Its API exposes public state, both seats' private states and
statuses/rewards, plus Starter's integer hire-cost multiplier. The opaque holder
exposes no engine reference, RNG, seed or hidden-counter getters to controller
modules. Because both private states remain accessible, tests perturb the rival
state at seven checkpoints in both seats for every registered bot, with own-state
positive controls. Engine RNG, seed and hidden counters cannot be perturbed
through this view. The boundary prevents reads rather than testing arbitrary
mutations of those inaccessible fields.

HostedSeat serves a host that owns and steps its own engine: the root crate's
native training environment (env.opponent_mix). It holds an engine-less Game
view, so it can never step, and a SeatController with the same lifecycle
checks. Construct it with the step-zero snapshot of a new game (one per
environment, seat and episode) and call action with the pre-step snapshot once
per host transition. tests/hosted.rs shows a host-driven pair reproduces
play_match's actions and final banks for every bot in both seats. The host
clones the controller per step so a failed batch can retry the same turn.
Bot identity and state stay in the host's bookkeeping, never in learned
inputs.

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

Cha22 parity is deliberately light (owner request to accelerate the setup):
three default-config games against Starter, seeds 20260937-20260939, Cha22 in
seat 0 twice and seat 1 once (fixtures/oracle-cha22). The original submission
(SHA-256 127ed3e6..., entry ig_agent) ran under CPython 3.11.15 with
kaggle-environments 1.32.7; all 4,314 recorded actions (2,157 Cha22) match and
all 2,157 transitions agree on state. Regenerating those oracles under
PYTHONHASHSEED 0, 12345 and random reproduced the committed bytes. Limits kept
from v2's import: Python's equal-price ADV ordering depends on the hash seed
(Rust keeps tape order; these three games never reached a differing tie), and
the inactive PIPE opening alternatives are not exposed. The fuller v2 checks
(5,752 actions, 237 direct cases, 64 clones) are in
kaggriculture-v2 ops/cha22-opponent-import-2026-09-24/. The mid-episode Python
replay fixture covers only the four Task 7.1 bots.

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

Cha22 is Apache-2.0: notices/cha22/ keeps kaggriculture-v2's NOTICE.md and the
upstream notebook/README text byte-exact (v2 commit 30a3ac47), plus every
comment line of the original main.py in order (UPSTREAM-SOURCE-COMMENTS.txt),
which carries the full license text and its layers' attributions. The original
main.py is not copied; `--original-sources` reads a copy named by
KAGG_CHA22_SOURCE and checks its hash.

The imported Cha22 Rust headers (src/native_agents/cha22/mod.rs, early.rs,
market.rs, tail.rs) cite `agents/cha22/main.py`. That path exists neither in
this repository nor in kaggriculture-v2 commit 30a3ac47; it is the v2
translation's working name. Read it as the original main.py with SHA-256
127ed3e62988c0474d386db6527ae8ca9de9bb1fe7004128557ddef67126c652 (the file
KAGG_CHA22_SOURCE names). The headers are byte-pinned imports and stay
unchanged. The Apache-2.0 text is kept only as the comment-prefixed lines of
notices/cha22/UPSTREAM-SOURCE-COMMENTS.txt, with no plain LICENSE file beside
the derivative. That suffices while the derivative trains locally and is not
distributed (the Kaggle build drops it). Before any redistribution, add the
plain license text and a statement of the changes made (Apache-2.0 section 4).
