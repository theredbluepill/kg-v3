You are Codex, stream D (evaluation opponents). Worktree: this checkout, `/Users/poonszesen/kg-v3-t71`, branch `kg/rebuild-7-1`, based on integration `b8747b6e8acece5f561d09a75bb914364a60ac05`. IMPLEMENT Task 7.1 exactly as recommended by the Codex-approved brief `ops/rebuild-2026-09-29/briefs/7.1-opponents.md`: import **starter, r04, ecobot and e776** as byte-exact native controllers with their own manifest, a small v3-owned registry, lifecycle/visibility checks and parity checks against each bot's **original reference behaviour**. Work test-first. Do not widen scope: no model, PPO, reward, grammar, panel script (Task 7.2), replay export (7.3) or packaging (7.4) work.

Read first: the whole brief (its "Import and evaluation contract" items 1-7 are requirements); `ops/rebuild-2026-09-29/plan.md` Phase 7, C2 and I0-I12; `docs/kaggriculture-contract.md` (v4.1; roles not seats, no opponent identity in model inputs); `CLAUDE.md`; `cookbook/index.md`, the newest `cookbook/log.md` entries and `cookbook/decisions/the-policy-is-stateless-and-observation-only.md`; `engine_rs/VENDORED_FROM.md`, `engine_rs/TRIM_MANIFEST.json`, `scripts/check_engine_trim.py`, `engine_rs/tests/replay_parity.rs`, `scripts/kaggriculture_parity/generate_traces.py` and `ops/rebuild-2026-09-29/merge-1.2/update_trim_manifest.py`. Brief review history: `ops/rebuild-2026-09-29/codex/stream-d-report.md` and `verify2-lane-D-r1.md` in `/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/codex/` (read-only; approval deferred parity, custody and lifecycle to this task). The reference branch `kg/reference-2026-09-29` (pin `65f0eac5bb00b18a9d3acce319c2a231cbd5dff0`) is an oracle: read it with `git show 65f0eac5bb00b18a9d3acce319c2a231cbd5dff0:<path>`; never write to it.

## Placement (Claude's design decision; follow it)

The vendored kernel `engine_rs/` is frozen: never edit its bytes, including `src/lib.rs` module lines, `Cargo.toml`, `Cargo.lock`, and never add files under `engine_rs/` (the trim checker fixes its inventory). The four controllers use `crate::{Farm, Game, Inventory, PrivateState, fib}` and e776 uses `include_str!("../../fixtures/e776-kenjo-trace.json")`, so put them in a **new standalone crate `opponents_rs/`** (package `kaggriculture-opponents`, lib `kaggriculture_opponents`, **edition 2024** like the engine; path dependency on `../engine_rs`; `serde_json` with the engine's `arbitrary_precision` + `preserve_order` features; its own tracked `Cargo.lock`). Create it with `cargo init --lib`/`cargo add --offline`, not hand-written TOML/lock edits.

- Byte-exact imports (copy with `git show <pin>:<path> > dest`, then verify SHA-256 against the brief's table):
  `opponents_rs/src/native_agents/{starter,r04,ecobot,e776}.rs` and `opponents_rs/fixtures/e776-kenjo-trace.json` (so the `include_str!` path resolves unchanged).
- Authored `opponents_rs/src/lib.rs`: an explicit (not glob) `pub use kaggriculture_engine::{...}` of exactly the crate-root names the imported files need (plus `Config` for Starter's inline tests), so their `crate::` paths resolve without edits; the registry module.
- Authored `opponents_rs/src/native_agents.rs` (NOT a `mod.rs`; do not copy the reference `native_agents/mod.rs`, which pulls in all 20 controllers) declaring only the four modules. If rustfmt/clippy flag the imported files, suppress only with attributes on these authored `mod` declarations, one lint per line with a reason comment; never reformat or edit imported bytes. Starter's five inline `#[cfg(test)]` cases must compile and run.
- If the separate crate cannot compile because a needed engine item is not `pub`, STOP that path and report the exact items; do not edit engine bytes and do not move code into the root crate.
- The root `owl` crate does not depend on `opponents_rs` yet (that edge belongs to the Task 1.4 binding).

## Manifest and custody

1. `opponents_rs/OPPONENT_MANIFEST.json` (dedicated manifest, brief item 1): schema_version, reference_commit; `imported` entries {path, reference_path, reference_sha256, sha256} that must be equal; `authored` entries {path, sha256, reason}; `python_oracles` {bot, source_repo `/Users/poonszesen/kaggriculture`, source_commit `e8884aae82eddeb7a1aeae99ecceeca7c830d67e`, files with SHA-256, provenance/license note}; `oracle_traces` {path, sha256, bytes, seed, policies for both seats, transitions, compared_actions per seat}; a byte budget for the traces (4,000,000 B). Record the license/notice gap from the brief and the sibling `PROVENANCE.md` files (EcoBot and E776 declare no software license; do not redistribute) - do not claim it resolved.
2. `scripts/check_opponent_import.py` + `tests/tools/test_check_opponent_import.py`, modelled on `scripts/check_engine_trim.py` (strict keys, safe paths, exact inventory of `opponents_rs/` excluding `target/`, hash checks, reference-blob comparison via `git show`, trace inventory/hash/size/budget). Test-first: drift, extra file, missing file, edited import byte, authored-hash, trace-hash and budget attacks each fail.
3. Wire into `justfile` (`rs-format`, `rs-lint`, `rs-test`, so `just prepare` covers them): the checker; `cargo fmt --manifest-path opponents_rs/Cargo.toml --check`; `cargo clippy --manifest-path opponents_rs/Cargo.toml --all-targets --locked -- -D warnings` (plus only the lint allowances you justified); `cargo test --manifest-path opponents_rs/Cargo.toml --locked`.
4. Register this task's changes in `engine_rs/TRIM_MANIFEST.json` **only through an updater script** `ops/rebuild-2026-09-29/7.1/update_trim_manifest.py` following the merge-1.2 pattern: `retained`/`authored` unchanged; the five imported reference paths' `excluded` reasons gain "byte-exact copy imported to opponents_rs/... under OPPONENT_MANIFEST.json (Task 7.1)"; append `non_engine_changes` entries for every non-engine path this task adds or changes. The updater must be idempotent and refuse unexpected input. `uv run --offline python scripts/check_engine_trim.py` must still pass.

## Registry, lifecycle, visibility, determinism (Rust, test-first)

- `OpponentKind` with exactly `starter`, `r04`, `ecobot`, `e776` string keys; unknown keys fail with an explicit error.
- One fresh controller per environment, seat and episode (brief item 2): a per-seat wrapper that owns the controller, is bound to one seat, calls `action` exactly once per step, fails explicitly on a repeated or skipped step or wrong seat, and has an explicit `reset` for a new episode. Tests: two seats get independent state; reset reproduces a fresh controller's actions; repeated/skipped step errors.
- A bounded Rust match runner `play_match(config, seed, [kind; 2])` on the default game configuration only (brief item 4): drives the engine's own action path with the controllers' official `farmer`/`hands`/`market` JSON (no truncation, no reinterpretation), records per seat the applied action, whether the engine accepted it (legality judged by the engine, not by the bot), controller errors, final raw banks and winner. No opponent identity leaves this evaluator bookkeeping. Tests: same seed twice -> identical action-hash sequence and banks; a different seed -> different sequence (non-vacuity); non-default config is rejected explicitly.
- Hidden-state perturbation (brief item 3): at several steps across a game, for each bot, clone controller and game, perturb the rival's private state and any RNG/hidden state reachable through public engine API, and assert the action is unchanged; include a positive control that perturbing the bot's own private state (or public state it reads) does change some action, so the test is not vacuous. If some hidden field cannot be perturbed without engine edits, list it as a gap.

## Parity against reference behaviour (the core check)

The reference behaviour is the ORIGINAL submissions, not the Rust copies:
- starter: `starter_agent` in `kaggle_environments==1.32.7` `envs/kaggriculture` (already `builtin:starter` in the generator);
- r04: `agents/r04/main.py` (SHA-256 `22d074391822206872448a6114a37ba2a4a39eda2ddbbcdc0bc8aa8cc6a64188`);
- ecobot: `agents/ecobot/main.py` (`0dc02e03c94ef60c06b5093efc2e2fd0530aa6eea20df507a90b90d6651bd067`);
- e776: `agents/e776/main.py` + `e776_pkg/` + its data per `agents/e776/MANIFEST.sha256` (entry `kaggriculture_e776_agent`, main `0cc2a88594f82b6c8d3cb15fcd4aa2df2bdd2a0a7fb0133d95a3204dd2d6ba38`),
all read-only from `/Users/poonszesen/kaggriculture` at commit `e8884aae82eddeb7a1aeae99ecceeca7c830d67e` (read blobs with `git -C /Users/poonszesen/kaggriculture show e8884aae...:<path>` or verify file hashes; never write there; do not copy these Python sources into this repo).

1. Extend `scripts/kaggriculture_parity/generate_traces.py` (existing `kaggriculture-re-parity-v1` format, Kaggle's own Python engine) with policies `sibling:r04`, `sibling:ecobot`, `sibling:e776` loaded hash-verified, one FRESH module/agent instance per seat per game (these bots keep module-level state; purge/rename modules so seats never share state), and a preset `opponents` that writes to `opponents_rs/fixtures/oracle/` with its own MANIFEST.json. Default configuration only. Games: every bot in BOTH seats, e.g. starter-vs-r04, r04-vs-ecobot, ecobot-vs-e776, e776-vs-starter at one seed each, plus one more seed if the 4 MB budget allows. Extend `tests/scripts/test_kaggriculture_parity.py` for the new policies (hash mismatch refused, fresh instance per seat). Generation must stay bounded (at most two live games per invocation; each under two minutes; otherwise hand the exact command back).
2. `opponents_rs/tests/oracle_parity.rs`: for every oracle trace, rebuild the game from configuration and seed in the native engine, and at every step have fresh native controllers for both seats produce actions from the native state, compare each with the recorded Python action (canonical JSON content, number representation included), apply the recorded actions and compare the expected state. Report actions compared and first mismatch with step/seat/field. Any mismatch is a finding to report with its cause; do not edit imported bytes and do not weaken the comparison to pass.
3. Coverage accounting (brief item 6): count per bot in the oracle traces openings, day resets, weed presence, market shortages/rejected orders, hires, late liquidation (sales in the final day) and any mid-episode reset/replay path; assert the counts the traces actually exercise and list uncovered categories as explicit gaps.
4. Non-vacuity: for each oracle comparison, show one mutation fails it (e.g. a tampered recorded action, swapped seats, a changed seed) and record the red.

## Binding-dependent pieces

Task 1.4's Python `owl.rs.KaggricultureEnv` binding is not merged. Implement only the Rust side above. Add `tests/owl/kaggriculture/test_opponents.py` with the learned-seat-vs-opponent test(s) the binding will enable, each `pytest.skip("needs Task 1.4 binding")`, stating the intended API briefly. Do not create a substitute binding.

## Mac rules (owner's Mac, non-negotiable)

- CPU-light only: no training, GPU, model runs or panel runs. Bounded checks only (at most two live games per diagnostic, under 1 GB RAM and two minutes each).
- Export in every shell: `CARGO_BUILD_JOBS=2 CARGO_NET_OFFLINE=true UV_OFFLINE=true RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 TMPDIR=/Users/poonszesen/kg-v3-t71/.codex-tmp`. Use `--offline` for cargo/uv/uvx; invoke just as `uvx --offline --from rust-just just`. If a crate or package is not cached offline, stop and report; never go online.
- GIT: your sandbox cannot write `.git`. Do NOT commit, stage, branch, push or merge. Leave changes in the working tree. Do not touch other worktrees.

## Docs and cookbook

Update `docs/rules-parity-coverage.md` (opponent parity: what is tested, what is not) and any doc `just docs-fresh` maps to your changes. Add one cookbook adaptation record (Reference under `cookbook/references/`, first tag `kaggriculture-v3`, required provenance fields, literal `repository:` sources, actual checks, gaps: license/notice custody, uncovered oracle categories, default-config-only support, binding-dependent tests), its folder `index.md` line, and a prepended `cookbook/log.md` entry. Record red/green receipts under `ops/rebuild-2026-09-29/7.1/`, planned expectations separate from actual results.

## Run at the end and report actual results

```
cargo test --locked --offline --manifest-path opponents_rs/Cargo.toml
cargo test --locked --offline --manifest-path engine_rs/Cargo.toml
uv run --offline python scripts/check_engine_trim.py
uv run --offline python scripts/check_opponent_import.py
uv run --offline pytest tests/tools/test_check_opponent_import.py tests/tools/test_check_engine_trim.py tests/scripts/test_kaggriculture_parity.py tests/owl/kaggriculture/test_opponents.py -q
uvx --offline --from rust-just just prepare
```

FINAL REPORT (Claude reads this; precise and brief):
1. Files changed/added, one line each with purpose.
2. Each command above with actual pass/fail/ignored/skipped counts or exit status; red/green evidence per piece; mutation checks performed.
3. Parity results per bot and seat: traces, seeds, actions compared, mismatches (first mismatch detail), coverage counts and gaps; oracle fixture sizes vs budget and SHA-256s.
4. Deviations from the brief or this prompt, with reasons.
5. Open questions and unresolved items, one line each.
