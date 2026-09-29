# Brief — Phase 4: Teacher distillation (Claude), v2

Plan: Phase 4 (4.1 distributions and KL, 4.2 targets and cache, 4.3 model methods and trainer wiring, 4.4 configs); principles I0, I5, I6, I11; lessons L6, L14. Reviewer: Codex (brief review before any code, then one review per sub-task before it merges into `kg/isaiah-gap-closure`). The previous plan's Tasks 3.2–3.5 (`kg/reference-2026-09-29:ops/gap-closure-2026-09-29/plan.md`) supply the task shapes; this brief rewrites them for the rebuilt grammar heads (`python/owl/model/kaggriculture_actor.py`) and for the `TeacherTargets` protocol that stream C already merged.

Written on `kg/rebuild-trainer-model` at `4cac1a1`. Line numbers refer to that tree unless a branch is named.

This v2 replaces v1 as the operative text. Review history: `codex/brief-4-review.md` (v1 at `4ba9081`, REVISE: two P1 and three P2 findings plus a stale test option; all applied here — P1-1 stateless teacher dispatch in §3.2 and T15b; P1-2 grammar signature on the cached path in §1.2, §2.1, §3.1 and T15; P2-3 one calculation/cache dtype rule in §1.1–§1.3 and T6; P2-4 phase completion requires T18/T19 to execute in §6; P2-5 T18's metric and first-minibatch check; `--extra reference` removed from §5). The reviewer's open-point answers (single-chunk no-copy `concat`, Orbit's asymmetric `concat` kept) are adopted.

## 0. Inputs read, status and dependencies

**Isaiah's teacher path (read, not changed in behavior):**
- `python/owl/train/ppo.py`: `PPOConfig.teacher_*` (defaults 0.001/0.001/32; `scaling_6m` sets `last_best`, 0.005, 0.005, 128), `set_teacher_model` (stateless check, `supports_cached_*` gates, frozen teacher), `train_iteration` → `_precompute_teacher_targets` (chunked `no_grad` pass over `teacher_segments_per_minibatch` segments, outside DDP, joined by `type(chunks[0]).concat`), `_update_minibatch` (`teacher_targets.index(idx)`, `_output_action_kl` sums `per_player_entity` over its last dim, `_teacher_value_cross_entropy` at `:2773`), the loss (`teacher_kl` weighted by `policy_weight`; teacher value CE weighted by `_value_state_weight` = any live player per state).
- `python/owl/model/stateless_transformer_v1.py`: `CachedTeacherDistillationTargets` (masked `target_logits` plus selected-target size params, winner probabilities), `compute_teacher_distillation_targets`, `evaluate_actions_with_cached_teacher`, the combined `evaluate_actions_with_teacher`, `_critic_distillation` (masked `log_softmax`, fp32 under autocast).
- `python/owl/model/actor/discrete_targets.py:383–491`: the KL is **teacher-forced on the replayed action**: the target categorical over its full masked support, and the size distribution at the *selected* target. `python/owl/model/actor/common.py:113` `categorical_kl_from_logits(teacher_logits, student_logits, mask)`.
- `scripts/run_ppo.py`: last-best is created from the loaded weights and activated as teacher (`:221–246`), refreshed in place on promotion (`:459–467`, `_refresh_eval_model_from_weights`), and reloaded on resume (`:205–222`). `_teacher_obs_spec_for_student` (`:891`) accepts only entity-based specs.
- `docs/model-architecture.md` "Teacher Distillation".

**Stream C, merged (`07568e0`):** `python/owl/model/teacher_targets.py` (`TeacherTargets` protocol: `index`, classmethod `concat`), cookbook `references/ppo-trainer-seams-map-any-schema-and-alarm-on-replay-drift.md`. It left two items for Phase 4: whether `concat` validates symmetrically, and trainer annotations that still name `CachedTeacherDistillationTargets` because `base.py` types the model methods concretely.

**Kaggriculture model (this branch):** `KaggricultureTransformer` (`python/owl/model/kaggriculture.py`) and `KaggricultureGrammarActor.policy_core` (`kaggriculture_actor.py:230`). Event log-probs are `[E,2,252,12]`; `_policy` chunks rows by `head_rows_per_chunk` (11,096 at D = 256); `_run_trunk` guards and chunks the trunk. The model accepts any lead shape: it reads `obs.still_playing.shape` and reshapes to `rows = prod(lead)`, so segment-major `[N,T,2]` batches need no flattening.

**Evidence for sizing:** `kg/rebuild-model:ops/rebuild-2026-09-29/results.md` "Model-only SPS ceiling (component)" (`ddf1fb2`) and its `model-sps-ceiling-2026-09-29/README.md`; the "GEMM limits at our shapes" section for the chunk bounds.

**Dependencies (none blocks 4.1 or 4.2):**

| Needed by | What | Owner / where |
|---|---|---|
| 4.3 trainer tests | Kaggriculture rollout storage and action mapping in `ppo.py` (`_PPORolloutBuffer` is still Orbit-only; `_map_action_bundle` has no `KaggricultureActions` branch) | Task 3.1 trainer seam |
| 4.3 trainer tests | `ppo.py:1299–1302` views `output.winner_log_probabilities` as `batch_old_values`. Kaggriculture's are `[..,2,2]` against values `[..,2]`, so every Kaggriculture PPO update fails there today, teacher or not | Task 3.2 (value-mode guards) |
| 4.3 `run_ppo` tests, 4.4 | `KaggricultureEnvConfig` in `FullConfig` and the three Kaggriculture configs | `kg/rebuild-configs` (`59bcee9`, Task 3.4), not yet in this branch |
| real (non-synthetic) masks | native grammar tables binding | Task 1.4 |

Until a dependency lands, the test that needs it is written and marked `pytest.mark.skip(reason="needs <task>")`, and the skip is listed in the sub-task's report. No shim stands in for a missing seam.

## 1. The KL (4.1)

### 1.1 Definition

Notation: seat row `r`; unit frame `f ∈ [0, 241)`; market position `p ∈ [0, 11)` (10 queue slots plus the forced sentinel); policy slot `k ∈ POLICY_SLOTS = (1, 3, 4, 5, 6, 7, 8, 9, 10)` with width `W_k`. `a` is the **replayed** program (the rollout tokens). `M_k(r, ·)` is the **effective mask** that `policy_core` already uses for the density at that slot, given the replayed prefix (Task 2.3 brief §2–§3):
- unit slots: `unit_kind` masked by `unit_live`; item, quantity-present and digit masks indexed by the replayed kind/present/high;
- `market_kind`: queue availability, then HIRE removed where `actor_counts + exclusive_final_HIRE_prefix(p) ≥ hire_limit`;
- market item and digits indexed by the replayed kind.

Teacher and student each produce logits `z` in the density dtype (`_density_dtype`: FP32, or FP64 when the model runs in FP64 for exactness tests), masked as Isaiah masks target logits with that dtype's minimum: `ẑ = z.masked_fill(~M, torch.finfo(z.dtype).min)`. Then

```text
KL_k(r, f) = Σ_{v ∈ M_k} p_T(v) · (log p_T(v) − log p_S(v)),   p = softmax(ẑ)
           = categorical_kl_from_logits(ẑ_T, ẑ_S, M_k)          # common.py:113
```

**One dtype rule.** `categorical_kl_from_logits` computes in `torch.promote_types(teacher.dtype, student.dtype, float32)` instead of `.float()`: it upcasts BF16/FP16 to FP32 and never demotes. This is a schema-generic refactor of Isaiah's helper, identical for every input he passes (BF16/FP16/FP32 give FP32 as before); only FP64 inputs change, from a lossy FP32 cast to FP64. The masked fill therefore never meets a narrower dtype, so `finfo(z.dtype).min` stays finite in both FP32 and FP64. The cache stores the teacher's density dtype (FP32 in production, FP64 only in FP64 tests); cached admission accepts FP32 or FP64 slot logits and rejects anything else.

Weights:
- unit slots × `unit_live[r, f]` (`live & f < actor_counts`)
- market slots × `market_live[r, p]` (`live & p ≤ stop_position`)

These are exactly the weights on `log_probs` and `entropies` in `policy_core`. Each weighted term is placed into frame layout `[R, 252, 12]` with the same `place` scatter as the log-probs. Slots 0, 2 and 11 (implicit, zero density) stay 0.

Reduction: `per_player_entity[*lead, 252] = Σ_slots`. PPO's unchanged `_output_action_kl` sums it over frames to `[N, T, 2]`, the layout of `batch_policy_weight`, so each acting seat gets `Σ_frames Σ_slots KL`. The weighted mean over acting seats and `rl.teacher_kl_coef` then follow Isaiah.

### 1.2 Why this form (and what it is not)

- **Direction and gates follow Isaiah:** `KL(teacher ‖ student)`, the same masks and factorization gates as PPO replay, and zero for non-acting entries.
- **Teacher-forced conditionals, like Isaiah's size KL at the selected target.** Within a frame the factors are chained: `unit_item`'s mask and hidden depend on the replayed `unit_kind`. The sum of per-slot KLs at the replayed prefix is the chain-rule integrand of the joint-program KL evaluated at a behavior-policy sample, not at a teacher sample. It is Isaiah's estimator; unbiasedness for the joint KL is **not** claimed. Unit frames are conditionally independent given the observation (no cross-frame prefix), and so are market positions, except for the HIRE-capacity mask and STOP, which both depend only on earlier positions' replayed kinds.
- **Same masked supports on both sides, only when the grammar matches.** The student and the teacher see the same replayed tokens and the same runtime context, so `M_k` is identical **if and only if they share the grammar tables and `hire_limit`**. Replay admission does **not** establish this: removing HIRE from one model's `market_kind` table leaves a PASS-plus-STOP program legal for both while changing its probabilities, and with mismatched supports Isaiah's helper can return a negative "KL" (review probe: −0.27031). Each model therefore carries a host-side `GrammarSignature` (SHA-256 over the eight tables' names, shapes and bytes, taken once at construction, plus the live `action_spec.hire_limit`). Cached targets carry the teacher's signature, and the student rejects a different one before any kernel (§3.1); the combined path compares signatures and tables. The HIRE-capacity mask uses the *final* HIRE prefix, which the Task 2.3 enumeration test proved equals the sequential sampler's density.
- **Marginalization after STOP.** The program ends at the first final `NONE` (position `s`). Positions `p > s` are marginalized in the density (zero log-prob and entropy) because they are not part of the program. They therefore contribute **zero** KL, and the KL is never computed against logits the program never used. The STOP decision itself is distilled: at every `p ≤ s`, `market_kind`'s KL includes `P(NONE)`. At `p = s`, the item and digit masks are `{0}`, which gives exactly 0.
- **Degenerate supports give exactly 0.** Padded unit frames, unavailable positions, the forced sentinel at `p == order_limits` and inactive rows all have singleton supports. `log_softmax` gives 0 on the single entry and `p = 0` elsewhere, so `KL = 0` exactly, before any liveness weight.
- **Numerics.** No `-inf` enters the KL path: the `finfo(z.dtype).min` fill gives `p = 0` exactly at masked entries, the helper never narrows that dtype (§1.1), and `categorical_kl_from_logits` zeroes those terms with `masked_fill`, so neither the forward nor the backward pass forms `0 · ∞`. The density path keeps its `-inf` masking (Task 2.3). Because a masked entry contributes `exp(·) = 0` in both fills, `log_softmax` agrees on unmasked entries; test 4.1-T2 pins this.
- **Scale.** A seat's KL sums up to 241 × 5 + 11 × 4 conditional KLs. Orbit sums over its action entities. `teacher_kl_coef` stays Isaiah's 0.005 (recipe alignment). The larger per-seat sum is a recorded residual difference, not an escalation.

### 1.3 Where it is computed

Inside `policy_core`, while each slot's logits, mask and liveness are in hand, so no student distribution is materialized outside the core and the cached and combined paths share one code path:

```python
# kaggriculture_actor.py, inside the per-slot loop (unit and market stages)
# logits are already in the density dtype (FP32, or FP64 in FP64 tests)
masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
if collect_logits:
    slot_logits[slot] = masked                      # [B, 241 | 11, W_k]
if teacher_logits is not None:
    kl = categorical_kl_from_logits(teacher_logits[slot], masked, mask)  # promotes, never demotes
    slot_kl[slot] = kl * unit_live                  # market: * market_live
```

`policy_core` gains two trailing parameters, `teacher_logits: dict[int, Tensor] | None` and `collect_logits: bool`. `GrammarPolicyResult` gains `slot_logits: dict[int, Tensor] | None` and `kl: Tensor | None` (`[B, 252, 12]`, placed like `log_probs`). Keys are the `POLICY_SLOTS` ints; unit slots have frame dim 241 and market slots 11. `_policy` slices `teacher_logits[slot][rows]` per head chunk and concatenates `slot_logits` and `kl` per key when there is more than one chunk. Market logits are computed once per position (`kind_logits` before the loop), so `slot_logits[7]` is `kind_logits` masked with the final `kind_mask`, the mask used for the density.

## 2. Targets and cache (4.2)

### 2.1 Type

`python/owl/model/kaggriculture_teacher.py`:

```python
@dataclass(frozen=True)
class KaggricultureTeacherTargets(TeacherTargets):
    """Frozen last-best targets for one rollout, in the observation lead layout.

    slot_logits[k]: density dtype (FP32; FP64 only in FP64 tests)
    [*lead, 241, W_k] (unit slots) or [*lead, 11, W_k] (market slots),
    finfo(dtype).min outside the replay-conditioned mask.
    winner_probabilities: fp32 [*lead, 2], (self, opponent) from each seat's view.
    grammar: the teacher's GrammarSignature (host value, no tensor).
    """
    slot_logits: dict[int, torch.Tensor] | None
    winner_probabilities: torch.Tensor | None
    grammar: GrammarSignature

    def index(self, indices: torch.Tensor) -> Self: ...        # dim 0 (segments)
    @classmethod
    def concat(cls, chunks: Sequence[Self]) -> Self: ...       # dim 0, symmetric
    def nbytes(self) -> int: ...                               # tensor metadata only
```

- **`concat` validates symmetrically** (the Phase 4 decision stream C deferred). Every chunk must carry the same optional fields, the same slot keys and the same `grammar`; otherwise it raises `ValueError` naming the field. An empty list also raises. A single chunk is returned as is, with no copy. Isaiah's `CachedTeacherDistillationTargets.concat` keeps its inherited first-chunk rule: the trainer never builds mixed chunks, and stream C's tests pin that rule.
- **`nbytes()` joins the protocol** (`teacher_targets.py`), and `CachedTeacherDistillationTargets` implements it too (sum of `tensor.nbytes`; no device sync). PPO logs it as `teacher/cache_bytes`, the Phase 6.1 "teacher cache bytes" record.
- Rollout masks are **not** cached: the student recomputes them from the same tokens and its own tables, and the `grammar` check (§3.1) makes those tables the teacher's. `index` is plain `tensor[indices]` along dim 0 and carries `grammar` unchanged.

### 2.2 Size at the configured chunk shapes (arithmetic from contract widths)

Per seat row:
- unit: 241 × (20 + 16 + 2 + 32 + 32) = 24,582 values;
- market: 11 × (8 + 16 + 32 + 32) = 968 values;
- total 25,550 fp32 = **102,200 B**, plus 8 B of winner probabilities.

(The frame-layout alternative `[252, Σ W_k = 190]` would be 191,520 B per row; it is rejected.)

| Per rank | Rollout rows (`n_envs × 64 × 2`) | Teacher chunk rows (`min(128, n_envs) × 64 × 2`) | Trunk / head calls per chunk (dense, from the configs test) | Cache |
|---|---|---|---|---|
| 2-rank (`n_envs` 128) | 16,384 | 16,384 (one chunk) | ≤ 3 / 2 | 1,674,575,872 B = **1.560 GiB** |
| 4-rank (`n_envs` 64) | 8,192 | 8,192 (one chunk) | ≤ 2 / 1 | 837,287,936 B = 0.780 GiB |
| per update minibatch slice (2-rank, spm 8) | — | 1,024 rows | 1 / 1 | 104,660,992 B = 0.0975 GiB |

**Memory estimate (arithmetic on measured component peaks, not a measurement):**
- **Precompute:** the component probe's teacher proxy (C: `evaluate_actions` under `no_grad`, 16,384 rows in one call) peaked at 31.93 GiB allocated at dense (32.13 sparse, 30.74 mid). The proxy already computes each slot's full `log_softmax` per head chunk; the real path keeps the masked logits instead. It adds at most the two head chunks' logits plus `_policy`'s concatenation (≤ 2 × 1.560 GiB) and nothing for `concat` (single chunk, no copy). That gives **≲ 35.1 GiB** at dense.
- **Update:** the probe's B (1,024-row train step) peaked at 40.28 GiB. The persistent cache (1.560 GiB), the minibatch slice (0.098 GiB) and the KL's recomputed student `log_softmax` plus saved tensors (≈ 0.1–0.3 GiB at 1,024 rows × 25,550) give **≲ 42.3 GiB**.
- **Target:** 85 % of 97,887 MiB = 81.25 GiB, so both estimates fit.
- **Not included:** the rollout buffer, engine state, evaluation, DDP buckets and allocator fragmentation. Task 6.1 measures them.

**Reserved-memory risk (open).** The probe's caching-allocator reserved peak was 84.994 GiB (89.5 % of 94.97 GiB) during dense C run after B; the carryover explanation is an untested hypothesis. The trainer has the same order (the previous iteration's B, then C), and the targets path adds up to ~3.1 GiB of transients. Task 6.1 therefore records `max_memory_reserved` as well as `max_memory_allocated`. If it OOMs, the order of remedies is:
1. lower `teacher_segments_per_minibatch` (a chunk size with no learning effect, which Isaiah also varies across per-rank configs);
2. `torch.cuda.empty_cache()` before the precompute;
3. `expandable_segments`.

All three are untested.

**Time estimate (component, conditional):**
- **Precompute:** proxy C took 852.0 / 987.7 / 1,690.6 ms per 16,384 rows (sparse/mid/dense), 20–22 % of the synthetic update wall. The real path writes about 3 GiB more.
- **Update:** the KL adds elementwise work on 1,024 × 25,550 values per minibatch against a 147–327 ms B step.

Both are unmeasured. Task 6.2 reports `time/teacher_seconds` and `perf/teacher_sps`.

### 2.3 Chunking goes through the model's guards

`compute_teacher_distillation_targets` calls `self.encode_observations` (`_run_trunk`: overflow guard, padded chunking or packed row-boundary chunking) and `self._policy` (`head_rows_per_chunk`). It never calls `policy_core`, the compiled trunk or `_forward_transformer_trunk` directly. The student side of `evaluate_actions_with_cached_teacher` uses the same two entry points. The KL adds no GEMM, so the head bound `rows × 252 × max(3D, D, widest head) < 2^31` is unchanged. PPO's own `teacher_segments_per_minibatch` chunking stays as Isaiah wrote it, and the startup workload check (`kaggriculture_workload.py`, configs branch) already covers the teacher-chunk rows.

### 2.4 Value distillation via `winner_log_probabilities`

- **Teacher target:** `_winner_log_probabilities(encoded).exp()` reshaped to `[*lead, 2]`: the seat's own masked winner softmax over (self, opponent), fp32.
- **Student:** the `winner_log_probabilities` from the same encode that produces the PPO log-probs and values (one student encode per minibatch, as Isaiah's `_value_distillation_from_encoded`).
- **Per seat:** `CE_s = −Σ_j p_T[s, j] · log q_S[s, j]`.
- **Per state:** `Σ_s CE_s · live_s / max(1, Σ_s live_s)`, where `live = still_playing`. The loss then applies Isaiah's `_teacher_value_weighted_mean` with his state weight (`any(still_playing)`).
- **Why the mean:** Isaiah's Orbit state carries one joint winner distribution, so there is one CE per state. Kaggriculture gives one full distribution per seat view. Averaging keeps one CE per state, so `teacher_value_coef` 0.005 keeps its per-state meaning; summing would double it. A seat that is not live has a uniform, gradient-free distribution (Isaiah's all-masked rule) and is excluded rather than adding a constant `log 2`.
- **Where the reduction lives:** the reduction depends on the game's winner layout (joint over player slots vs per seat). It therefore becomes a model method rather than a `ppo.py` branch (§3.2).

## 3. Model methods and trainer wiring (4.3)

### 3.1 `KaggricultureTransformer`

- `supports_cached_teacher_distillation() -> True`; `supports_cached_value_distillation() -> True` (the critic is always the masked softmax).
- `compute_teacher_distillation_targets(obs, actions, *, compute_action_kl, compute_value) -> KaggricultureTeacherTargets`, under `torch.no_grad()`:
  1. `_check_action_layout`;
  2. `encode_observations` once;
  3. if `compute_action_kl`: `_policy(..., actions, collect_logits=True)`, then `check_replay_flags(result.valid)`. This is one host sync per chunk, which is one per iteration at the configured shapes. It rejects a program that the teacher's own grammar does not admit, with the flag group named; it does **not** detect a teacher grammar that differs while still admitting the program (§1.2), which is why the targets carry `grammar = self.grammar_signature()`;
  4. reshape to the lead;
  5. if `compute_value`: winner probabilities.

  A disabled target is `None` (Isaiah's behavior).
- `evaluate_actions_with_cached_teacher(obs, actions, teacher_targets, *, hidden_state=None, dones=None, compute_teacher_action_kl, compute_teacher_value) -> ModelTeacherEvaluation`:
  1. **Admission before any kernel:** stateless checks as in `evaluate_actions` (`hidden_state` and `dones` must be `None`; PPO's wrappers dispatch statelessly, §3.2). `isinstance(teacher_targets, KaggricultureTeacherTargets)`, else `TypeError`. A required target that is `None` raises `ValueError` ("cached teacher action targets are missing" / "value targets"). When the action KL is required, `teacher_targets.grammar != self.grammar_signature()` raises `ValueError` naming both signatures, slot keys must equal `POLICY_SLOTS`, slot dtypes must be FP32 or FP64, and shapes must equal `(*lead, 241 | 11, W_k)`; winner targets must be `(*lead, 2)`. Each failure raises `ValueError`.
  2. **Student:** one encode; `_policy(..., actions, teacher_logits=<flattened to rows>)`; `check_replay_flags`.
  3. **Assembly:**
     - The student `ModelEvaluation` is built exactly as `evaluate_actions` builds it: extract `_evaluation_from(result, encoded, obs)` and use it in both.
     - `action_kl = ModelActionKLDivergences(launch=zeros_like(per_player_entity), event=kl.reshape(*lead, 252, 12), per_player_entity=event.sum(-1), components={SLOT_NAMES[k]: event[..., k] for k in POLICY_SLOTS}, target=None)`.
     - Winner tensors: `[*lead, 2]`.
- `evaluate_actions_with_teacher(obs, actions, teacher, ...)` (the combined path Isaiah retains for the bit-for-bit contract):
  1. `isinstance(teacher, KaggricultureTransformer)`, else `ValueError`;
  2. `teacher.action_spec == self.action_spec` and `teacher.grammar_signature() == self.grammar_signature()`;
  3. per-table `torch.equal` (catches an in-place table edit after construction), else `ValueError`;
  4. student encode;
  5. teacher `encode_observations` and `_policy(collect_logits=True)` under `no_grad`;
  6. pass the live row-layout logits into the student `_policy` without the lead reshape or `TeacherTargets` round trip, then assemble as above.
- `teacher_value_cross_entropy(...)` override: per-seat CE and live-seat mean (§2.4).

### 3.2 Shared seams (refactors, no shims)

- **`python/owl/model/base.py`:**
  - `compute_teacher_distillation_targets -> TeacherTargets` and `evaluate_actions_with_cached_teacher(teacher_targets: TeacherTargets)`. Overrides narrow the argument with `isinstance` (Liskov-safe) and may return their concrete type.
  - New `teacher_value_cross_entropy(student_winner_log_probabilities, teacher_winner_probabilities, *, value_mask) -> Tensor` (per state). Its default body is Isaiah's formula moved verbatim from `ppo._teacher_value_cross_entropy`: `(-teacher.detach() * student).sum(dim=-1)`, with `value_mask` unused because his joint distribution already zeroes inactive slots.
  - The `supports_cached_teacher_distillation` docstring is made model-generic.
- **`stateless_transformer_v1.py`:** `evaluate_actions_with_cached_teacher` adds the `isinstance(teacher_targets, CachedTeacherDistillationTargets)` narrowing (`TypeError`), and the class gains `nbytes()`. Nothing else changes.
- **`python/owl/model/actor/common.py`:** `categorical_kl_from_logits` computes in `torch.promote_types(teacher.dtype, student.dtype, torch.float32)` (§1.1). Isaiah's inputs give the same FP32 computation as before; his teacher tests must pass unchanged.
- **`python/owl/train/ppo.py`:**
  - `TeacherTargets | None` annotations.
  - **Stateless teacher dispatch (review P1-1).** `_model_evaluate_actions_with_teacher` and `_model_evaluate_actions_with_cached_teacher` follow `_model_evaluate_actions`: when `hidden_state is None` they call the model without `hidden_state` and `dones`; otherwise they pass both. Isaiah's stateless model ignores sequence-shaped `dones` (`_encode_distillation_observations`), so Orbit results are unchanged, while Kaggriculture's stateless checks (which reject non-`None` `dones`) now hold on the teacher path. T15b tests both wrappers at helper level, without the pending rollout seam.
  - `teacher_value_loss_values = unwrap_model(self.model).teacher_value_cross_entropy(student_log, teacher_probs, value_mask=batch_value_mask)` at the same point, outside autocast, where the free function ran. The two `.view_as(batch_old_values)` calls on the teacher/student winner tensors are dropped: for Orbit those tensors already have that shape, so the result is bit-identical. `_teacher_value_cross_entropy` is removed, and `tests/owl/train/test_loss.py:321` retargets to the base method (call syntax only).
  - Log `teacher/cache_bytes = teacher_targets.nbytes()` when targets are precomputed.
  - `set_teacher_model`'s error text drops "discrete_targets actor" for "cached action-KL support".
- **`scripts/run_ppo.py`:** `_teacher_obs_spec_for_student` dispatches with `isinstance`:
  - both `EntityBasedBaseConfig`: Isaiah's `max_entities` rule, unchanged;
  - both `KaggricultureObsConfig`: exact equality, returning the teacher spec;
  - anything else: `TypeError` naming both types.

  This makes `teacher_mode: fixed` and a last-best seeded from `teacher_init` loadable for Kaggriculture. The last-best refresh (`_refresh_eval_model_from_weights`, `load_state_dict` in place) needs no change: the grammar tables are non-persistent buffers, so the refresh never overwrites them, and the compiled trunk callable closes over the same modules.

### 3.3 Invariants that must keep holding

- Stateless and observation-only: no hidden state, and no targets persisted beyond one iteration. The cache never enters a checkpoint, and a test checks the checkpoint key set.
- No opponent identity anywhere in targets, KL, value CE, weights or selection.
- The teacher is frozen (`requires_grad_(False)`, `eval()`), runs under `no_grad`, and is never DDP-wrapped.
- Bit-for-bit cached = combined (I6) on CPU fp32 for the same batch.

## 4. Configs (4.4)

- **Already present on `kg/rebuild-configs` (`59bcee9`):** `kaggriculture_2rank.yaml` and `kaggriculture_4rank.yaml` carry `teacher_mode: last_best`, `teacher_kl_coef: 0.005`, `teacher_value_coef: 0.005` and `teacher_segments_per_minibatch: 128`, not divided by world size, as in Isaiah's per-rank configs (`winner_ce_6m_4x5090` uses 256). `test_ranked_config_optimizer_and_ppo_equal_scaling_6m` covers them through whole-`rl` equality apart from `segments_per_minibatch`.
- **Phase 4 adds:**
  - one named test asserting the four teacher fields equal `scaling_6m`'s in both ranked configs (explicit, so a future `rl` exemption cannot silently drop them);
  - the CPU config's teacher settings;
  - the teacher cache bytes in the startup headroom line (`teacher_chunk` rows × 102,208 B, and rollout rows × 102,208 B for the whole cache), with a test pinning 1,674,575,872 B (2-rank) and 837,287,936 B (4-rank). It uses a constant derived from `kt.SLOT_WIDTHS`, `MAX_ACTORS` and `MAX_ORDER_LIMIT`, not a literal;
  - removal of "the last-best teacher needs Phase 4" from the configs Reference's pending list.
- **Launch semantics are Isaiah's, recorded for Phase 5/6:**
  - A launch from the BC best via `load_model_weights` seeds last-best from those weights and activates the teacher at iteration 1 (`run_ppo.py:235–246`).
  - A scratch launch has no teacher until the first promotion.
  - Nothing changes here.

## 5. Tasks (TDD; tiny CPU only)

Owner's Mac rules for every step: CPU only, `OMP_NUM_THREADS=2`, tiny models (`_tiny`: D = 16, depth 1), ≤ 2 envs where a trainer is involved, no training run and no GPU. Commands:

```bash
OMP_NUM_THREADS=2 uv run pytest tests/kaggriculture/test_teacher.py -q
OMP_NUM_THREADS=2 uv run pytest tests/owl tests/scripts tests/tools -m "not slow" -q
OMP_NUM_THREADS=2 uvx --from rust-just just py-prepare     # not full prepare: no Rust changes
```

Shared test helpers (`_tiny`, `_base_case`, `_replay_case`, `_map_obs`, `_cat_obs`, `_take_obs`, `_take_actions`) move from `tests/kaggriculture/test_model_heads.py` to `tests/kaggriculture/helpers.py` in 4.1's first commit (a refactor; the heads tests keep passing unchanged in behavior). A segment-major batch is `_map_obs(obs, lambda t: t.reshape(N, T, *t.shape[1:]))` over `make_obs(envs=N*T, ...)`, with the same reshape for actions.

### Task 4.1 — distributions and KL

**Files:** modify `python/owl/model/kaggriculture_actor.py` (`policy_core`, `GrammarPolicyResult`) and `python/owl/model/kaggriculture.py` (`_policy`, `_policy_chunk`, `_evaluation_from`); create `tests/kaggriculture/helpers.py` and `tests/kaggriculture/test_teacher.py`.

- [ ] **Step 1: failing tests** (`test_teacher.py`, 4.1 block)
  - **T1 identity and degenerate zeros.** For every `_replay_case` (dense, hire_capacity, stop_every_position, mixed), student = teacher (the same module): `kl` is `torch.equal` to zeros. With a perturbed teacher (`_tiny(seed=6)`), `kl` is `≥ 0` everywhere, finite, `> 0` somewhere, and **exactly 0** at slots 0, 2 and 11, at unit frames `f ≥ actor_counts`, at market positions after STOP, at the forced sentinel, at unavailable positions and on inactive rows (`mixed` has two).
  - **T2 collected logits reproduce the density.** For every live `(row, frame, slot)`, `log_softmax(slot_logits[k])` gathered at the replayed token equals `evaluate_actions(...).log_probs.event` at the placed position, with `torch.equal` on CPU fp32 and fp64 (`_obs_double`). If fp32 exactness fails, stop and report; do not loosen the tolerance silently.
  - **T3 brute-force oracle (the "same masked supports" proof).** On `_base_case` and `hire_capacity` (budget exhausted mid-queue), for each live `(row, frame|position, slot)` and each admissible `v`:
    - Set that token to `v` and run `policy_core` directly with supplied tokens (ignoring `valid`) for teacher and student, then read `log_probs` at that slot.
    - The admissible set comes from the tables indexed in Python by the replayed prefix, plus the HIRE rule `actor_counts + prior_final_HIREs < hire_limit`.
    - `Σ_v p_T (log p_T − log p_S)` must match `kl` there within 1e-6.

    This needs no second KL formula in the oracle beyond the definition.
  - **T4 chunking.** Monkeypatch `km.head_rows_per_chunk` to 2 and the trunk limit as in `test_model_encoder.py:445–484`. `slot_logits` and `kl` are `torch.equal` (heads) and `assert_close` (trunk, same tolerance as those tests) to the unchunked result. A spy counts ≥ 2 head calls and ≥ 2 trunk calls.
  - **T5 compile capture.** Extend the fullgraph test (`test_model_heads.py:669`): the captured core with `teacher_logits` and `collect_logits=True` equals eager.
  - **T6 gradients.** The KL loss sum gives finite gradients on every student head parameter and none on teacher parameters (`requires_grad_(False)`, `.grad is None`). In FP64 (student and teacher `.double()`, `_obs_double`), a central finite difference (ε = 1e-6) on one student head weight matches autograd within the existing FD test's `rel=1e-6`. This is valid only because the KL helper no longer demotes FP64 (§1.1); the review's probe (analytic 0.08540231 vs FD 0.08940697) is what the old cast produced, and a mutation restoring `.float()` must fail this test.
  - **Non-vacuity (KL tests).** Each mutation is applied, run and reverted, with its log kept under `ops/rebuild-2026-09-29/trainer-model/`: (a) KL computed without the student mask (`categorical_kl_from_logits(..., logits, mask)` on unmasked student logits) must fail T3; (b) the market KL not weighted by `market_live` must fail T1; (c) the helper's `.float()` cast restored must fail T6's FD check; (d) `slot_logits` collected from `-inf`-masked logits must fail T2 or T1 (NaN/inf).
- [ ] **Step 2:** run, expect FAIL (missing parameters and fields).
- [ ] **Step 3: implement** §1.3 and `_evaluation_from`. Keep `forward`/`evaluate_actions` behavior identical: `teacher_logits=None`, `collect_logits=False`, and `kl`/`slot_logits` `None`.
- [ ] **Step 4:** the 4.1 tests, `tests/kaggriculture` (all heads tests unchanged), the Isaiah suites and `py-prepare`.
- [ ] **Step 5: cookbook and commit.** Create `cookbook/references/kaggriculture-teacher-distills-per-slot-kl-and-per-seat-winner-ce.md` (type Reference; changed paths, reason, actual checks, gaps), update `cookbook/references/index.md` and prepend `cookbook/log.md`. Commit "Expose replay-conditioned per-slot logits and KL in the Kaggriculture grammar core (Phase 4.1)".

### Task 4.2 — targets and cache

**Files:** create `python/owl/model/kaggriculture_teacher.py` (`KaggricultureTeacherTargets`, `GrammarSignature`, `grammar_signature(tables, hire_limit)`, `TEACHER_TARGET_BYTES_PER_ROW`); modify `python/owl/model/teacher_targets.py` (`nbytes`), `stateless_transformer_v1.py` (`nbytes` on the cached type only), `python/owl/model/__init__.py` (export), `kaggriculture_actor.py` (the tables' digest, taken once at construction) and `kaggriculture.py` (`grammar_signature()`, `compute_teacher_distillation_targets`).

- [ ] **Step 1: failing tests**
  - **T7 shapes and flags.** On a segment-major `[N=3, T=2, 2]` batch: every unit key is `[3, 2, 2, 241, W_k]` fp32 and every market key `[3, 2, 2, 11, W_k]`; winner probabilities are `[3, 2, 2, 2]` and sum to 1. `compute_action_kl=False` gives `slot_logits is None`; `compute_value=False` gives `winner_probabilities is None`. The teacher must not require grad on outputs.
  - **T8 replay admission in the teacher path.** A non-canonical program (from `test_model_heads.py`'s rejection cases) raises `GrammarReplayError` from `compute_teacher_distillation_targets`. So does a teacher built with `hire_limit = actor_counts + 1` replaying a program with 3 HIREs that the student (`hire_limit` 241) sampled; the support group is named.
  - **T9 index/concat.** Three layouts (both targets, KL only, value only) × chunk sizes {1, 2, 3} over `N = 3`: `concat([t.index(a), t.index(b), ...]) == t`, exactly per tensor, and `grammar` is carried. A single chunk returns the same object. Mixed presence raises in **both** orders; mismatched slot key sets raise; mismatched `grammar` raises; an empty list raises. Error texts name the field.
  - **T9b grammar signature.** Equal for two models built from the same tables and `hire_limit`; different when one table entry flips or `hire_limit` differs; stamped on the targets by `compute_teacher_distillation_targets`.
  - **T10 bytes.** `nbytes()` equals the sum of tensor `nbytes`, and equals `rows × 102,208` for full targets on a `_tiny` model (widths come from the contract, not D). `CachedTeacherDistillationTargets.nbytes()` works on Isaiah's existing fixtures.
  - **T10b cache arithmetic.** A module constant `TEACHER_TARGET_BYTES_PER_ROW` derived from `kt.SLOT_WIDTHS`, `MAX_ACTORS` and `MARKET_POSITIONS` equals 102,208, and rollout rows give 1,674,575,872 B (2-rank, 16,384 rows) and 837,287,936 B (4-rank, 8,192 rows). Phase 4.4 reuses the constant.
  - **T11 chunked precompute.** The trainer-style loop (`index` segments in chunks of 1, then `concat`) equals one whole-batch call: `torch.equal` for heads-only differences, `assert_close` where the trunk batch size differs (mirror `test_ppo.py:2939`).
  - **Non-vacuity (target tests).** Mutations, each run and reverted with logs kept: (a) `concat` validating only against the first chunk must fail T9's reversed-order case; (b) `index` slicing dim 1 must fail T9; (c) `grammar` dropped from `concat`'s equality check must fail T9's mismatched-grammar case; (d) `nbytes` counting only slot logits must fail T10.
- [ ] **Steps 2–4:** FAIL; implement §2.1 and §3.1's first method; the tests, the Isaiah suites (stream C's `test_teacher_targets.py` unchanged) and `py-prepare`.
- [ ] **Step 5:** update the Phase 4 Reference, the index line and the log. Commit "Cache Kaggriculture teacher targets under the TeacherTargets protocol (Phase 4.2)".

### Task 4.3 — model methods and trainer wiring

**Files:** `kaggriculture.py` (cached, combined and value-CE methods, `supports_*`), `python/owl/model/base.py`, `stateless_transformer_v1.py` (narrowing only), `python/owl/train/ppo.py`, `scripts/run_ppo.py`, `tests/owl/train/test_loss.py` (call syntax), `tests/kaggriculture/test_teacher.py`, `tests/scripts/test_run_ppo.py`; docs: `docs/model-architecture.md` (Teacher Distillation: protocol typing, model-owned value CE; Kaggriculture section: KL form, cache, value mean), `docs/kaggriculture-model.md` (a conformance row "Teacher: same / game form"), `docs/rl-api-specs.md` (target shapes), `README.md` (`teacher/cache_bytes`).

- [ ] **Step 1: failing tests**
  - **T12 bit-for-bit (I6).** CPU fp32, `_tiny` student and a different-seed teacher, segment-major `mixed` batch: `evaluate_actions_with_teacher` vs `compute_teacher_distillation_targets` + `evaluate_actions_with_cached_teacher`. Every field is `torch.equal`: `action_kl` `event`, `per_player_entity` and each component, teacher winner probabilities, student winner log-probs, `teacher_value_cross_entropy`, student log-probs, entropies and values.
  - **T13 teacher = student copy.** (`load_state_dict`): KL is exactly zero, and the value CE equals the live-seat mean of the student's winner entropy (`assert_close` 1e-6).
  - **T14 value CE reduction.** Hand tensors: two live seats give the mean of two CEs. One non-live seat is excluded. No live seat gives 0, with state weight 0 in `_value_state_weight`. Summing would fail the test (a guard against doubling).
  - **T15 admission.** Wrong target type → `TypeError` (and `CachedTeacherDistillationTargets` into Kaggriculture, and the reverse into Isaiah's model); wrong slot shape or missing key → `ValueError`; missing required target → `ValueError`; all before any kernel (a spy on `encode_observations` sees zero calls). The combined path rejects a non-Kaggriculture teacher, a different `action_spec` and different tables. **Grammar mismatch that replay cannot see (review P1-2):** a teacher built with HIRE removed from `market_kind` computes targets for a HIRE-free program that both grammars admit (no `GrammarReplayError`); the student's cached path then raises `ValueError` naming the grammar before any kernel. Non-vacuity: the same targets re-stamped with the student's signature (`dataclasses.replace`) pass admission and yield a `market_kind` KL different from a matched teacher's, so the signature is the only guard. A mutation removing the signature check must fail this test.
  - **T15b stateless teacher dispatch (review P1-1, runs now).** `ppo._model_evaluate_actions_with_cached_teacher` and `ppo._model_evaluate_actions_with_teacher`, called on a Kaggriculture student with `hidden_state=None` and a non-`None` `dones` tensor (as `_update_minibatch` passes), succeed and equal the direct model call; on Isaiah's `StatelessTransformerV1` fixture they equal the direct call with `dones` passed (Orbit unchanged).
  - **T16 Orbit regression.** Isaiah's teacher tests pass unchanged. `StatelessTransformerV1.teacher_value_cross_entropy` is `torch.equal` to the removed free function's formula on `test_ppo.py:2939`'s rollout. `test_loss.py:321` passes with its new call syntax.
  - **T17 seat isolation and statelessness.** Changing seat 1's observation leaves seat 0's targets and KL unchanged (`torch.equal`). Two calls give equal targets.
  - **T18 trainer** (skip until Task 3.1 trainer seam and 3.2 land). A fake 2-env Kaggriculture env, `teacher_mode` last-best with an active teacher:
    - the precompute runs once per iteration (spy) under `no_grad`, and minibatches use `targets.index`;
    - `teacher/kl` is finite and > 0 for a perturbed teacher; for a student copy it is exactly 0 **on the first minibatch** (checked by spying `_update_minibatch`'s loss metrics before any optimizer step, or with a single update step at zero learning rate), because later minibatches follow student updates;
    - `teacher/cache_bytes == 2 × 64 × 2 × 102,208` at horizon 64 (or the test's horizon);
    - `teacher/kl_coef` and `teacher/value_coef` are logged;
    - `set_teacher_model` rejects a hidden-state teacher through the existing path.
  - **T19a `_teacher_obs_spec_for_student` (runs now; a pure function).** The Kaggriculture equality path, a mismatch and the cross-game `TypeError`; Isaiah's two existing tests pass unchanged.
  - **T19b `run_ppo` last-best** (skip until `kg/rebuild-configs` is merged). After a forced promotion, the teacher's targets equal the student's (KL 0), and the tables survive the refresh (`torch.equal` to the grammar's). A resume restores the teacher from `checkpoint_last_best.pt`. A fresh launch from weights activates the teacher. The checkpoint key set contains no teacher cache.
- [ ] **Steps 2–4:** FAIL; implement §3.1–§3.2; the tests, the full suite, the Isaiah suites and `py-prepare` (with docs-fresh against the mapped docs above).
- [ ] **Step 5:** update the Phase 4 Reference. Revise the stream C Reference's "Trainer typing is still concrete" and "concat asymmetry" limits in place, linking the Phase 4 note (note, index and log together). Commit "Wire Kaggriculture teacher distillation through Isaiah's cached-teacher path (Phase 4.3)".

### Task 4.4 — configs (after `kg/rebuild-configs` merges into this line)

- [ ] **Step 1: failing tests** in `tests/kaggriculture/test_configs.py`:
  - explicit teacher-field equality with `scaling_6m` for both ranked configs;
  - the CPU config's teacher fields;
  - the headroom lines include the teacher cache bytes, pinned to 1,674,575,872 B / 837,287,936 B for the whole rollout and to the `teacher_chunk` rows × 102,208 B.
- [ ] **Steps 2–4:** FAIL; add the cache-bytes figure to `kaggriculture_workload.py` (a `teacher_cache_bytes` property from a model-module constant); the tests, the Isaiah suites and `py-prepare`.
- [ ] **Step 5:** revise the configs Reference ("Pending dependencies" loses the teacher item; Limits keeps "cache and chunk memory unmeasured on GPU until Task 6.1"), the index and the log. Commit "Pin Kaggriculture teacher settings and cache size to the scaling_6m recipe (Phase 4.4)".

## 6. Acceptance for Phase 4

- **Tests:** T1–T17, T15b, T19a and the 4.4 tests pass on CPU. **Phase 4 is complete only when T18 and T19b execute and pass** (review P2-4). During intermediate sub-tasks they are written and skipped with their named dependency, and each sub-task report lists them as open; a report with them skipped may close 4.1–4.3 as sub-tasks but must state that Phase 4 is not complete.
- **No regressions:** Isaiah's suites stay green; `py-prepare` passes; no Rust changes.
- **Refactors, not shims:** no free-function shims for the removed `_teacher_value_cross_entropy`, no `getattr`/`setattr` in first-class paths, and every narrowing is `isinstance` with an explicit error.
- **Cookbook:** one Phase 4 Reference, the revised stream C and configs References, index lines and prepended log entries.
- **Unverified until Phase 6:**
  - GPU memory: allocated and reserved per phase at dense BC positions, with the actual cache bytes;
  - BF16/compiled replay equality of teacher targets across the teacher chunk vs the minibatch;
  - teacher time and SPS in a complete iteration.

## 7. Open points for the reviewer

1. Model-owned value CE (§3.2) versus keeping `ppo._teacher_value_cross_entropy` and adding a reshape. The reshape alone gives a **sum** over seats (a doubled coefficient); the mean needs game knowledge, hence the method.
2. The single-chunk no-copy `concat` saves 1.56 GiB of transient. Is returning the same object acceptable under the protocol? Nothing mutates targets after the precompute.
3. Isaiah's `CachedTeacherDistillationTargets.concat` keeps its asymmetric rule; only the Kaggriculture type validates symmetrically. The alternative is to make both symmetric (it changes only an unreachable failure path, but touches Isaiah's behavior and stream C's pinned tests).
