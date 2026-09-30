**Verdict: APPROVE WITH EDITS.** No valid-input Orbit regression found. Full-suite verification remains blocked by the read-only sandbox.

1. **Should-fix — teacher contract overstates validation.** [teacher_targets.py:16](/Users/poonszesen/kg-v3/.claude/worktrees/agent-a6b402c3efeaea928/python/owl/model/teacher_targets.py:16) and [model-architecture.md:703](/Users/poonszesen/kg-v3/.claude/worktrees/agent-a6b402c3efeaea928/docs/model-architecture.md:703) promise rejection whenever optional targets disagree. Actual concatenation silently drops populated later targets when the first chunk lacks them. I reproduced this for action parameters, winner probabilities and continuation logits. This behavior is inherited; the refactor preserves it. **Fix:** narrow the documentation and test both chunk orders, or deliberately introduce symmetric validation and record the behavior change.

2. **Note — the alarm threshold has different meaning across clipping modes.** [ppo.py:1106](/Users/poonszesen/kg-v3/.claude/worktrees/agent-a6b402c3efeaea928/python/owl/train/ppo.py:1106) consumes the existing loss metric: `per_player` sums entity log-probabilities; `per_entity` averages their log-ratios. A synthetic probe with 241 entities and `0.0003` drift per entity produces **0.07233 versus 0.00030 nats**. Thus small coherent numerical differences could trigger the joint-action alarm; legitimate BF16/compile false positives have neither been demonstrated nor excluded. **Fix:** document these units, add both-mode/multiple-entity tests, and measure representative noise during the planned GPU qualification before asserting that 0.05 is safely above it.

3. **Note — test claims need narrower wording and stronger assertions.** [test_ppo_observation_mapping.py:3](/Users/poonszesen/kg-v3/.claude/worktrees/agent-a6b402c3efeaea928/tests/owl/train/test_ppo_observation_mapping.py:3) calls its oracle “frozen copies,” but it is a successful-path extraction whose field list uses the current schema and whose copy helper omits original validation. The substantive comparisons are sound. Separately, the invalid-limit tests at [test_ppo.py:1992](/Users/poonszesen/kg-v3/.claude/worktrees/agent-a6b402c3efeaea928/tests/owl/train/test_ppo.py:1992) also passed before implementation because unknown-field rejection matched their assertions. **Fix:** describe the oracle accurately or freeze it literally; assert specific Pydantic validation error types and acceptance of a custom positive limit. Add an unequal-rank-weight reduction test; current alarm tests exercise one process.

4. **Note — one confirmed merge conflict.** With stream C pinned to `87c71d7` and `kg/rebuild-model` to `23da55f`, read-only merge simulation conflicts at [cookbook/log.md:3](/Users/poonszesen/kg-v3/.claude/worktrees/agent-a6b402c3efeaea928/cookbook/log.md:3). **Fix:** retain both branches’ entries. `docs/model-architecture.md` and `cookbook/references/index.md` merge textually cleanly. Stream C leaves `base.py` and `optimizer.py` untouched. Generic teacher typing remains later integration work: trainer/model signatures still name the concrete Orbit targets.

The remaining checks support approval:

- All refactored mapping helpers preserve successful Orbit behavior, including CPU cloning, accelerator no-clone dispatch, optional tensors and all three mask types. Tests check values, destination allocation and aliasing meaningfully. Failure ordering, error wording and partial-copy state can differ; those paths are not strictly unchanged.
- The alarm defaults on at `0.05`, runs after the first backward and **before every possible optimizer step**, including accumulation. Its numerator and denominator are globally reduced, so identically configured ranks make the same decision. Signed cancellation across samples or ranks remains a detection limitation.
- Teacher slicing/concatenation bodies preserve behavior. No stale free-function imports or calls remain in `python`, `tests` or `scripts`.

Command results:

| Command | Result |
|---|---|
| Requested `uv run pytest tests/owl tests/scripts tests/tools -m "not slow" -q` | Exit 2: uv cache access denied; tests never started |
| Requested `uv run mypy python/owl scripts` | Exit 2: same cache restriction |
| Existing-venv pytest fallback, bytecode/cache disabled and capture set to `sys` | **12 collection errors; 0 tests executed.** PyTorch cannot create its temporary directory |
| Existing-venv mypy fallback with `--no-incremental --cache-dir=/dev/null` | **Success: 48 source files, no issues** |
| New teacher-target tests through existing venv | **25 passed** |
| Branch `git diff --check` | Exit 2: trailing whitespace in recorded red-test logs |

The committed receipt reports **924 passed, 3 skipped**; I could not independently reproduce that full-suite count under this sandbox. Source files remain unchanged.