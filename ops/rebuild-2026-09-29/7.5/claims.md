# Task 7.5 claim ledger

Snapshot: `bde337465a9fa7c07bedded88d5d696d7cefb7ef`, 2026-09-29. Documentation only.
`R/` expands to `ops/rebuild-2026-09-29/`. All file:line references are at this
snapshot unless they name new `7.5/` receipts. `BC` references mean `git show
933d661:<path>` on `kg/rebuild-bc-now`: the preparer is
`scripts/kaggriculture_prepare_bc.py`, and `pairing.json` / `receipts.md` are under
`ops/rebuild-2026-09-29/bc-a100-2026-09-29/`. No other unmerged implementation
or result is used. Later branch status is not substituted for this tip's tracker.

Each row is an independently reviewable factual claim or a named evidence limit.
Table file names, test entry points and receipts are accounted for with their
layer claims. Receipt paths are evidence pointers, not fresh execution claims.
No historical count is presented as a completed current root/Python suite.

| ID | Claim in the summary or minimal correction | Exact support / scope |
| --- | --- | --- |
| S01 | Snapshot is integration HEAD bde337465a9fa7c07bedded88d5d696d7cefb7ef on 2026-09-29. | R/7.5/source-audit.txt:1–2; R/7.5/engine-tests.log:2–3. |
| S02 | Detailed sections retain historical results; current checks are separate. | git diff -- docs/rules-parity-coverage.md; historical count paragraphs retain their text (see corrections C01–C06). |
| S03 | Dependency target is kaggle-environments==1.32.7. | pyproject.toml:11; uv.lock:1774–1775,3045. |
| S04 | Cargo metadata pins the full interpreter SHA-256 printed in the summary. | engine_rs/Cargo.toml:8–11. |
| S05 | Vendored reference is full commit 65f0eac5bb00b18a9d3acce319c2a231cbd5dff0. | scripts/check_engine_trim.py:21; engine_rs/TRIM_MANIFEST.json:3 (reference_commit); R/7.5/trim-after-setup.log:4. |
| S06 | Named installed-pin test checks version/hash through load_pinned_kaggle. | tests/scripts/test_kaggriculture_parity.py:89–92; scripts/kaggriculture_parity/generate_traces.py:120–145. |
| S07 | Generator rejects another version/hash before trace output. | scripts/kaggriculture_parity/generate_traces.py:104–117,120–145; tests/scripts/test_kaggriculture_parity.py:58–86 (including test_generator_refuses_before_writing_when_the_pin_differs). |
| S08 | Parity test file checks live byte-identical committed regeneration (may skip offline). | tests/scripts/test_kaggriculture_parity.py:349–394. |
| S09 | Trim checker checks custody, manifest inventory/hashes/Cargo-pin consistency, not installed interpreter. | scripts/check_engine_trim.py:291–378,389–440; tests/tools/test_check_engine_trim.py:583–674. It consumes Cargo metadata and fixture bytes; it does not load the installed Kaggle package. |
| S10 | Receipt prefix R/ expands to ops/rebuild-2026-09-29/; counts from different layers overlap. | Notation. Overlap: grammar_kernel_tests.rs:571–630 uses the same four episode IDs as engine_rs/tests/replay_parity.rs:600–614; observation manifest:1 source_counts includes official rows. |
| K01 | Four official trace IDs and 719 transitions each. | engine_rs/tests/replay_parity.rs:585–614; fresh header/row aggregation R/7.5/source-audit.txt:3–6. |
| K02 | Official denominator 2,876 transitions and 2,880 snapshots. | 4×719 and 4×(719+1), from K01; initial snapshot comparison engine_rs/tests/replay_parity.rs:488–491 and successors :531–560. |
| K03 | Official oracle is recorded Python-engine traces; Rust resets from config and seed only. | Trace header source module_version/engine_sha256 in R/7.5/source-audit.txt:3–6; engine_rs/tests/replay_parity.rs:466–491. Test entry points :600–614; original receipt R/1.1/results.md:1–41. |
| K04 | Replay compares public/private values and recursive key order. | engine_rs/tests/replay_parity.rs:35–94,183–203,488–491,532–535. |
| K05 | Replay compares statuses, typed rewards, step/done and terminal banks. | engine_rs/tests/replay_parity.rs:536–576. |
| K06 | Rejected actions must leave checked state unchanged. | engine_rs/tests/replay_parity.rs:413–454,505–522; rollback regression tests :278–327. |
| D01 | Live differential oracle is Kaggle's own pinned Python engine. | scripts/kaggriculture_parity/generate_traces.py:120–145,744–841; R/1.1b/results.md:8–10,19–24. |
| D02 | Eight committed games, 3,960 transitions. | engine_rs/fixtures/generated/MANIFEST.json (all traces without expected_divergence); aggregation R/7.5/source-audit.txt:7; replay validation engine_rs/tests/replay_parity.rs:638–657. |
| D03 | Seven committed one-step divergence repros. | Generated manifest entries with expected_divergence; R/7.5/source-audit.txt:8; tests/scripts/test_kaggriculture_parity.py:170–187. |
| D04 | Recorded local sweep: 40 games, 21,824 agreeing transitions. | R/1.1b/verify-r1/sweep-summary.json:26–38 and per_trace records; R/1.1b/results.md:125–131; R/7.5/source-audit.txt:9. The recorded commit is 6217868, not current HEAD. |
| D05 | 303 probes, 1,515 transitions, 266 agreements, 37 D1/D2 divergences. | R/1.1b/verify-r1/sweep-summary.json:26–38; 23,339 total minus 21,824 game transitions = 1,515; 306 agreeing traces minus 40 games = 266; 25 D2 + 12 D1 = 37. R/1.1b/results.md:125–131. |
| D06 | Named generated and external-directory replay entry points exist. | engine_rs/tests/replay_parity.rs:638,793–832. Directory test returns early with no KAGG_PARITY_TRACES (:794–797); passing engine totals do not imply a new external sweep. |
| D07 | Task 7.5 uses the historical sweep receipt; no new sweep is launched. | R/7.5/results.md, command inventory. Required pytest includes bounded committed-fixture regeneration, but no sweep.py invocation; it stopped without a final summary. |
| U01 | 41 retained library tests and nine RNG tests. | R/7.5/engine-tests.log:7–65, including result lines 50 and 65. |
| U02 | Library tests use synthetic hand-derived expectations, no episode reads. | engine_rs/src/lib.rs:1717–2787 test module: fresh_header :1720, literal assertions including :1957–1980 and :2367–2545; no filesystem/episode loader in the module. |
| U03 | RNG integration uses embedded CPython vectors, no episode reads. | engine_rs/tests/py_random.rs:1–32 (generator and Python identity), vectors :33 onward, test functions near file end. Test entry point is that file; current nine-test receipt R/7.5/engine-tests.log:52–65. |
| G01 | Grammar expectations use three historical reference decoders, not the new grammar. | R/1.2/oracle/README.md:3,24; src/kaggriculture/grammar_tests.rs:1064–1071; R/1.2/results.md:49–51,99–100. |
| G02 | 320 scheduled accepted programs = 256 synthetic + 64 replay. | tests/fixtures/kaggriculture/grammar-v4-reference.manifest.json:56–64; src/kaggriculture/grammar_tests.rs:1077–1148. |
| G03 | 44 controls = 43 rejections + one padding acceptance. | Same manifest:56–64; grammar_tests.rs:1092–1104,1115–1148; R/1.2/oracle/README.md:28. |
| G04 | All 964 table bits checked against literal supports and reference-plan bits. | src/kaggriculture/grammar_tests.rs:627–644; R/1.2/results.md:52–57. |
| G05 | Root kernel tests compare decoded execution with direct execution. | src/kaggriculture/grammar_kernel_tests.rs:144–200,521–567; current module registration src/kaggriculture/mod.rs:20–24. |
| G06 | 64 selected replay seat actions checked in actual states. | src/kaggriculture/grammar_kernel_tests.rs:571–630; four episodes × 16 selected rows. |
| G07 | Grammar test files and listed receipts describe these layers; engine bridge retired. | src/kaggriculture/grammar_tests.rs:1064–1148; grammar_kernel_tests.rs:1–5; R/1.2/results.md:49–100; R/1.3/r1-fixes/results.md:20–35. |
| O01 | Observation oracle is reference Rust encode_invest output. | scripts/kaggriculture_observation_oracle/record.rs:1–3,146–176; regenerate.py:31–35 pins reference commit and feature-source hash. |
| O02 | 512 states / 1,024 seat rows / 8,176 offsets per row. | tests/fixtures/kaggriculture/observation-v3/manifest.json:1, reference_shape=[512,2,8176]; src/kaggriculture/oracle_corpus.rs:2211–2257; R/1.3/claude-review.md:36–53. |
| O03 | Comparison is tensor-only reconstruction, bitwise. | src/kaggriculture/oracle_corpus.rs:1477 (ObsRowMut input),2252–2253; bit comparator :1426–1441; test compare_observation_oracle :2274. |
| O04 | Python actual observation schema checked across frozen corpus. | tests/kaggriculture/test_observe.py:495–524 streams all 512 records, helper check_contract at :210; R/1.3/claude-review.md:74–75 records the focused observation/custody suites passing. |
| O05 | Observation-information coverage does not add rules parity. | Inference bounded by recorder record.rs:146–176 and oracle_corpus.rs:2230–2253: encode/reconstruct, no transition comparison to Kaggle Python. |
| N01 | Native lifecycle tests use synthetic expectations and untouched controls. | src/kaggriculture/env_tests.rs:438–522,602–653; reset/step/truncate rollback cases compare control execution; no independent Python-engine step in these tests. |
| N02 | Native admission and lifecycle use 35 destination buffers. | tests/kaggriculture/test_native_env.py:86–133,311–349,685–717; src/kaggriculture/admission.rs:59–95. Entry points named in table are these files and env_tests.rs. |
| N03 | Native recorded oracle is historical Rust TrainingBatch, not Kaggle Python. | scripts/record_kaggriculture_env_reference.py:36–47,94–103,523–558; R/1.4/reference_recorder.rs:1–2,59–73. |
| N04 | Native fixture .json manifest and .npz archive exist with pinned reference commit. | tests/fixtures/kaggriculture_env_reference_v1.json:1 (sources.reference and fixture hashes); recorder default paths scripts/record_kaggriculture_env_reference.py:34–35; loader tests/kaggriculture/test_env_reference.py:121–124. |
| N05 | 16 games, seeds 17000–17015, 719 transitions each = 11,504. | Manifest :1; test_env_reference.py:124–141,255; R/1.4/claude-review/pod-oracle/record.log:29–45; latest recording receipt R/1.4/p3-rerecord/reference-recording-attempt.json:1. |
| N06 | Bitwise rewards, dones, before/after banks and economic counters compared. | tests/kaggriculture/test_env_reference.py:95–102,145–163. |
| N07 | Terminal records, metrics, seeds and autoreset checked. | tests/kaggriculture/test_env_reference.py:191–252. |
| N08 | Separate independent mathematical Python reward formula allows one f32 ULP. | tests/kaggriculture/test_env_reference.py:169–190; R/1.4/claude-review/pod-oracle/replay-green.log:2–3. |
| N09 | Native fixture does not compare every structured tensor or whole snapshot to Python rules. | test_env_reference.py:121–255 field inventory; N03 oracle identity. It checks reset clock/still_playing/snapshot step but not complete observation equality against Python. |
| A01 | Task 1.5 uses real native binding, with lifecycle/rollback expectations. | tests/kaggriculture/test_env.py:480–517,520–649; R/stage2-adapter/native/results.md:14–21,32–34. |
| A02 | Adapter keeps 35 buffer/view identities stable. | tests/kaggriculture/test_env.py:480–517,626–649; R/stage2-adapter/native/results.md:14–17. |
| A03 | Codec has 321 accepted and 43 rejected corpus records. | tests/kaggriculture/test_codec.py:323–358; native corpus totals asserted in test_native_grammar_bindings.py:211–270; R/stage2-adapter/native/results.md:28–30. |
| A04 | CPU bridge compares native tables to independently expected tables, 964 bits. | tests/kaggriculture/test_native_tables.py:17–25; test_native_grammar_bindings.py:50–54; R/stage2-adapter/tables/results.md:27–38. |
| A05 | Reward tests use recorded trajectories plus three extreme-coefficient 96-transition games. | tests/kaggriculture/test_rewards.py:327–419; R/stage2-adapter/native/results.md:60–71. |
| A06 | Factory real tests exist in test_game.py; all named adapter entry points/receipts are real. | tests/kaggriculture/test_game.py:360–387; R/stage2-adapter/native/results.md:22–30,54–55; table receipts R/stage2-adapter/tables/results.md:27–38. |
| B01 | BC evidence belongs to unmerged kg/rebuild-bc-now at full 933d661 commit. | R/7.5/source-audit.txt:13; R/7.5/bc-audit.txt:2–6. Branch tip advanced; pairing JSON and preparer are byte-identical at the inspected tip. |
| B02 | BC evidence paths are pairing.json, receipts.md and scripts/kaggriculture_prepare_bc.py at that commit. | git show 933d661:ops/rebuild-2026-09-29/bc-a100-2026-09-29/pairing.json; same prefix receipts.md:28–34; git show 933d661:scripts/kaggriculture_prepare_bc.py:92,554–621. |
| B03 | BC receipt says pairing result was copied, not rerun. | BC receipts.md:28–34 at 933d661. Do not adopt its unsupported day-end or unaffected-label claims. |
| B04 | BC uses Kaggle 1.32.7 interpreter. | BC pairing.json:2–5; BC preparer:561–565; these are branch evidence, not a fresh interpreter execution. |
| B05 | BC re-steps archived steps[t] observations with steps[t+1] actions and episode configuration. | BC preparer:567–596, including recorded seed and resetting each transition from archived observations. |
| B06 | BC compares day/hour/farms/market/town and both private values. | BC preparer:92 (PAIRING_PUBLIC_KEYS),597–608. |
| B07 | BC excludes step, key order, statuses/rewards from comparison. | BC preparer:91–92,592–608; dict value equality at :608 ignores object insertion order. |
| B08 | BC eight sampled episodes, 5,743/5,752 transitions match. | BC pairing.json:6–124; independent aggregation R/7.5/bc-audit.txt:7; sample selection BC preparer:650–655. |
| B09 | Nine mismatches, private-only, at the listed six distinct turns. | BC pairing.json:19–25,40–52,60–84,99–105,113–119; R/7.5/bc-audit.txt:7. |
| B10 | Mismatch cause remains unattributed; no proof of label correctness. | BC pairing.json reports differing fields only; preparer:554–621 compares states only. No causal diagnostic or label check in that function. This is a stated evidence limit, not a causal claim. |
| B11 | Sampled configurations unavailable in inspected tracked evidence; day ends not asserted. | R/7.5/bc-audit.txt:8–13 records exact git grep command and missing receipt archive locations. Only pairing.json matched sampled IDs. Arithmetic modulo 24 alone was not used. |
| B12 | No Rust engine participates; BC pairing is not engine parity. | BC preparer:554–621 imports/calls Kaggle interpreter only. This does not refer to other parts of the BC pipeline. |
| L01 | D1/D2 remain known malformed-input disagreements, not repairs. | engine_rs/tests/replay_parity.rs:629–636,659–684; generated manifest expected_divergence records; R/1.1b/results.md:70–87. |
| L02 | Bounded probes/policies do not exhaust Python paths or malformed actions; grammar cannot emit D1/D2. | R/1.1b/results.md:70–87,138–147; src/kaggriculture/grammar.rs literal command rendering and checked integer codec; tests grammar_tests.rs:1092–1104 reject malformed controls. Policy domain is ASCII names and integer quantities, not arbitrary external JSON. |
| L03 | Framework timeouts, agent errors, INVALID behavior not modeled. | R/1.1b/results.md:140–145; comparator engine_rs/tests/replay_parity.rs:457–578 executes Game directly, not Kaggle agent runner. |
| L04 | No strong-play worlds beyond four official episodes or larger pod sweep credited. | R/1.1b/results.md:138–145; committed manifest policies; source-audit.txt:3–9 and recorded sweep. Scope is evidence available at this tip, not an assertion about all external runs. |
| L05 | RNG/shop schedule headers not compared directly. | engine_rs/tests/replay_parity.rs:457–578 uses config/seed constructor, initial and resulting states, not schedule equality. Original R/1.1/results.md identifies replay scope. |
| L06 | No full-season codec parity on official action streams or exhaustive actor/order/HIRE product. | src/kaggriculture/grammar_kernel_tests.rs:571–630 selects only 64 actions; R/1.2/results.md:52–58 qualifies support classes. The Task 1.4 complete games are a different fixed policy and denominator. |
| L07 | Task 1.4 fixture does not establish full Python observation/snapshot parity. | N03 and N09; tests/kaggriculture/test_env_reference.py:121–255. |
| L08 | Historical observation custody omits three engine hashes, generation dirty state and full producer inventory. | R/merge-1.3/verify-r1-fix/results.md:35–42 names engine Cargo.toml, py_random.rs, econ_attrib.rs and root-module limits. Current recorder repair does not retroactively fill these. |
| L09 | Task 7.1 opponents/oracle approved on unmerged branch, not current integration coverage. | R/phase-status.md:118; R/7.5/source-audit.txt:10 (ancestry exit 1),14 (current opponents file absent). No implementation/results from that branch imported. |
| L10 | Task 7.3 replay export / Kaggle episode round trip not merged; tip's tracker contains no approval. | R/phase-status.md:120 explicitly says no approving verdict yet; R/7.5/source-audit.txt:11,15. Approval wording is tied to this tip's dated tracker, not a claim about later branch work. |
| L11 | Task 7.4 packaging at this tip is brief review, no implementation. | R/phase-status.md:121; R/7.5/source-audit.txt:12. Existing starter packaging is not Kaggriculture qualification. |
| L12 | Task 3.1 rollout/mask/action mapping and trainer-level/learning qualification remain gaps. | scripts/run_ppo.py:178–188,1440–1448; tests/kaggriculture/test_teacher.py:870–877,1356,1625,1654,1690 skips; R/stage2-adapter/final-report.txt:85–92,189–190. scripts/run_ppo.py remains the canonical trainer; no alternate trainer or v2 model introduced. |
| L13 | Native-adapter CUDA/BF16, hardware table upload, pinned DMA reuse fence unqualified here. | tests/kaggriculture/test_env_cuda_fence.py:11–41; test_observe.py:279–296; R/stage2-adapter/tables/results.md:42–46; R/stage2-adapter/final-report.txt:93–96,149–151. CPU oracle equality does not qualify those device paths. |
| L14 | Separate GPU model diagnostics exist but are not native-adapter parity evidence. | R/results.md:305–426, explicitly GPU checks bundle (component); no complete native-adapter parity claim inferred. |
| L15 | Complete-update throughput unqualified; Task 1.3 dedicated diagnostic incomplete despite later Task 1.4 observation/lifecycle component costs. | R/1.3/timing.json:15–31; R/1.4/claude-review/pod-oracle/timing-enabled.json:1 explicitly scopes component costs and records prepare_snapshot_acquisition, validate_existing_snapshot, write_prepared_both_seats; src/kaggriculture/lifecycle_timing_tests.rs:86–114; R/stage2-adapter/native/results.md:66–72. |
| V01 | Current commands were run on 2026-09-29 at the full stated HEAD. | R/7.5/{engine-tests,root-tests,root-tests-after-setup,trim-after-setup,pytest,docs-fresh}.log:1–3; their .json sidecars. |
| V02 | Offline/thread environment, resource limits and development setup. | R/7.5/results.md records exact shell exports; dev-dependencies.log:1–6,158 and dev-build.log:1–3,58–64; JSON sidecars record 120s/960MiB guards. Subsequent uv checks used UV_NO_SYNC=true. |
| V03 | Engine suite 69 passed, zero failed/ignored, command exit 0. | R/7.5/engine-tests.log:50,65,90,96,99: 41+9+19, zero doc tests. |
| V04 | Initial root test failed missing local Python, exit 101. | R/7.5/root-tests.log:22–26. |
| V05 | Root retry stopped by SIGTERM at RSS guard, status -15, no final counts. | R/7.5/root-tests-after-setup.log:78–80; .json stop_reason. Do not convert emitted partial test lines into a completed suite total. |
| V06 | Trim checker exit 0 after development setup. | R/7.5/trim-after-setup.log:4–6. Initial auto-build attempt has incomplete monitor custody, recorded in results.md. |
| V07 | Requested pytest stopped by SIGTERM, shell exit 143, no final counts. | R/7.5/pytest.log:5–7; .json sidecar. Dots are not promoted to a suite result. |
| V08 | Doc freshness exit 0, No doc updates required. | R/7.5/docs-fresh-final.log:4–6; script mapping scripts/check_doc_freshness.py:12–58. No code/doc mapping edit requested. |
| C01 | Opening paragraph gains summary pointer; Orbit sections unchanged. | docs/rules-parity-coverage.md:3–7; git diff; final structural audit in R/7.5/structure-check.txt. |
| C02 | Existing grammar future tense changed to completed tense. | Original line 200 -> current line 324: 'Grammar coverage returns with Task 1.2' -> 'Grammar coverage returned with Task 1.2'. G01–G07 support completion. |
| C03 | Existing temporary engine grammar bridge changed to already-retired root route. | Original lines 390–395 -> current 514–518: engine-side include/future Tasks 1.3/1.4 retirement -> Task 1.3 retired it; root grammar_kernel_tests.rs. Sources G07; Cargo.toml root path dependency; R/1.3/r1-fixes/results.md:20–35. |
| C04 | Existing nonexistent Task 1.5 end-of-page pointer points to summary row. | Original line 480 -> current 603: 'Task 1.5 note at the end of this page' -> '[Task 1.5 summary row](#what-is-tested)'. Original HEAD file ends in Task 1.4; new row lines 176–177. |
| C05 | Existing ten-case admission wording includes merged strengthening case. | Original lines 561–562 -> current 686–688: 'exact ten-case binary64 admission predicate' -> 'ten-case binary64 admission predicate plus Task 1.5’s strengthening case'. src/kaggriculture/env_tests.rs:63–86 has 11 entries; tests/kaggriculture/test_rewards.py:36–52,110–138. |
| C06 | Existing Task 1.5 allocation wording distinguishes merged CPU work from pending CUDA proof. | Original lines 657–659 -> current 783–787: adapter/rewards/codec/table bridge and fence 'belong to Task 1.5' -> CPU implementation merged, pinned CUDA reuse-fence qualification open. A01–A06, L13. |
| C07 | Scope the old observation phase-cost gap to the stopped Task 1.3 diagnostic and credit Task 1.4 components. | Original docs/rules-parity-coverage.md:533–535 -> current :656–661: 'optimized timing build stops ... phase costs remain unmeasured' -> Task 1.3 build stopped; diagnostic incomplete; later Task 1.4 snapshot/validation/write costs measured. L15 supplies actual receipt/code support. |
| C08 | Existing replay constructor name updated to the current comparator. | Original docs/rules-parity-coverage.md:182 -> current :306: Game::new -> Game::new_with_seed_decimal. engine_rs/tests/replay_parity.rs:483–487 calls the decimal-seed constructor with configuration and seed only. |
