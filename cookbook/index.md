# Kaggriculture v3

The repository's cookbook records reusable decisions and knowledge across bounded **changes**. Isaiah training infrastructure is being adapted to Kaggriculture; source-v2 evidence is historical and does not qualify the port.

# Start here

- [[decisions/restart-the-port-from-isaiahs-clean-base|Restart from Isaiah's clean base]] (References are not all reference-branch notes: the References index's sections and entries state whether each describes the current tree or branch `kg/reference-2026-09-29`)
- [[decisions/v3-reuses-isaiah-infrastructure-with-kaggriculture-semantics|Adaptation scope and one trainer]]
- [[decisions/diagnose-the-mechanism-before-spending-on-runs|Reasoning, capability mapping and failure attribution]]
- [[decisions/the-policy-is-stateless-and-observation-only|Stateless, observation-only model constraints]]
- [[decisions/recipe-choices-align-to-isaiah-without-owner-escalation|Recipe choices align to Isaiah]]
- [[decisions/throughput-means-correct-complete-work|Throughput and correctness]]
- [[decisions/evaluation-preserves-generality-and-evidence|Evaluation, telemetry and artifact custody]]
- [[workflows/profile-cuda-bottlenecks-with-nsight-systems|NVIDIA profiling contract]]

# Browse

- [[references/index|Adaptation contracts and verification boundaries]], grouped by rebuild phase
- Phase tracker (working artifact, task state per phase): `ops/rebuild-2026-09-29/phase-status.md`

- [[decisions/index|Decisions]]
- [[workflows/index|Workflows]]
- [[log|Change log]]
- Derived view: `cookbook/kaggriculture-v3.base` in the vault, linked to the repository's sibling Base.
- Setup checks and source fingerprints: `ops/cookbook-setup-checks.md`.

No v3 result board, comparison register or empirical lesson exists yet. Add a single standing board only after two results can be ranked; preserve outcome denominators and historical scope.
