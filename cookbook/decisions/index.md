# Decisions

- [[v3-reuses-isaiah-infrastructure-with-kaggriculture-semantics|V3 reuses Isaiah infrastructure with Kaggriculture semantics]] — Retain the starter training path and useful model infrastructure while adapting the engine, observations and actions to Kaggriculture.
- [[diagnose-the-mechanism-before-spending-on-runs|Diagnose the mechanism before spending on runs]] — Keep decisions reproducible, map game capabilities to executable mechanisms, and distinguish four failure classes before repairing.
- [[the-policy-is-stateless-and-observation-only|The policy is stateless and observation-only]] — Carry forward explicit current-observation tokens and the prohibition on opponent identity conditioning; old checkpoint strength does not transfer.
- [[throughput-means-correct-complete-work|Throughput means correct complete work]] — High throughput is required; preserve semantic parity and measure equivalent work rather than inferring gains from language or kernels.
- [[evaluation-preserves-generality-and-evidence|Evaluation preserves generality and evidence]] — Measure win rate and bank margin across opponents, retain source-bound evidence and telemetry, and keep submission authority with the owner.
- [[start-multi-gpu-qualification-with-four-ranks|Start multi-GPU qualification with two ranks]] — Start SPS qualification with two RTX 5090/RTX PRO 6000 GPUs; the earlier four-rank preference is superseded and GPU execution remains unverified.
