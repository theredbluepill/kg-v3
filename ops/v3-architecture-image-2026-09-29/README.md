# V3 architecture illustration — 2026-09-29

Requested artifact: one image explaining the Kaggriculture v3 model and training pipeline. Completion condition: a readable image whose implemented components and target integration agree with the inspected checkout. No training or benchmark was requested or run.

- Snapshot: `kg/rebuild-model` at `8093d51`.
- Output: [v3-agent-architecture.png](v3-agent-architecture.png), 1586 × 992 pixels.
- Generation: built-in ImageGen, one generation; [exact prompt](prompt.txt). The tool did not expose an underlying model version.
- PNG SHA-256: `6f789aad52272e830cc9df14b77af55e3bf37fca3e5d7fee9a7dadfa2a4a9ad3`.

The diagram was checked against `docs/kaggriculture-model.md`, `docs/kaggriculture-contract.md`, `configs/model/kaggriculture.yaml`, `python/owl/model/kaggriculture.py`, and `ops/rebuild-2026-09-29/plan.md`. A separate read-only agent checked the factory, trainer and environment boundary. Concept searches covered architecture, pipeline, statelessness, pending integration and existing diagram records before attaching this artifact to the existing encoder Reference.

Visual inspection of the generated image checked the title, token ordering, stem dimensions, trunk settings, actor/critic split, value formula, arrows, stateless constraints and status legend. Solid teal denotes the implemented observation schema, encoder and critic; dashed amber denotes the actor and target Kaggriculture training integration. The observation card does not assert that a native environment writer is present. `forward` and `evaluate_actions` still raise for the absent actor in this checkout; code on other work branches is outside this snapshot.

The pipeline is a schematic: grammar support masks do not guarantee affordability or successful execution; exact action admission and reward/bootstrap rules remain in the contract. Public replays → BC is compressed and does not depict reconstruction/admission or held-out-NLL selection. The critic uses both value tokens from a single seat's legal view, not cross-seat attention. The full-model 6–10M count is a target. No GPU, complete-work throughput, learning quality or end-to-end qualification is established by the image.

No Python/Rust implementation changed, so code suites were not rerun. Before reusing the image as a current-status artifact, check its pinned checkout; regenerate after actor/factory/environment/trainer integration lands. Cookbook custody: `cookbook/references/kaggriculture-encoder-reuses-isaiah-stateless-layers.md`.
