---
type: "Decision"
title: "Diagnose the mechanism before spending on runs"
description: "Keep decisions reproducible, map game capabilities to executable mechanisms, and distinguish four failure classes before repairing."
tags: ["kaggriculture-v3", "decisions"]
status: "stable"
generated: {"by": "openai/codex", "at": "2026-09-28"}
decider: "Owner requests reuse of useful v2 decisions and discipline, then directs: just clone the repo in and start adapting; scoped implementation interpretation below."
sources: [{"resource": "user-directive:2026-09-28:kaggriculture-v3-reuse-and-adapt"}, {"resource": "repository:ops/cookbook-setup-checks.md"}, {"resource": "external-repository:/Users/poonszesen/kaggriculture-v2/cookbook/decisions/kaggriculture-v2-starts-with-a-clean-design.md"}, {"resource": "external-repository:/Users/poonszesen/kaggriculture-v2/cookbook/decisions/the-architecture-must-express-the-decisions-needed-to-win.md"}, {"resource": "external-repository:/Users/poonszesen/kaggriculture-v2/cookbook/decisions/failures-must-distinguish-information-decision-execution-and-architecture-errors.md"}, {"resource": "external-repository:/Users/poonszesen/kaggriculture-v2/cookbook/decisions/stalled-improvement-requires-attribution-and-explicit-architecture-limits.md"}]
---

# Diagnose the mechanism before spending on runs

The owner's reuse instruction adopts useful v2 discipline for this adaptation. Its original owner directives include:

> crafting a system that is undebuggable and reuqesting blind guess/ experiment is banned explicitly in this project.

> 證據是你有信心的時候才去收集的，不是沒有信心下的賭博.

> 贏所需的決策，架構能否表達？

V3 operationalizes those requirements as follows. Before a costly run, state the observed problem, code/data path, supported mechanism or diagnostic question, expected discriminating observation, and stopping condition. Preserve source/configuration, inputs, seeds and outputs sufficient to replay and localize it. Inspect implementation or add focused diagnostics when visibility is missing. A diagnostic may resolve an unknown cause; mark uncertainty rather than inventing confidence.

Before relying on an architecture, map investment payback, labor allocation, sale timing and market adaptation to available information, represented state, a choice mechanism and executable actions. Trace a worked case through those boundaries. Explicit capacity is necessary, but does not prove effective learned use.

Distinguish **information insufficiency**, **decision error**, **execution error**, and **architecture error**, following the original owner quote “資訊不足、決策錯誤與執行錯誤，或者是架構錯誤”. Trace observed inputs → representation → intended choice → emitted action → engine outcome. Information available but discarded by representation is not an environmental information shortage. Multiple causes may coexist; unresolved attribution must be named.

When repeated repairs fail, inspect the engine, relevant original research and applicable competition discussion before spending again. A justified repair needs a mechanism and discriminating check. Record `ARCHITECTURE LIMITATION` only with its evidence, conditions, attempted repairs and reopening condition; qualify it as suspected when attribution remains unresolved. Repeated poor scores alone never prove impossibility. These rules add no approval step.
