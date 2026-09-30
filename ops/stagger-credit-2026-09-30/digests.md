# Default-off byte identity: rl.initial_stagger

Branch `kg/rebuild-stagger-credit`, cut from `kg/isaiah-gap-closure` `3e89425`.
Environment: torch 2.9.0, macOS 26.4 arm64 (Apple M5), Python 3.12.13,
`OMP_NUM_THREADS=2`, debug `maturin develop` build.

| Check | Base `3e89425` | After the change | Result |
| --- | --- | --- | --- |
| Native env digest (`baseline_digest.py`, truncation included) | `257eae38864aa2aa26373c7751be97b3b0d3c6d9d81ee80782036df41159a590` | same | equal |
| Trainer digest (2 updates, metrics + weights) | `3ffd53a026b8b3be079c64b111226a6cb313dbdad586c81fbd28c974bd322fa2` | same | equal |
| `config_sha256` of every `configs/*.yaml` (`config_digest.py`) | `config-digest-3e89425.json` | `config-digest-post.json` (the two new presets excluded) | equal |

`baseline_digest.py` is the opponent-mix receipt's script
(`ops/opponent-mix-2026-09-30/baseline_digest.py`), copied unchanged. The
trainer digest is CPU/BLAS/thread-count dependent. It matches the opponent-mix
receipt's `OMP_NUM_THREADS=2` value (`ops/opponent-mix/digest-environment.log`).
No Rust source changed, so the native digest checks only that the extension
still builds the same behavior.
