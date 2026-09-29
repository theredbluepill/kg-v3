# Independent verification of fb65e1f

Verdict: APPROVE WITH EDITS. Low-severity documentation correction only; no implementation/test blocker found. No tracked files modified.

## Finding

P3: cookbook/decisions/restart-the-port-from-isaiahs-clean-base.md:57 says “other 50 lockfile changes”. Exact inventory is 57 changed package names: kaggle-environments, owl metadata, and 55 other dependency names. The latter consist of 49 changed version sets, 3 dependency-only records, 1 added package and 2 removed packages. Some dependencies are shared: click also belongs to W&B. Correct the count and avoid implying isolation from training telemetry.

## Commands and evidence

- Checkout: kg/rebuild-deps at fb65e1fad5e0b772623eaf7e660cdc62cdfda02e.
- uv 0.10.9; uv lock --check exited 0 (181 resolved packages).
- Historical uv invocation is stated in the commit and cookbook. Independently replayed uv remove --offline --no-sync kaggle-environments then uv add --offline --no-sync kaggle-environments==1.32.7 on copied parent files: both pyproject.toml and uv.lock reproduce byte-for-byte. This corroborates, rather than proves, the historical command execution.
- Installed kaggle-environments 1.32.7 Kaggriculture engine SHA-256: bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e, equal to kg/reference-2026-09-29:engine_rs/Cargo.toml package.metadata.kaggriculture.python-engine-sha256.
- CARGO_BUILD_JOBS=3 uvx --from rust-just just prepare exited 0: Rust 155 passed / 2 ignored; Python 722 passed / 3 skipped. Mypy passed 48 source files; formatting, lint and doc freshness passed. See prepare.log.
- Orbit replay parity passed: 75930761, 2 seats, 103 transitions; 75926553, 4 seats, 222 transitions (325 total). All six generation fixture tests passed. No parity bypass environment variables were set. Two ignored Rust cases are expensive action-angle audits, not parity tests.
- Python skips: two FlashAttention CUDA tests and one unavailable x86 quantized-backend test on macOS ARM64/Python 3.12.13. No CUDA runtime qualification claim.
- uv run --locked python scripts/generate_reference_fixtures.py --outfile ops/deps-independent-verification-fb65e1f/reference_generation.current.json exited 0. Parsed output matches provided generation fixture except absolute reference_file path.
- docs/kaggriculture-contract.md diff is exactly one inserted blank line at 173; no wording change.
- git diff --exit-code and git diff --cached --exit-code both exited 0. Pre-existing .codex-tmp/ remains untracked; this receipt directory is untracked.

## Critical packages

Parent lock is byte-identical to Isaiah 32b3ec900ad406eedd965f53a1a0f4490d31c589. Complete critical package records remain unchanged: torch 2.9.0, 2.9.0+cpu, 2.9.0+cu128; triton 3.5.0; flash-attn 2.8.3; numpy 2.4.4; pydantic 2.13.3; wandb 0.26.1; maturin 1.13.1; mypy 1.20.2.

## Behavioral scope

- click 8.3.3 -> 8.5.0 is shared with W&B; possible training telemetry/CLI impact is not ruled out by unchanged wandb version. Live W&B was not exercised.
- JAX/Flax/Optax/Orbax/SciPy, OpenSpiel, Transformers/Hugging Face and LLM SDK changes can affect Kaggle reference/agent tooling. No direct canonical PPO import of the other changed packages was identified; no numerical regression was observed in the requested local suite.
- Installed Orbit source differs from old locked git source only by extracting episode-seed resolution into utils.resolve_episode_seed. Shared core adds lazy environment registration; utils also changes HTML payload insertion. Current generation fixtures match; supplied historical replay parity passes. See source diff artifacts. These checks do not exhaustively qualify all upstream environments or renderer behavior.

## Exhaustive package delta

Old: 176 records / 174 names. New: 181 records / 173 names. Top-level lock metadata is unchanged. Registry source remains PyPI for all changed third-party records other than kaggle-environments, which changes from git commit 6458c3191c2c4b37b6ad7530bd027df4b35369e4 to PyPI. New-version records include their generated artifact URLs/hashes/sizes/timestamps; a compact per-package version summary is in lock-delta-summary.json; exact before/after records are recoverable with `git diff fb65e1f^ fb65e1f -- uv.lock`.

| Package | Before | After | Changed fields |
|---|---|---|---|
| absl-py | 2.4.0 | 2.5.0 | sdist, version, wheels |
| aiohappyeyeballs | 2.6.1 | 2.7.1 | sdist, version, wheels |
| aiohttp | 3.13.5 | 3.14.3 | dependencies, sdist, version, wheels |
| annotated-doc | 0.0.4 | 0.0.5 | sdist, version, wheels |
| anyio | 4.13.0 | 4.14.2, 4.15.1 | record set (see JSON) |
| cffi | 2.0.0 | 2.1.1 | sdist, version, wheels |
| chex | 0.1.91 | 0.1.92 | dependencies, sdist, version, wheels |
| click | 8.3.3 | 8.5.0 | dependencies, sdist, version, wheels |
| contourpy | 1.3.3 | 1.3.3, 1.4.0 | record set (see JSON) |
| cryptography | 48.0.0 | 50.0.1 | sdist, version, wheels |
| flax | 0.12.7 | 0.12.10, 0.12.8 | record set (see JSON) |
| fonttools | 4.62.1 | 4.66.0 | sdist, version, wheels |
| google-auth | 2.50.0 | 2.59.0 | sdist, version, wheels |
| grpcio | 1.80.0 | absent | record set (see JSON) |
| grpcio-tools | 1.80.0 | absent | record set (see JSON) |
| gymnax | 0.0.8 | 0.0.8 | dependencies |
| hf-xet | 1.4.3 | 1.6.0 | sdist, version, wheels |
| httpx | 0.28.1 | 0.28.1 | dependencies |
| huggingface-hub | 1.13.0 | 1.33.0 | dependencies, sdist, version, wheels |
| humanize | 4.15.0 | 4.16.0 | sdist, version, wheels |
| importlib-metadata | 9.0.0 | 8.9.0 | sdist, version, wheels |
| jax | 0.10.0 | 0.10.2, 0.11.2 | record set (see JSON) |
| jaxlib | 0.10.0 | 0.10.2, 0.11.2 | record set (see JSON) |
| jiter | 0.14.0 | 0.17.0 | sdist, version, wheels |
| kaggle-environments | 1.29.0 | 1.32.7 | dependencies, sdist, source, version, wheels |
| kiwisolver | 1.5.0 | 1.5.1 | sdist, version, wheels |
| litellm | 1.82.4 | 1.93.2 | sdist, version, wheels |
| matplotlib | 3.10.9 | 3.11.2 | dependencies, sdist, version, wheels |
| ml-dtypes | 0.5.4 | 0.6.0 | sdist, version, wheels |
| msgpack | 1.1.2 | 1.2.3 | sdist, version, wheels |
| multidict | 6.7.1 | 6.9.1 | sdist, version, wheels |
| open-spiel | 1.6.13 | 2.0.1 | dependencies, sdist, version, wheels |
| openai | 2.34.0 | 2.54.0 | dependencies, sdist, version, wheels |
| optax | 0.2.8 | 0.2.8 | dependencies |
| orbax-checkpoint | 0.11.37 | 0.12.6 | dependencies, sdist, version, wheels |
| owl | 0.1.0 | 0.1.0 | metadata |
| pandas | 3.0.2 | 3.0.6 | sdist, version, wheels |
| pillow | 12.2.0 | 12.3.0 | sdist, version, wheels |
| prometheus-client | absent | 0.26.0 | record set (see JSON) |
| propcache | 0.4.1 | 0.5.4 | sdist, version, wheels |
| pyasn1 | 0.6.3 | 0.6.4 | sdist, version, wheels |
| pyjson5 | 2.0.0 | 2.0.1 | sdist, version, wheels |
| pyparsing | 3.3.2 | 3.3.3 | sdist, version, wheels |
| python-dotenv | 1.2.2 | 1.2.3 | sdist, version, wheels |
| regex | 2026.4.4 | 2026.9.29 | sdist, version, wheels |
| safetensors | 0.7.0 | 0.8.0 | sdist, version, wheels |
| scipy | 1.17.1 | 1.17.1, 1.18.1 | record set (see JSON) |
| simplejson | 4.1.1 | 4.1.2 | sdist, version, wheels |
| tensorstore | 0.1.82 | 0.1.85 | sdist, version, wheels |
| tiktoken | 0.12.0 | 0.14.0 | sdist, version, wheels |
| tokenizers | 0.22.2 | 0.23.2 | sdist, version, wheels |
| transformers | 5.8.0 | 5.17.0 | sdist, version, wheels |
| typer | 0.25.1 | 0.27.2 | dependencies, sdist, version, wheels |
| tzdata | 2026.2 | 2026.4 | sdist, version, wheels |
| werkzeug | 3.1.8 | 3.1.9 | sdist, version, wheels |
| yarl | 1.23.0 | 1.25.1 | sdist, version, wheels |
| zipp | 3.23.1 | 4.1.0 | sdist, version, wheels |

Marker splits: anyio uses 4.14.2 on Python <3.15 and 4.15.1 on >=3.15. On Python 3.11 / >=3.12 respectively: contourpy 1.3.3 / 1.4.0; flax 0.12.8 / 0.12.10; jax and jaxlib 0.10.2 / 0.11.2; scipy 1.17.1 / 1.18.1. gymnax and optax only gain the corresponding Flax/JAX dependency references; httpx only gains split anyio references; owl replaces git requirement metadata with ==1.32.7.

Dependency-edge details: aiohttp adds conditional typing-extensions; huggingface-hub replaces typer with click; click drops colorama; typer replaces click with Windows colorama; orbax-checkpoint replaces grpcio-tools with prometheus-client, removing the unused grpcio/grpcio-tools records. Remaining edge changes select the split Python-version records listed above. No hidden artifact-only package changes were found. Retained contourpy 1.3.3 and scipy 1.17.1 artifacts remain unchanged.

Independent graph audit: W&B's dependency closure intersects the changed set at click; Torch's closure does not intersect it. Kaggle's environment registration imports many environment modules, so the changed numerical/LLM dependencies can affect Kaggle package startup even without direct imports from the PPO implementation.
