# Stage 1 codec checks

Target: four cold codec wrappers preserve native argument/JSON transport and
publish batch results only after every row succeeds. Stop at CPU tests and
static checks; no native grammar or lifecycle implementation belongs here.

All commands below used `OMP_NUM_THREADS=2 CARGO_BUILD_JOBS=2
CARGO_NET_OFFLINE=true UV_OFFLINE=true` and no network access.

| Command | Exit | Outcome | Receipt |
| --- | --- | --- | --- |
| `uv run --offline pytest tests/kaggriculture/test_codec.py -q` before implementation | 2 | 1 collection error: codec module absent | `codec-red.log` |
| Same pytest command before parallel types alias landed | 2 | 1 collection error: `JsonValue` absent | `codec-missing-alias.log` |
| `uv run --offline ruff format python/owl/kaggriculture/codec.py tests/kaggriculture/test_codec.py` | 0 | 2 files reformatted | Tool output |
| `uv run --offline ruff check python/owl/kaggriculture/codec.py tests/kaggriculture/test_codec.py` initially | 1 | 5 test-style findings, then corrected | Tool output |
| Same Ruff check after correction | 0 | All checks passed | `codec-ruff.log` |
| Same pytest command after implementation and alias landing | 0 | 6 passed, 1 skipped | `codec-green.log` |
| `uv run --offline mypy python/owl/kaggriculture/codec.py` | 0 | No issues in 1 source file | `codec-mypy.log` |

`test_reference_programs_round_trip` is skipped with exactly
`needs Task 1.4 binding`. Its body reads the hashed compressed native/reference
fixture and checks 321 accepted canonical round trips and 43 rejections. The
current six runnable tests prove forwarding, one serialization/parse per single
call, own-seat actor counts, observation order limits, action-spec hire limits,
private batch publication, native failure propagation and no missing-binding
fallback. A spy rejecting before a write proves the adapter adds no write; it
does not qualify native atomicity. Stage 2 must run the fixture against the real
binding.
