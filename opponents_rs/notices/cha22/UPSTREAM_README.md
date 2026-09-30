# cha22 — native Rust agent

Exact [abhinav0370 notebook](https://www.kaggle.com/code/abhinav0370/cha22-agent),
ID135642255, retrieved 2026-09-24. The untouched source and full upstream license
notices are in `main.py`; source SHA256 `127ed3e62988c0474d386db6527ae8ca9de9bb1fe7004128557ddef67126c652`.

The original `submission_cha22.tar.gz` was submitted as **56523558**, **COMPLETE**.
Only the original Python archive was uploaded.

## Entry point

The complete source entry is `ig_agent(observation, configuration=None)`, also
aliased as `kaggle_agent` and `cha20_entry_agent`. The plain `agent` callable is
an earlier layer and omits the remaining suffix. The Rust port follows `ig_agent`.

## Native usage

```sh
cargo build --release --manifest-path engine_rs/Cargo.toml --lib --bin cha22_agent
engine_rs/target/release/cha22_agent
```

Each stdin line contains `{"observation": {...}, "configuration": {...}}`;
stdout emits one action per line. Keep one process per game. No Python interpreter
is used by the binary. The batch interface accepts `NativeAgentKind::Cha22` and
`RustExplicitBatch.native_actions([(game_index, seat, "cha22")])`.

Source custody and verification receipts are in
`data/intel/public-agents/abhinav0370-cha22-0924/`.

## Verification limits

Final bounded checks match5,752 saved/mirrored actions,237 direct cases and64 clones.
Metav4/V47/V56 regressions and native integration checks pass. ADV equal-price
ordering depends on the Python source hash seed; Rust keeps tape appearance order.
Inactive alternative PIPE openings are not exposed. No new games or strength/timing
claim. See the cookbook decision and retained verification artifacts for denominators.
