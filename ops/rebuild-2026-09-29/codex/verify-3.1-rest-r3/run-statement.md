# Independent verification, round 3

Target: three-dot diff e1458d2a717d9d731a367cbb78b98616ee6649f4...4cac1a18f2209c54d40bef80d44755334a1a1ed2, Task 3.1 model-side registration/compile and Task 2.2 critic. No complete trainer or GPU qualification is claimed.

Questions: Do factory dispatch and compile boundaries preserve Isaiah's path? Does the masked winner critic follow the brief and contract v4? Are prior review findings resolved, and do tests detect broken behavior?

Inputs: local plan, critic brief, contract v4, Isaiah git object 32b3ec9, source/tests and cookbook records. CPU checks: requested non-slow pytest and mypy, Orbit subset collection, two tiny targeted mutation runs, post-restoration py-prepare; plan's supplementary cargo test.

Mutations: replace masked critic log-softmax input with raw logits; separately return the compiled trunk before guard/chunking. Expected observations: masked-critic assertion failure, and recorded chunk calls violating the guard test. Restore original bytes in finally blocks and verify SHA-256 after each.

Stopping condition: requested checks finish, each mutation fails its intended behavioral assertion, post-restoration checks pass or failures are explained, every tracked file hash matches entry, and verdict scopes unresolved integration/CUDA limits. This is verification only, with no durable repository adaptation.
