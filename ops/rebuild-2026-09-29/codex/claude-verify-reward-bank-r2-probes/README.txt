Verifier r2 probes for kg/rebuild-reward-bank 8aeba3d (scratch worktrees, since removed).

byte_identity.py   Replays the 16 tracked fixture games (tokens from
                   tests/fixtures/kaggriculture_env_reference_v1.*) x 3 trials =
                   48 games x 719 steps under randomized relative-reward configs.
                   Run once at HEAD (arg 3 = 1: w_b 0 with random inactive S in
                   {0,1,1e5,7.3} and cap_b in {0,.25,.9}) and once at merge base
                   4731c49 (arg 3 = 0: seven-key dict). Result: rewards arrays
                   byte-identical; both npz sha256
                   c033c1b79c22bb9b200ca4d3979547f5deb569a91c3efbdc6feef75a5ea58ae2.
admission_fuzz.py  60,000 random coefficient vectors (pool of edge values incl.
                   subnormals, f64 max, budget edges) through
                   KaggricultureRewardConfig and the native constructor.
                   Result: 60,000 agree, 23,787 accepted, 0 mismatches.
diff_check.py      Compares Python transition_rewards bits against a temporary
                   ignored Rust test (appended to src/kaggriculture/env_tests.rs in
                   the scratch tree only) that dumped RewardConfig::transition bits
                   for 6 configs x 4,000 random transitions (random counters,
                   banks from {-5000,-0,0,1,2980,3000,3001,3003,3006,24999,25000,
                   70000,73000,99999,100000,150000,1e300,f64::MAX,-1e308}, done).
                   Result: 23,888 of 24,000 bit-equal; all 112 mismatches are
                   config 5 (w_b 5e-324, S 1e300), a signed-zero difference only
                   (Rust -0.0, oracle +0.0) on non-terminal steps; see finding 2.
