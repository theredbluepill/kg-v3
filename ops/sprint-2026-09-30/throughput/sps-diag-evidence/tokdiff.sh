set -e
cd /root/kg-v3
python3 - <<'PY'
src = open("scripts/bench_rollout_step.py").read()
a = "        action_hash.update(actions.lengths.numpy().tobytes())\n"
assert a in src
src = src.replace(a, a + "        _TOK.setdefault(variant, []).append((actions.tokens.clone(), actions.lengths.clone()))\n")
src = src.replace("_VARIANTS = (", "_TOK: dict = {}\n_VARIANTS = (", 1)
b = "    mean = statistics.mean(times)\n"
assert b in src
src = src.replace(b, b + "    torch.save(_TOK[variant], f'/root/sps-diag/tok-{variant}.pt')\n")
open("/root/sps-diag/bench_tokens.py", "w").write(src)
PY
cat > /root/sps-diag/tokcmp.py <<'PY'
import torch
b = torch.load("/root/sps-diag/tok-baseline.pt"); a = torch.load("/root/sps-diag/tok-actor.pt")
rows_diff = rows_tot = steps_diff = pad_only = 0
for i, ((tb, lb), (ta, la)) in enumerate(zip(b, a)):
    L = torch.equal(lb, la)
    pos = torch.arange(tb.shape[-1]).expand_as(tb) < lb.unsqueeze(-1)
    valid_eq = L and torch.equal(tb[pos], ta[pos])
    full_eq = torch.equal(tb, ta)
    rd = int(((tb != ta) & pos).any(-1).sum()) + int((lb != la).sum())
    rows_diff += rd; rows_tot += tb.shape[0]
    if not valid_eq: steps_diff += 1
    if valid_eq and not full_eq: pad_only += 1
    if i < 3 or not valid_eq and steps_diff <= 3: print(i, tuple(tb.shape), "len_eq", L, "valid_eq", valid_eq, "full_eq", full_eq)
print(f"steps={len(b)} steps_with_valid_token_diff={steps_diff} steps_pad_only_diff={pad_only} rows_with_diff={rows_diff}/{rows_tot}")
PY
