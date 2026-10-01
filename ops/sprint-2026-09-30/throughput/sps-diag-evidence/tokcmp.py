import torch
b = torch.load("tok-baseline.pt"); a = torch.load("tok-actor.pt")
tot_seat = diff_seat = len_diff = steps_valid = steps_pad = 0
first = []
for i, ((tb, lb), (ta, la)) in enumerate(zip(b, a)):
    F = tb.shape[2]
    pos = (torch.arange(F) < lb.unsqueeze(-1)).unsqueeze(-1).expand_as(tb)
    seat_diff = ((tb != ta) & pos).flatten(2).any(-1) | (lb != la)
    tot_seat += seat_diff.numel(); diff_seat += int(seat_diff.sum()); len_diff += int((lb != la).sum())
    v = bool(seat_diff.any()); full = torch.equal(tb, ta)
    steps_valid += v; steps_pad += (not v and not full)
    if v and len(first) < 3: first.append(i)
print("shape", tuple(b[0][0].shape), "steps", len(b), "steps_with_executed_token_diff", steps_valid, "steps_padding_only_diff", steps_pad, "seat_turns_diff", diff_seat, "/", tot_seat, "length_diffs", len_diff, "first_diff_steps", first)
if steps_pad:
    tb, lb = b[0]; ta, la = a[0]
    F = tb.shape[2]; pad = ~(torch.arange(F) < lb.unsqueeze(-1)).unsqueeze(-1).expand_as(tb)
    print("padding values baseline", tb[pad].unique()[:8].tolist(), "actor", ta[pad].unique()[:8].tolist())
