import struct, sys, torch
from owl.kaggriculture.rewards import KaggricultureRewardConfig, transition_rewards
K = ["econ_shaping","econ_starvation_weight","econ_drought_weight","econ_cap","econ_ineffective_weight",
     "econ_ineffective_cap","econ_bank_weight","econ_bank_scale","econ_bank_cap"]
CFGS = [(0.2,4.,1.,0.25,0.,0.1,0.25,100_000.,0.25),(0.2,4.,1.,0.25,0.,0.1,1.,100_000.,0.25),
        (0.2,4.,1.,0.25,0.001,0.1,0.37,7.3,0.3),(0.,0.,0.,0.,0.,0.,1e300,1e-300,0.5),
        (0.02,4.,1.,0.25,0.001,0.1,0.,100_000.,0.),(0.2,4.,1.,0.25,0.,0.1,5e-324,1e300,0.25)]
cfgs = [KaggricultureRewardConfig.model_validate(dict(zip(K, c))) for c in CFGS]
f64 = lambda b: struct.unpack("<d", struct.pack("<Q", int(b)))[0]
rows = [l.split() for l in open(sys.argv[1])]
bad = 0; nonzero_bank = 0
for r in rows:
    ci = int(r[0]); c = list(map(int, r[1:13]))
    before = torch.zeros(2, 32, dtype=torch.int64); after = torch.zeros(2, 32, dtype=torch.int64)
    before[0, :3] = torch.tensor(c[0:3]); before[1, :3] = torch.tensor(c[3:6])
    after[0, :3] = torch.tensor(c[6:9]); after[1, :3] = torch.tensor(c[9:12])
    bb = torch.tensor([f64(r[13]), f64(r[14])], dtype=torch.float64)
    ba = torch.tensor([f64(r[15]), f64(r[16])], dtype=torch.float64)
    done = torch.tensor([r[17] == "1"] * 2)
    out = transition_rewards(before, after, bb, ba, done, cfgs[ci])
    bits = [struct.unpack("<I", struct.pack("<f", float(v)))[0] for v in out]
    if bits != [int(r[18]), int(r[19])]:
        bad += 1
        if bad < 5: print("MISMATCH", r, bits)
print(f"rows={len(rows)} mismatches={bad}")
