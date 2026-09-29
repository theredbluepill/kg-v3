"""Bounded probe: reference actor_codec admission on one local non-slice episode."""
import collections, hashlib, importlib.util, json, sys, time
spec = importlib.util.spec_from_file_location("ref_codec", sys.argv[1])
codec = importlib.util.module_from_spec(spec); spec.loader.exec_module(codec)
raw = open(sys.argv[2], "rb").read()
d = json.loads(raw)
shared = [k for k, v in d["specification"]["observation"].items() if v.get("shared")]
cfg = dict(d["configuration"]); cfg["seed"] = None
drops = collections.Counter(); admitted = 0; norm = collections.Counter(); t0 = time.time()
for turn in range(len(d["steps"]) - 1):
    obs = []
    for seat in (0, 1):
        o = dict(d["steps"][turn][seat]["observation"])
        for k in shared:
            o[k] = d["steps"][turn][0]["observation"][k]
        obs.append(o)
    try:
        for seat in (0, 1):
            a = d["steps"][turn + 1][seat]["action"]
            if not isinstance(a, dict): raise codec.CodecError("recorded action is not a mapping")
            if set(a) - {"farmer", "hands", "market"}: raise codec.CodecError("unsupported action keys")
            for key in ("hands", "market"):
                if key not in a or a[key] is None: norm[f"{key}_absent_or_null"] += 1
            n = len(obs[seat]["farms"][seat].get("hands") or [])
            hands = a.get("hands") or []
            if len(hands) > n: raise codec.CodecError("extra hand commands")
            act = {"farmer": a.get("farmer"), "hands": hands + [None] * (n - len(hands)), "market": a.get("market") or []}
            codec.encode_action(act, {"observation": obs[seat], "configuration": cfg}, hire_limit=241)
    except (codec.CodecError, ValueError, TypeError, KeyError) as e:
        drops[f"{type(e).__name__}:{e}"] += 1; continue
    admitted += 1
print(json.dumps({"episode_sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "paired_turns": len(d["steps"]) - 1,
  "admitted": admitted, "rejected": dict(drops), "normalized": dict(norm), "seconds": round(time.time() - t0, 1)}, indent=1))
