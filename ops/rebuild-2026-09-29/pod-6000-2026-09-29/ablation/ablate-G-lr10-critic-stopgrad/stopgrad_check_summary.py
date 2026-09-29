import sys, json
dec = json.JSONDecoder()
print("run\trank\titer\tstopgrad\tpassed\t" + "\t".join(f"{g}_none/allzero/norm" for g in ("shared","actor_only","critic_value_tokens","critic_head")))
for path in sys.argv[1:]:
    for l in open(path, errors="replace"):
        i = l.find("[kg-probe] ")
        if i < 0: continue
        try: r, _ = dec.raw_decode(l, i + 11)
        except json.JSONDecodeError: continue
        if r.get("kind") != "stopgrad_check": continue
        g = r["groups"]
        cells = [f"{g[k]['none_grads']}/{g[k]['all_zero_grads']} of {g[k]['tensors']} / {g[k]['grad_norm']:.4g}" for k in ("shared","actor_only","critic_value_tokens","critic_head")]
        print(f"{path.split('/')[-2]}\t{r['rank']}\t{r['iteration']}\t{r['stopgrad']}\t{r['passed']}\t" + "\t".join(cells))
