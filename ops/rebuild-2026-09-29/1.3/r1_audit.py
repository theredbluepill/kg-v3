"""Read-only R1 feasibility: stream official states; bound seeded HIREs arithmetically."""
from collections import Counter
import gzip
import json
from pathlib import Path

root = Path(__file__).resolve().parents[3]
selected = set(range(32)) | set(range(344, 376)) | set(range(687, 719))
items = ['WHEAT','CARROT','TOMATO','STRAWBERRY','MELON','EGG','MILK','WOOL','FERTILIZER','GOOSE','COW','SHEEP']
counts = Counter()
max_actors = 0
for episode in [95324500,95901360,95921764,95990191]:
    with gzip.open(root / f'engine_rs/fixtures/episode-{episode}.jsonl.gz', 'rt') as stream:
        header = json.loads(next(stream))
        states = [(header['initial']['public'], header['initial']['privates'])]
        def visit(public, privates):
            global max_actors
            if public['step'] not in selected:
                return
            counts['records'] += 1
            max_actors = max(max_actors, *(1 + len(f['hands']) for f in public['farms']))
            counts['actor_gt16_states'] += any(1 + len(f['hands']) > 16 for f in public['farms'])
            counts['shops_ge4_states'] += len(public['town']['unlocked_shops']) >= 4
            counts['both_hires_nonzero_states'] += all(f['hires_today'] > 0 for f in public['farms'])
            for farm, private in zip(public['farms'], privates):
                for inv in private['inventories']:
                    counts['reordered_inventories'] += list(inv) != sorted(inv, key=items.index)
                counts['reordered_sheds'] += list(private['shed']) != sorted(private['shed'], key=items.index)
                for row in farm['tiles']:
                    for tile in row:
                        kind = 'EMPTY' if tile is None else tile if isinstance(tile,str) else tile['kind']
                        counts['tile_' + kind] += 1
                        if not isinstance(tile, dict):
                            continue
                        if kind == 'PLANT':
                            counts['crop_' + tile['crop']] += 1
                            fert = tile['fertilized_until_day']
                            counts['fert_current'] += fert >= public['day']
                            counts['fert_expired'] += 0 <= fert < public['day']
                            counts['unwatered'] += tile['consecutive_unwatered'] > 0
                        if 'animal' in tile:
                            counts['animal_' + tile['animal']] += 1
                            counts['unfed'] += tile['consecutive_unfed'] > 0
        visit(*states[0])
        for line in stream:
            row = json.loads(line)
            visit(row['expected'], row['privates'])
limits=[]
for D,M in [(24,10),(12,4),(8,3),(6,1),(30,8),(16,5)]:
    limits.append(1 + max(sum((t+s+q)%8 == 0 for t in range((n//D)*D,n) for q in range(min(M,2)))
                          for n in range(96) for s in range(2)))
print(json.dumps({'official_selected':dict(counts),'official_max_actors':max_actors,
                  'seeded_actor_upper_bounds':limits,'quota_actor_gt16_required':4,
                  'quota_actor_gt16_possible':counts['actor_gt16_states']},indent=2))
