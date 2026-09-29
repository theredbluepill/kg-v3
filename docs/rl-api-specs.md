# RL API Specs

This document describes the currently available RL observation and action specs.
The Python config API uses pydantic discriminator fields so future specs can add
different options without changing the outer `VectorizedEnv` constructor shape.
`EnvConfig.n_envs` defaults to `2` and must be even so built-in checkpoint
evaluation can split evaluation games across 2-player and 4-player batches.
All tensor shapes in this document are local to one `VectorizedEnv` instance.
Distributed PPO creates one vectorized environment per rank, so global rollout
width is a training-layer concern rather than an RL API shape change.

```python
from owl.rl import ActionPureConfig, EntityBasedConfig, VectorizedEnv

env = VectorizedEnv(
    n_envs=128,
    obs_spec=EntityBasedConfig(max_entities=256),
    action_spec=ActionPureConfig(max_per_planet_launches=1),
    reward_mode="win_loss",
)
```

## Shared Constants

- `MAX_PLANETS = 40`
- `MAX_COMETS = 4`
- `MAX_COMET_PATH_LENGTH = 40`
- `DEFAULT_MAX_ENTITIES = 256`
- `ACTION_ENTITY_SLOTS = MAX_PLANETS + MAX_COMETS = 44`
- `OUTER_PLAYER_SLOTS = 4`
- `GLOBAL_EXT_V2_CHANNELS = 14`
- `PLAYER_FEATURE_CHANNELS = 14`
- `CROSS_ATTENTION_FLEET_CHANNELS = 46`
- `TARGET_INCOMING_CHANNELS = 48`

`max_entities` controls total non-global entity capacity. Fleet capacity is:

```text
max_fleets = max_entities - (MAX_PLANETS + MAX_COMETS)
```

The default `max_entities=256` gives `max_fleets=212`.

`MAX_PLANETS` matches the current generated-map upper bound:
`MAX_PLANET_GROUPS * 4 = 10 * 4`. `MAX_COMET_PATH_LENGTH` matches the comet
generator's maximum accepted visible path length; generated comet paths outside
the range `5..=40` are rejected.

## EntityBased

Config:

```python
{"obs_spec": "entity_based", "max_entities": 256}
```

`EntityBased` writes observations into reusable caller-owned buffers. The vectorized
environment returns an `ObsBatch` with these tensors:

| Tensor | dtype | Shape |
| --- | --- | --- |
| `planets` | `float32` | `(n_envs, MAX_PLANETS, 107)` |
| `orbiting_planets` | `bool` | `(n_envs, MAX_PLANETS)` |
| `fleets` | `float32` | `(n_envs, max_fleets, 79)` |
| `comets` | `float32` | `(n_envs, MAX_COMETS, 330)` |
| `entity_mask` | `bool` | `(n_envs, max_entities)` |
| `still_playing` | `bool` | `(n_envs, 4)` |
| `global_features` | `float32` | `(n_envs, 3)` |
| `action_mask.can_act` | `bool` | action-spec dependent |
| `action_mask.max_launch` | `int64` | pure and discrete-target masks only; `(n_envs, 4, ACTION_ENTITY_SLOTS)` |

All reused buffers are fully overwritten on each observation write. Inactive
rows are zero-filled and their `entity_mask` slots are set to `False`.
`entity_mask` is ordered as all planet slots, then comet slots, then fleet
slots. This matches the model token order and keeps the action entity axis in
the first `ACTION_ENTITY_SLOTS` positions.

`still_playing` is true for outer player slots that are active in the current
observation's episode and have not finished. The Rust vectorized environment
samples a fresh random internal-to-outer player-slot mapping for each sub-env
reset. In 4-player episodes all four outer slots are active; in 2-player
episodes any two of the four outer slots may be active. Inactive outer player
slots are `False`. This mapping also randomizes starting-seat assignment for
fixed outer-slot policies in evaluation and benchmarking code, so callers do not
need to rotate policy-to-slot assignments themselves. After a terminal
auto-reset, `still_playing` describes the returned reset observation, while
`dones` still describes the transition that just finished.
For target-bin observations, `DiscreteTargetBinActionMask` has no `max_launch`
member.

## EntityBasedExtV1

Config:

```python
{"obs_spec": "entity_based_ext_v1", "max_entities": 256, "ship_count_one_hot_max": 50}
```

`EntityBasedExtV1` includes every `EntityBased` feature and appends configurable
ship-count one-hot vectors to planet and fleet rows only. Comet rows are
unchanged.

With `ship_count_one_hot_max=N`, planet rows append `N + 1` channels: bin `0`
is exact zero ships, bins `1..N-1` are exact ship counts, and bin `N` is the
overflow bin for `ships >= N`. Fleet rows append `N` channels because fleets
cannot have zero ships: bins `0..N-2` represent exact ship counts `1..N-1`, and
bin `N-1` is the overflow bin for `ships >= N`. The default is `N=50`, so
planet rows have `158` channels and fleet rows have `129` channels by default.

### Normalization

- Positions use `x_norm = (x / BOARD_SIZE) * 2 - 1`, with `BOARD_SIZE = 100`.
  The four map corners are `(-1, -1)`, `(1, -1)`, `(-1, 1)`, and `(1, 1)`.
- Radius uses `radius / 3`.
- Neutral planet linear ships use `ships / 100`. Player-owned planet, fleet,
  and comet linear ships use `ships / 500`.
- Log ships use `ln(ships + 1) / ln(100)`.
- Ship-count basis channels append linear two-hot, ln-space two-hot, and
  overflow features. Zero ships are a special-case bucket when the bucket grid
  includes zero. Neutral planet and comet buckets are
  `[0, 1, 2, 4, 8, 16, 32, 64, 99]`; owned planet buckets are
  `[0, 1, 2, 4, ..., 1024]`; owned comet buckets are
  `[0, 1, 2, 4, ..., 512]`. Fleet buckets omit the zero bucket, start at `1`,
  then continue with the next power of two strictly above `min_fleet_size`, up
  to `512`. The two overflow channels are `ships > max_bucket` and
  `ln(max(ships - max_bucket, 1))`.
- Angular velocity uses `(angular_velocity - 0.025) / 0.025`. Generated games
  currently map the expected range `[0.025, 0.05]` to `[0, 1]`. The value is
  not clamped.
- `steps_until_next_comet_spawn` is divided by `100`.
- Appended spatial channels use the already-normalized `x` and `y` values.
  Cartesian Fourier features use frequencies `[1, 2, 4, 8, 16, 32]`. Radial
  Fourier features use frequencies `[1, 2, 4, 8]`.

### Planet Tensor

Shape per env: `(MAX_PLANETS, 108 + N)`, where `N` is
`ship_count_one_hot_max`. The default `N=50` gives a width of `158`; the base
`EntityBased` width before the ExtV1 appendix is `107`.

Only non-comet planets are included. If more than `MAX_PLANETS` non-comet
planets exist, the encoder panics. Generated games currently produce up to
`MAX_PLANET_GROUPS * 4 = 40` planets. Rows are written in increasing planet ID
order after excluding comet planets. This matches generated-map order because
generated planet IDs are unique and contiguous before comet insertion.

| Channels | Feature |
| --- | --- |
| `0..3` | owner one-hot for players `0..3` |
| `4` | neutral owner |
| `5` | normalized `x` |
| `6` | normalized `y` |
| `7..11` | production one-hot for production values `1..5` |
| `12` | normalized radius |
| `13` | neutral normalized ships, else `0` |
| `14` | neutral normalized log ships, else `0` |
| `15` | player-owned normalized ships, else `0` |
| `16` | player-owned normalized log ships, else `0` |
| `17..37` | neutral planet ship-count basis, else `0` |
| `37..63` | player-owned planet ship-count basis, else `0` |
| `63..87` | Cartesian Fourier position features for normalized `(x, y)` |
| `87` | sun-centered radius `r = sqrt(x^2 + y^2)` |
| `88` | `log1p(r)` |
| `89` | `sin(theta)` for `theta = atan2(y, x)` |
| `90` | `cos(theta)` |
| `91..97` | angular harmonics `sin(k theta), cos(k theta)` for `k = 2..4` |
| `97..105` | radial Fourier features `sin(pi f r), cos(pi f r)` |
| `105` | orbiting planet `vx = -angular_velocity * y`, else `0` |
| `106` | orbiting planet `vy = angular_velocity * x`, else `0` |

### Orbiting Planet Tensor

Shape per env: `(MAX_PLANETS,)`.

Rows are aligned with the planet tensor. A row is `True` if the matching planet
row is orbiting, else `False`. Inactive rows are `False`.

`entity_mask[i]` is `True` only for active planet rows for
`i < MAX_PLANETS`.

### Fleet Tensor

Shape per env: `(max_fleets, 79 + N)`, where `N` is
`ship_count_one_hot_max`. The default `N=50` gives a width of `129`; the base
`EntityBased` width before the ExtV1 appendix is `79`.

The low-level `encode_entity_based` Rust API filters fleets smaller than its
fleet-filter threshold before writing fleet rows. The default threshold is
`min_fleet_size`; callers that need agent-only filtering without changing action
masks can pass a separate `fleet_filter_min_size`. Fleets at or above the
threshold are kept. If a player owns no current planet and none of their fleets
meet the threshold, the encoder keeps that player's largest below-threshold
fleet, using the lower fleet id as the tie-breaker. The API also returns the
number of fleets dropped by this filter.

When all remaining active fleets fit in `max_fleets`, fleets are emitted in
simulator fleet order. If there are more remaining active fleets than
`max_fleets`, fleets are sorted by descending ship count, with fleet id as the
tie-breaker, so the largest fleets are kept and the rest are ignored. Overflow
is silent; vector-env training metrics report
`max_entities_exceeded_per_game` for terminal episodes.

| Channels | Feature |
| --- | --- |
| `0..3` | owner one-hot for players `0..3` |
| `4` | normalized `x` |
| `5` | normalized `y` |
| `6` | normalized `vx`, divided by `shipSpeed` |
| `7` | normalized `vy`, divided by `shipSpeed` |
| `8` | normalized ships |
| `9` | normalized log ships |
| `10..32` | fleet ship-count basis |
| `32..56` | Cartesian Fourier position features for normalized `(x, y)` |
| `56` | sun-centered radius `r = sqrt(x^2 + y^2)` |
| `57` | `log1p(r)` |
| `58` | `sin(theta)` for `theta = atan2(y, x)` |
| `59` | `cos(theta)` |
| `60..66` | angular harmonics `sin(k theta), cos(k theta)` for `k = 2..4` |
| `66..74` | radial Fourier features `sin(pi f r), cos(pi f r)` |
| `74` | normalized speed `sqrt(vx^2 + vy^2)` |
| `75` | heading `x` component, or `0` for zero-speed fleets |
| `76` | heading `y` component, or `0` for zero-speed fleets |
| `77` | radial velocity in the sun-centered radial basis |
| `78` | tangential velocity in the counterclockwise tangent basis |

`entity_mask[ACTION_ENTITY_SLOTS + i]` is `True` only for active fleet rows.

### Comet Tensor

Shape per env: `(MAX_COMETS, 330)`.

Comets are encoded separately from normal planets. Active comet planet IDs are
sorted in ascending ID order, deduplicated, and emitted up to `MAX_COMETS`.
This matches the comet portion of the action entity axis.

| Channels | Feature |
| --- | --- |
| `0..3` | owner one-hot for players `0..3` |
| `4` | neutral owner |
| `5` | normalized ships |
| `6` | normalized log ships |
| `7..27` | neutral comet ship-count basis, else `0` |
| `27..51` | player-owned comet ship-count basis, else `0` |
| `51` | remaining stored path points, including the current point, divided by `MAX_COMET_PATH_LENGTH` |
| `52` | current normalized `x` from the path |
| `53` | current normalized `y` from the path |
| `54..96` | current Cartesian Fourier, polar, angular-harmonic, and radial Fourier spatial features |
| `96` | normalized `vx` from the next path point minus the current path point |
| `97` | normalized `vy` from the next path point minus the current path point |
| `98` | speed `sqrt(vx^2 + vy^2)` |
| `99` | heading `x` component, or `0` if no next path point exists |
| `100` | heading `y` component, or `0` if no next path point exists |
| `101` | radial velocity in the sun-centered radial basis |
| `102` | tangential velocity in the counterclockwise tangent basis |
| `103..108` | future-valid flags for offsets `[1, 2, 4, 8, 16]`, encoded as `0.0` or `1.0` |
| `108..118` | selected future normalized `(x, y)` pairs for offsets `[1, 2, 4, 8, 16]` |
| `118..328` | spatial features for each selected future position |
| `328` | normalized final path `x` minus current normalized `x` |
| `329` | normalized final path `y` minus current normalized `y` |

The current path point starts at the comet group's current `path_index`. If
`path_index < 0`, the encoder starts at path index `0`. If a selected future
offset is outside the remaining known path, its valid flag and feature slots are
zero-filled. If no next path point exists, comet velocity, speed, heading,
radial velocity, and tangential velocity are zero-filled. The displacement to
the final path point uses the path's last known point.

`entity_mask[MAX_PLANETS + i]` is `True` only for active comet rows.

### Global Tensor

Shape per env: `(3,)`.

| Index | Feature |
| --- | --- |
| `0` | `step / episode_steps` |
| `1` | `steps_until_next_comet_spawn / 100` |
| `2` | normalized angular velocity |

## EntityBasedExtV2

Config:

```python
{"obs_spec": "entity_based_ext_v2", "max_entities": 256}
```

`EntityBasedExtV2` extends the base `EntityBased` observation directly. It does
not include the `EntityBasedExtV1` ship-count one-hot appendices, so planet and
fleet row widths remain `107` and `79`.

The spec adds:

| Tensor | dtype | Shape |
| --- | --- | --- |
| `global_features` | `float32` | `(n_envs, 17)` |
| `player_features` | `float32` | `(n_envs, 4, 14)` |

For `entity_based` and `entity_based_ext_v1`, `ObsBatch.player_features` is
`None`. `EntityBasedExtV2` and `EntityBasedCrossAttnV1` provide player features.

V2 appends fourteen channels after the three base global channels. The
neutral features are for planets that are still neutral in the current
observation; if every planet has been colonized, all neutral production, ship,
and count channels are `0`.

| Channel | Feature |
| --- | --- |
| `3` | neutral total production, including comet planets, divided by `100` |
| `4` | neutral comet-planet production divided by `100` |
| `5` | neutral non-comet planet production divided by `100` |
| `6` | neutral total ships, including comet planets, divided by `5000` |
| `7` | `log1p` of neutral total ships, divided by `ln(1000)` |
| `8` | neutral comet-planet ships divided by `5000` |
| `9` | `log1p` of neutral comet-planet ships, divided by `ln(1000)` |
| `10` | neutral non-comet planet ships divided by `5000` |
| `11` | `log1p` of neutral non-comet planet ships, divided by `ln(1000)` |
| `12` | neutral comet count divided by `MAX_COMETS` |
| `13` | neutral non-comet planet count divided by `MAX_PLANETS` |
| `14` | one-hot for exactly `2` alive players |
| `15` | one-hot for exactly `3` alive players |
| `16` | one-hot for exactly `4` alive players |

Each outer player slot receives fourteen absolute summary channels:

| Channel | Feature |
| --- | --- |
| `0` | total production, including owned comet planets, divided by `100` |
| `1` | comet-planet production divided by `100` |
| `2` | non-comet planet production divided by `100` |
| `3` | total ships, including planets, comet planets, and fleets, divided by `5000` |
| `4` | `log1p` of total ships, divided by `ln(1000)` |
| `5` | comet-planet ships divided by `5000` |
| `6` | `log1p` of comet-planet ships, divided by `ln(1000)` |
| `7` | non-comet planet ships divided by `5000` |
| `8` | `log1p` of non-comet planet ships, divided by `ln(1000)` |
| `9` | fleet ships divided by `5000` |
| `10` | `log1p` of fleet ships, divided by `ln(1000)` |
| `11` | non-comet planet count divided by `MAX_PLANETS` |
| `12` | comet count divided by `MAX_COMETS` |
| `13` | fleet count divided by `100` |

Component production and linear ship channels use the same normalizer as their
total, so comet plus non-comet production equals total production, and comet
plus non-comet planet plus fleet linear ships equals total linear ships.
Inactive outer player slots are zero-filled.
When standalone/Kaggle observation encoding filters fleets out of the fleet
entity tensor, these per-player fleet ship and fleet count aggregate channels
still include the filtered fleets.

## EntityBasedCrossAttnV1

Config:

```python
{"obs_spec": "entity_based_cross_attn_v1", "max_entities": 256}
```

`EntityBasedCrossAttnV1` keeps the same planet, comet, global, and
per-player features as `EntityBasedExtV2`, but fleet rows are no longer
self-attention entity tokens. The vectorized environment and standalone encoder
add two tensors:

| Tensor | dtype | Shape |
| --- | --- | --- |
| `fleets` | `float32` | `(n_envs, max_fleets, 46)` |
| `fleet_target` | `int64` | `(n_envs, max_fleets)` |
| `target_incoming_features` | `float32` | `(n_envs, ACTION_ENTITY_SLOTS, 48)` |
| `global_features` | `float32` | `(n_envs, 17)` |
| `player_features` | `float32` | `(n_envs, 4, 14)` |

Rust routes current fleets by forward-simulating them until they collide with a
current planet or comet planet, leave the board, hit the sun, or the episode
ends. Fleets that leave the board, hit the sun, or collide only with an
expiring comet planet are omitted. Existing comet paths are simulated; future
comet spawns are intentionally ignored. Routed fleet rows are sorted by
descending ship count only when there are more routed fleets than `max_fleets`.

`entity_mask` keeps the same planet, comet, fleet-tail order. For this spec,
the model treats only the first `ACTION_ENTITY_SLOTS` mask entries as trunk
entity tokens and uses the fleet tail as the cross-attention memory mask.
`fleet_target[i]` is the action-entity slot that fleet row `i` will collide
with, or `-1` for inactive rows. Kaggle serving may compact inactive action
entities before inference; in that runtime path `target_incoming_features` is
sliced to the compacted action-entity axis and `fleet_target` is remapped to the
same compacted indices.

Fleet channels:

| Channels | Feature |
| --- | --- |
| `0..3` | owner one-hot for players `0..3` |
| `4` | normalized ships |
| `5` | normalized log ships |
| `6..27` | fleet ship-count basis |
| `28..43` | ETA one-hot buckets for turns `1..15` and `16+` |
| `44` | ETA overflow `(eta - 15) / 10` for `eta >= 16`, else `0` |
| `45` | post-comet flag; `1` when arrival is at or after the next scheduled comet spawn |

`target_incoming_features` has three 16-bucket groups over ETA buckets
`1..15` and `16+`:

| Channels | Feature |
| --- | --- |
| `0..15` | incoming fleet count divided by `100` |
| `16..31` | incoming ships divided by `5000` |
| `32..47` | `log1p` incoming ships divided by `ln(1000)` |

These per-target aggregates include all routed fleets before fleet-memory
truncation, so counts remain accurate even when more than `max_fleets` fleets
are inbound.

Standalone observation encoding uses strict application-boundary parsing:
required observation keys are read directly, `step` and `episode_steps` must be
integers rather than coerced strings or floats, comet groups must provide
matching `planet_ids`, `paths`, and `path_index`, and comet/path overflow is
rejected rather than truncated. It also rejects non-finite `angular_velocity`
and requires `episode_steps > 0` before computing these values.
The parser is still an internal high-throughput bridge into typed Rust state:
impossible ID invariants such as duplicate planet IDs or IDs outside the fixed
rules-engine limit may trip Rust assertions instead of being converted into
Python `ValueError`s.
`encode_python_observation()` returns a single-env `ObsBatch` with tensors
shaped as `(1, ...)`, matching model input shape directly.

Comet spawn steps currently come from the rules engine constant:

```text
[50, 150, 250, 350, 450]
```

If no future comet spawn remains, `steps_until_next_comet_spawn` is `0`.

## Action Entity Slots

Action entity slots are ordered as all `MAX_PLANETS` planet tokens first,
followed by `MAX_COMETS` comet tokens:

```text
0..39  -> non-comet planet slots in ascending planet ID order
40..43 -> comet slots in ascending comet planet ID order
```

Unused planet slots, unused comet slots, and inactive player slots are explicitly
filled with `False` / `0` in action-spec output tensors.

## Pure Action Spec

Config:

```python
{"action_spec": "pure", "max_per_planet_launches": 1, "min_fleet_size": 6}
```

The pure action spec exposes all launch decisions in direct tensor form. The
same entity axis is used for action masks and submitted actions.
`max_per_planet_launches` is validated in Python and Rust and must equal `1`.
`min_fleet_size` is validated in Python and Rust and must fit in the positive
`i32` ship-count range. `ActionPureConfig()` defaults to
`max_per_planet_launches=1` and `min_fleet_size=6`.

### Pure Output Tensors

These are written alongside the observation tensors:

| Tensor | dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `can_act` | `bool` | `(n_envs, 4, 44)` | whether a player can launch from an entity slot |
| `max_launch` | `int64` | `(n_envs, 4, 44)` | maximum launchable ship count for that slot |

`can_act[player, entity]` is true when the entity is owned by that outer player
slot and has at least `min_fleet_size` ships. In 2-player games, two random
outer player slots are active for the episode and the other two are inactive.

### Pure Submitted Actions

Call:

```python
actions = PureActions(launch=launch, angle=angle, ships=ships)
obs, rewards, dones, episode_metrics = env.step(actions)
```

`rewards` and `dones` have shape `(n_envs, 4)`. Inactive player slots are
always `done=True` with reward `0`. Active players receive reward `0` with
`done=False`. A player that newly loses receives reward `-1` with `done=True`;
later steps for that already-finished player stay `done=True` with reward `0`.
A sole winner receives reward `1` with `done=True`. If multiple players tie as
winners, each tied winner receives the average of one winner reward and the
remaining tied loser rewards: `(1 - (winner_count - 1)) / winner_count`.

| Tensor | dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `launch` | `bool` | `(n_envs, 4, 44, max_per_planet_launches)` | `True` means execute a launch |
| `angle` | `float32` | `(n_envs, 4, 44, max_per_planet_launches)` | launch angle in radians |
| `ships` | `int64` | `(n_envs, 4, 44, max_per_planet_launches)` | requested ship count |

Python requires exact submitted action dtypes at the boundary; wrong dtypes are
rejected instead of cast.
If `launch` is `False`, that slot is a no-op and `angle` / `ships` are ignored.
If `launch` is `True`, `ships >= min_fleet_size`, `ships <= i32::MAX`, a finite
`angle`, a valid source entity, source ownership by the acting player, and
enough remaining ships on that source are required. Invalid submitted actions
raise `ValueError`. Invalid-action errors are not atomic across sub-envs:
because decoding, stepping, and observation writing run in one parallel pass per
env, other sub-envs may have advanced before the error is returned.
Each player/source entity has one launch slot.

For Kaggle submissions, `actions_to_kaggle(obs, player, actions, action_spec=...)`
accepts a `PureActions` bundle with single batched tensors shaped
`(1, 4, 44, max_per_planet_launches)` and returns the selected player's
`[from_planet_id, angle, ships]` action triples. `from_planet_id` and `ships`
are returned as Python `int`; `angle` is returned as `float`. Pure triples use
the same Rust validation path as environment stepping.

## Discrete Targets Action Spec

Config:

```python
{
    "action_spec": "discrete_targets",
    "max_per_planet_launches": 1,
    "min_fleet_size": 6,
    "targeting_mode": "full_mask",
}
```

`ActionDiscreteTargetsConfig` uses the same launch-count and minimum-fleet
validation as `ActionPureConfig`, but submitted actions choose integer target
entity slots instead of raw launch angles. `StatelessTransformerV1` and PPO can
train against this action spec when the model actor config also uses
`"discrete_targets"` and `max_per_planet_launches=1`.

### Discrete Targets Output Tensors

| Tensor | dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `can_act` | `bool` | `(n_envs, 4, 44, 44)` | whether a player can launch from a source slot to a target slot |
| `max_launch` | `int64` | `(n_envs, 4, 44)` | maximum launchable ship count for that source slot |

`targeting_mode` controls target masking and selected bad-launch handling:

| Mode | Target mask | Bad selected target launch |
| --- | --- | --- |
| `"full_mask"` | Existing targets except self, plus the simulator's full static-target eligibility filter. | Selected sun-blocked static targets are replaced with no-op. Selected planet-blocked static targets can still fall back to a sun-safe ray, subject to the neutral-intercept gate below. Selected dynamic targets are replaced with no-op when no allowed or gated fallback ray exists. |
| `"stop_bad_launch"` | Existing targets except self; static obstruction, sun crossing, and dynamic feasibility are not masked. | Falls back through the target cone for a sun-avoiding ray and is replaced with no-op only when no gated fallback ray exists. |
| `"anything_goes"` | Existing targets except self; static obstruction, sun crossing, and dynamic feasibility are not masked. | Submitted even when the computed ray crosses the sun. Dynamic targets with no target-hit window still become no-ops because no launch angle is defined. |

In `"full_mask"`, static-source to static-target pairs use the reset-time
cached blocker-safe static target-cone result for masking. Fully
blocker-covered static targets are masked out. Selected static-source to
static-target launches also reuse the cached static-safe target arcs, then only
check dynamic blockers at launch time. Dynamic-source to static-target pairs
recompute the same static target-cone sun/static-blocker check for the current
step while masking.
Dynamic targets remain eligible in the mask because full obstruction checks are
deferred until the selected launch is decoded. In the loose modes,
`can_act[player, source, target]` is true when the source entity is owned by
that outer player slot, has at least `min_fleet_size` ships, the target slot
exists, and `source != target`.

### Discrete Targets Submitted Actions

Call:

```python
actions = DiscreteTargetActions(launch=launch, target=target, ships=ships)
obs, rewards, dones, episode_metrics = env.step(actions)
```

| Tensor | dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `launch` | `bool` | `(n_envs, 4, 44, max_per_planet_launches)` | `True` means execute a launch |
| `target` | `int64` | `(n_envs, 4, 44, max_per_planet_launches)` | target action entity slot index in `[0, 44)` |
| `ships` | `int64` | `(n_envs, 4, 44, max_per_planet_launches)` | requested ship count |

Validation matches `pure` for inactive players, source ownership, source budget,
ship count, missing source slots, and stale source slots. Launched target slots
must be in range, present, and different from the source slot.
The Kaggle conversion helper accepts the same batched model output shape and
uses the Rust target decoder to convert discrete target slots into submitted
`[from_planet_id, angle, ships]` triples for one selected player, with integer
source IDs and ship counts.

### Targeting Rules

For static planets, decoding uses the fixed target angular cone with
`target_radius - eps`, subtracts sun plus static and dynamic blocker forbidden
arcs inflated by small avoidance epsilons, then chooses the feasible angle
closest to the target centerline. Static-source/static-target selected launches
start from the cached static-safe arc and only check dynamic blockers at the
shot horizon. Other static-target selected launches recompute sun/static arcs
live, with cached static blocker geometry reused per decode. Dynamic blockers
use their cached orbit paths or comet path segments up to the selected shot's
impact horizon. If the sun removes the whole eligible arc, decoding skips
blocker search and falls back immediately. If blockers cover the whole static
target cone, selected launches fall back to the closest sun-avoiding angle.
For `"full_mask"` and `"stop_bad_launch"`, fallback rays are gated when the
first simulator-ordered intercept before the intended target is an unintended
neutral object. The decoder mirrors simulator tick order and planet-ID
collision order. Unintended neutral planet intercepts are submitted only when
the selected fleet size exceeds that planet's current ships. Unintended neutral
comet intercepts are submitted only when the selected fleet size exceeds the
comet's current ships and the comet has enough useful lifetime after capture to
produce more ships than were spent capturing it; hits on comets that expire
before combat are no-ops. Fallback intercepts with friendly planets, enemy
planets, friendly comets, or enemy comets are still submitted. Intended neutral
targets are not gated by these fallback-intercept rules.
This selected-launch fallback is separate from the `full_mask` target mask:
fully planet-blocked static targets are masked out as legal targets, but if one
is selected anyway it may still fire. If no sun-avoiding angle exists,
`"full_mask"` and `"stop_bad_launch"` decode the launch as a no-op, while
`"anything_goes"` uses the centerline.

For orbiting non-comet planets, reset caches future tick positions across the
episode horizon, and decoding solves target-hit time windows against the cached
linear segments for the selected target. Manually reconstructed states without
that cache build the selected target path lazily during decode. For comets,
decoding solves bounded target-hit time windows against each stored linear path
segment. Windows are processed in increasing impact time and capped to avoid
searching a hopeless long tail. Within each window, the decoder first tries the
window midpoint angle against sun, bounds, and cached blocker metadata. Only if
that optimistic path fails does it compute the target's eligible angular arc,
subtract sun and static/dynamic blocker forbidden arcs up to the window end
with the same avoidance epsilons, and choose the closest feasible angle for
that window. Dynamic blockers use the same orbit/comet path sources as dynamic
targets, sampled at fixed horizon fractions plus radial crossing times. If no
window has a collision-avoidance angle, decoding falls back to the first
sun-avoiding, in-bounds target arc, subject to the same neutral-intercept gate.
If no such fallback exists,
`"anything_goes"` fires along the first window midpoint and
`"full_mask"`/`"stop_bad_launch"` decode the launch as a no-op. Submitted
discrete-target launches that cannot produce an allowed or defined ray are
counted in `launch_failures_per_game`.

## Discrete Target Bins Action Spec

Config:

```python
{
    "action_spec": "discrete_target_bins",
    "min_fleet_size": 6,
    "n_bins": 11,
    "targeting_mode": "full_mask",
}
```

`ActionDiscreteTargetBinsConfig` uses the same `targeting_mode` target-slot
eligibility and bad-launch handling as `discrete_targets`, but fleet size is
selected as a categorical bin instead of an integer ship count. `n_bins` is the
total action count, including no-op, and must be at least `2`.

### Discrete Target Bins Output Tensors

| Tensor | dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `can_act` | `bool` | `(n_envs, 4, 44, 44, n_bins)` | whether a player can choose a source, target, and fleet-size bin |
| `max_launch` | omitted/`None` | n/a | not used by this action spec |

For valid source-target pairs, bin `0` is always available and decodes as
no-op. Bins `1..n_bins-1` map to:

```text
round_half_up(bin * available_ships / (n_bins - 1))
```

Bin `n_bins - 1` maps to all available source ships. Launch bins that would
produce fewer than `min_fleet_size` ships are masked. If multiple bins round to
the same ship count, only the highest bin for that ship count remains
available. For example, with `available_ships=5`, `n_bins=11`, and
`min_fleet_size=1`, the available bins are `0, 2, 4, 6, 8, 10`.

### Discrete Target Bins Submitted Actions

Call:

```python
actions = DiscreteTargetBinActions(target=target, fleet_bin=fleet_bin)
obs, rewards, dones, episode_metrics = env.step(actions)
```

where `actions` carries:

| Tensor | dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `target` | `int64` | `(n_envs, 4, 44)` | target action entity slot index in `[0, 44)` |
| `fleet_bin` | `int64` | `(n_envs, 4, 44)` | fleet-size action bin in `[0, n_bins)` |

Bin `0` ignores the target value and emits no launch. Nonzero bins validate the
selected source-target-bin tuple against `can_act`, decode the bin to a ship
count, then use the same target-to-angle decoder and launch-failure accounting
as `discrete_targets`.
The Kaggle conversion helper returns `[from_planet_id, angle, ships]` triples
with integer source IDs and ship counts.

## Decoded Launch Actions

`VectorizedEnv` can expose action masks for action specs other than the env's
construction spec, expose observations for observation specs other than the
env's construction spec, decode model actions from those specs, and step the
simulator with a common launch representation. This is intended for evaluation
code that benchmarks checkpoints trained with different observation or action
specs against each other.

```python
obs_a = env.observation_for_spec(
    checkpoint_a.config.env.obs_spec,
    checkpoint_a.config.env.action_spec,
)
obs_b = env.observation_for_spec(
    checkpoint_b.config.env.obs_spec,
    checkpoint_b.config.env.action_spec,
)
decoded_a = env.decode_actions(output_a.actions, action_spec=checkpoint_a.config.env.action_spec)
decoded_b = env.decode_actions(output_b.actions, action_spec=checkpoint_b.config.env.action_spec)
obs, rewards, dones, episode_metrics = env.step_decoded_actions(selected_decoded)
```

`DecodedLaunchActions` has fixed-rank tensors:

| Tensor | dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `valid` | `bool` | `(n_envs, 4, max_actions)` | whether this decoded launch slot should execute |
| `from_planet_id` | `int64` | `(n_envs, 4, max_actions)` | source planet ID for valid launches |
| `angle` | `float32` | `(n_envs, 4, max_actions)` | launch angle for valid launches |
| `ships` | `int64` | `(n_envs, 4, max_actions)` | ship count for valid launches |

`decode_actions(...)` chooses `max_actions` from the source action spec:
`44 * max_per_planet_launches` for `pure` and `discrete_targets`, and `44` for
`discrete_target_bins`. Callers that merge decoded actions from different specs
should pad to the larger `max_actions` and select per outer player slot.
`step_decoded_actions(...)` ignores payload fields where `valid` is false,
rejects launches from inactive outer slots, validates source ownership, finite
angles, positive `i32` ship counts, and per-source ship budgets, then uses the
normal Rust rules-engine step path. Spec-specific constraints such as
`min_fleet_size`, target eligibility, target-to-angle conversion, and target-bin
rounding are enforced during `decode_actions(...)`.

## Replay Snapshots

`VectorizedEnv.state_snapshot(env_index)` returns a JSON-serializable snapshot
of one current Rust sub-env. `VectorizedEnv.terminal_snapshot(env_index)` returns
the terminal snapshot captured during the most recent `step(...)`, or `None` if
that sub-env did not terminate on that step. Terminal snapshots are captured
after the terminal transition and before the vectorized env auto-resets the
sub-env.

`VectorizedEnv.terminal_metrics(env_index)` returns scalar metrics for that
sub-env's most recent terminal transition, or `None` if it did not terminate on
the latest step. Reset and manual truncation clear the saved terminal metrics.

Snapshots are intended for replay rendering and debugging, not model input. They
include raw board-space values: board constants, step/config fields, player
count, owner IDs remapped into outer player slots, the internal/outer player map,
outer-slot `player_finished`, action entity slot planet IDs, planets, fleets,
comet groups, and comet paths. Neutral ownership remains `-1`, and each planet
or fleet also includes `internal_owner` when the engine-owned ID is needed.

## Reward Modes

The vectorized environment supports three reward shapes, selected by the
`VectorizedEnv(..., reward_mode=...)` constructor argument or `EnvConfig`
field. Non-terminal active-player transitions still reward `0`, and a player
eliminated mid-game still receives `done=True` on the step it is eliminated.

- `"win_loss"` (default): the scheme described above for submitted actions — a
  sole winner receives `+1`, losers receive `-1`, and tied winners split the
  reward via `(1 - (winner_count - 1)) / winner_count`.
- `"win_only"`: a sole winner receives `1` at terminal and every non-winner
  receives `0`, including players eliminated before terminal game reset. Tied
  winners split one unit of winner-probability mass evenly, so each tied winner
  receives `1 / winner_count`.
- `"ship_ratio"`: each winning player (the player or players with the maximum
  total ship count) receives `winner_ships / total_player_ships`, where
  `total_player_ships` is the summed planet-and-fleet ship count across all
  active players, excluding neutral ships. Every non-winner receives `0`, with no
  `-1` loss penalty, including players eliminated before terminal game reset.
  Tied winners each receive the same ratio because they share the maximum score.
  When no active player has any ships, all rewards are `0`.

## Manual Truncation

`VectorizedEnv.truncate_envs(truncate_mask)` resets a chosen subset of sub-envs,
for training loops that impose a time limit shorter than a game's natural
termination. `truncate_mask` is a length-`n_envs` boolean array or CPU boolean
tensor; each `True` env is reset to a fresh game and its observation rows are
overwritten with the reset observation exactly as a terminal auto-reset would,
while `False` env rows are left untouched. The raw Rust
`RlVecEnv.truncate_envs(truncate_mask, ...obs buffers...)` method accepts the
same observation buffers as `reset(...)`, with `max_launch=None` for action specs
that do not expose a max-launch tensor.

Unlike a terminal step, truncation emits no rewards, `dones`, terminal metrics,
or terminal snapshot — the caller decides how to treat the truncated transition.
The intended use is value bootstrapping: read a to-be-truncated env's current
observation and evaluate the critic on it first, so the truncated trajectory can
bootstrap from the predicted value instead of a real terminal reward. Because
`truncate_envs` overwrites the observation buffers for the truncated envs,
capture any needed bootstrap inputs before calling it.

## Episode Metrics

`episode_metrics` is a `dict[str, list[float]]` populated only for sub-envs that
terminated during this step. Empty steps return `{}`. Most entries append one
value for each terminal episode to which the metric applies. Player-slot
win-rate keys omit episodes where that outer slot is inactive, and
launch-conditioned means omit launchless episodes. Loss-rate and neutral-
undershot-rate keys are singleton aggregate ratios over all terminal episodes
returned by the step.

Terminal episode metrics:

| Key | Meaning |
| --- | --- |
| `total_games_played` | Count marker emitted once per terminal episode. Python training sums this as `train/total_games_played` across the rollout instead of averaging it. |
| `max_entities_exceeded_per_game` | Count of post-step turns where active fleets exceeded `max_fleets`. |
| `game_length_mean` | Terminal game step count. |
| `full_length_rate` | `1.0` when a game reaches the configured episode horizon, otherwise `0.0`. |
| `terminal_ship_count` | Total ships on planets and in active fleets at terminal. |
| `planets_captured_per_game` | Total planet captures over the episode, counting repeat captures. |
| `comets_captured_per_game` | Total comet planet captures over the episode, counting repeat captures. |
| `_neutral_planets_captured_per_game` | Hidden aggregate input: successful neutral non-comet planet captures in the terminal episode. Used to compute neutral undershot rates; not logged directly. |
| `_neutral_comets_captured_per_game` | Hidden aggregate input: successful neutral comet captures in the terminal episode. Used to compute neutral undershot rates; not logged directly. |
| `_neutral_planet_undershots_per_game` | Hidden aggregate input: neutral non-comet planet capture undershots in the terminal episode. Used to compute neutral undershot rates; not logged directly. |
| `_neutral_comet_undershots_per_game` | Hidden aggregate input: neutral comet capture undershots in the terminal episode. Used to compute neutral undershot rates; not logged directly. |
| `neutral_planet_undershot_rate` | Neutral non-comet planet capture undershots divided by successful neutral non-comet planet captures plus those undershots. An undershot is a neutral arrival whose surviving incoming ships are less than or equal to the neutral planet ship count. Omitted when no neutral non-comet planet capture or undershot occurred. |
| `neutral_comet_undershot_rate` | Neutral comet capture undershots divided by successful neutral comet captures plus those undershots. An undershot is a neutral arrival whose surviving incoming ships are less than or equal to the neutral comet ship count. Omitted when no neutral comet capture or undershot occurred. |
| `launch_failures_per_game` | Submitted discrete-target launches skipped because no valid selected ray exists, including static targets with no sun-avoiding ray and dynamic targets with no intercept, no allowed fallback, or out-of-bounds impact points. Python training logs this as `train/launch_failures_per_game`. |
| `launches_per_game` | Total successfully decoded and executed launches in the terminal episode. |
| `launches_per_turn` | Mean launches per player per turn. |
| `fleet_size_max` | Largest fleet launched during the episode. |
| `fleet_size_min` | Smallest fleet launched during the episode, or `0.0` when no fleets launched. |
| `fleet_size_std` | Population standard deviation of launched fleet sizes during the episode. |
| `win_rate_player_0`..`win_rate_player_3` | `1.0` for a winning model-visible outer player slot, `0.0` otherwise; inactive outer slots have no value in 2-player games. |
| `launches_per_planet_mean` | Per-game mean launches per occupied non-comet planet per turn. |
| `launches_per_launch_mean` | Mean launches from a planet on planet-turns where that planet launched at least once; omitted when no launches execute. |
| `ships_per_launch_mean` | Mean ship count across successfully decoded and executed launches; omitted when none execute. |
| `ships_lost_in_combat_per_game` | Ships destroyed during fleet-vs-fleet and fleet-vs-planet combat resolution. |
| `ships_lost_per_game_mean` | Ships removed by combat, sun, or out-of-bounds fleet loss. |
| `ships_lost_in_sun_per_game_mean` | Ships removed by sun fleet loss. |
| `ships_lost_out_of_bounds_per_game_mean` | Ships removed by out-of-bounds fleet loss. |
| `ships_lost_to_sun_or_oob_rate` | `(ships_lost_in_sun_per_game_mean + ships_lost_out_of_bounds_per_game_mean) / ships_lost_per_game_mean`. Omitted when no ships were lost. |
| `fleets_lost_in_combat_per_game` | Fleets removed during planet/combat resolution. |
| `fleets_lost_per_game_mean` | Fleets removed by combat, sun, or out-of-bounds loss. |
| `fleets_lost_in_sun_per_game_mean` | Fleets removed by sun loss. |
| `fleets_lost_out_of_bounds_per_game_mean` | Fleets removed by out-of-bounds loss. |
| `fleets_lost_to_sun_or_oob_rate` | `(fleets_lost_in_sun_per_game_mean + fleets_lost_out_of_bounds_per_game_mean) / fleets_lost_per_game_mean`. Omitted when no fleets were lost. |
| `terminal_planet_occupancy_rate_2p` | Occupied non-comet planet fraction at terminal for 2-player games. |
| `terminal_planet_occupancy_rate_4p` | Occupied non-comet planet fraction at terminal for 4-player games. |

## Kaggriculture

The Kaggriculture game uses the same shared training path through its own observation and action types. The authoritative tensor contract (shapes, dtypes, pinned enums, normalization scales, padding, environment lifecycle, rewards and the information audit against the reference encoder) is `docs/kaggriculture-contract.md`. This section only summarizes how it maps onto the shared API.

- **Observation** `KaggricultureObsBatch`: every row is one seat's legal view (`[env, seat]` leading dims), so rival-private state never enters the other seat's row. Named per-entity tensors mirror `ObsBatch`: tiles (200), own/rival actors (482 slots), shops (8, ordered by `shop_slot`), market products (9), `player_features` (self, opponent) and `global_features`. Categorical fields are integer indices and exact values are `int64`/`float64` side tensors.
- **Model topology** (Isaiah's `StatelessTransformerV1` classes):
  - an `ObservationInputStem` per entity group, fed float channels plus one-hot categories
  - player tokens = `player_tokens + player_feature_proj(player_features)`
  - a global token = `global_proj(global_features)`
  - separate board-scratch, actor-plan (1) and per-player critic-value (2) token parameters
  - a `TransformerBlock` trunk with a final LayerNorm
  - an `OutputProjectionMLP` critic head per critic-value token
  - a `3D → D` actor input projection over [entity, player, plan]
  - Only input channel widths and the grammar action heads are game-specific. Seat rows are encoded independently.
- **Actions** `KaggricultureActions`: `tokens [E,2,252,12]` and `lengths [E,2]` (one unit frame per own actor, then up to `order_limits` market frames, then STOP). The slot widths and action enums are pinned in the contract; the native grammar supplies syntax/support masks.
- **Model outputs** (`KaggricultureTransformer`, Task 2.3): `forward` returns sampled `KaggricultureActions` with `log_probs.event` and `entropies.event` of shape `[E,2,252,12]` (per frame and slot; implicit slots 0/2/11 are zero), `per_player_entity = event.sum(-1)` `[E,2,252]`, zero `launch`, per-slot entropy `components`, `values [E,2]` and `winner_probabilities [E,2,2]`. `evaluate_actions` requires `int64` tokens/lengths of exactly these shapes, rejects non-canonical or out-of-support programs itself (`GrammarReplayError` naming the support, length or canonical group), and requires `hidden_state` and `dones` to be `None`.
- **Teacher targets** (`KaggricultureTeacherTargets`, Phase 4): in the observation lead layout (segment-major `[N,T,2]` in PPO), `slot_logits[k]` is `[*lead, 241, W_k]` for unit slots 1/3/4/5/6 and `[*lead, 11, W_k]` for market slots 7–10 (FP32, `finfo.min` outside the replay-conditioned mask), `winner_probabilities` is FP32 `[*lead, 2]`, and `grammar` is the teacher's `GrammarSignature`. `evaluate_actions_with_cached_teacher` returns `action_kl.event [*lead,252,12]`, `per_player_entity [*lead,252]`, zero `launch`, per-slot `components` and `target=None`; `teacher_value_cross_entropy` reduces `[*lead, 2]` distributions to `[*lead[:-1]]` per state. 102,208 B per seat row.
- **Environment**: seeds are `base_seed + rank` with stride `world_size`. Auto-reset is synchronous: on a terminal step, the observation belongs to the new game while rewards, dones and transition banks belong to the completed one. Evaluation decides winners from raw final banks (equal banks draw), and truncation keeps the transition's economic reward and bootstraps from the critic. Each evaluation seeds its games with `_evaluation_seed(base_seed, env_steps)` (`scripts/run_ppo.py`), a reproducible mix placed in `[2**62, 2**62 + 2**61)` that is distinct per evaluation step for a fixed base seed (and per base seed for a fixed step) and leaves int64 headroom for the seeds one evaluation consumes. It is not injective over `(base_seed, env_steps)` pairs, and consecutive seeds consumed by different evaluations or runs may overlap. Training seeds stay below the band only while `base_seed + rank + k * world_size < 2**62`. Enforcing any training-only band separation belongs to the training factory; it is not a native admission cap. The approved Task 1.4 native ABI accepts nonnegative i64 seeds for both training and evaluation, with checked consumption as specified below.

### Structured native observation buffers (Task 1.3)

The root `src/kaggriculture/` boundary uses the following named buffers. Every
shape starts with `[E, 2]`; each row is one legal seat perspective. The contract
is version 4, observation schema 3. Native writing is implemented. With Task
2.1's schema merged, the real `KaggricultureObsBatch.check_contract()` runs on
every binding test batch and on all 512 frozen oracle records. The optimized
phase timing still needs a pod run, which the Mac could not build.

| Fields | Scalar type | Trailing shape |
| --- | --- | --- |
| `tile_kind`, `tile_crop`, `tile_animal`, `tile_cell`, `tile_role` | int64 | `[200]` |
| `tiles_int` | int64 | `[200,7]` |
| `tiles_float` | float32 | `[200,15]` |
| `actor_slot`, `actor_cell`, `actor_role` | int64 | `[482]` |
| `actor_mask` | bool | `[482]` |
| `actor_inventory`, `actor_inventory_rank` | int64 | `[241,12]` |
| `actors_float` | float32 | `[482,26]` |
| `player_features` | float32 | `[2,44]` |
| `storage_counts` | int64 | `[17]` |
| `storage_rank` | int64 | `[12]` |
| `banks` | float64 | `[2]` |
| `shop_type`, `shop_slot` | int64 | `[8]` |
| `shop_mask` | bool | `[8]` |
| `market_product` | int64 | `[9]` |
| `market_float` | float32 | `[9,2]` |
| `market_int` | int64 | `[9,2]` |
| `global_features` | float32 | `[15]` |
| `globals_int` | int64 | `[16]` |
| `still_playing` | bool | scalar |
| `order_limits` | int64 | scalar |
| `can_act` (`action_mask.can_act` in Python) | bool | `[252]` |

`ObsBuffersMut::validate(E)` checks nonzero E, checked shape/byte products and
exact lengths before making typed row views. `ObsStaging::new(E)` allocates the
29 named vectors once; `buffers_mut()` provides serial or indexed parallel
views. `publish()` checks source/destination environment counts in release
before any copy. These native mutable references are disjoint typed slices;
NumPy admission runs on each binding call before making Rust mutable slices:
exact native dtype, complete shape, C-contiguous and aligned layout, fallible
writable borrowing, and pairwise disjoint byte ranges. Distinct Torch/NumPy base
objects do not bypass overlap checks. Publication gives no authorization to
overwrite data still in use by a reader. Task 1.4 implements lifecycle
rollback; the Task 1.5 Python adapter owns the entry fence for pinned CUDA
readers described below.

The explicit-header seam in the existing `owl.rs` extension is:

```python
encode_kaggriculture_headers_into(
    headers: str, *,
    tile_kind, tile_crop, tile_animal, tile_cell, tile_role,
    tiles_int, tiles_float, actor_slot, actor_cell, actor_role,
    actor_mask, actor_inventory, actor_inventory_rank, actors_float,
    player_features, storage_counts, storage_rank, banks,
    shop_type, shop_slot, shop_mask, market_product, market_float, market_int,
    global_features, globals_int, still_playing, order_limits, can_act,
) -> None
```

All 29 keyword arguments are caller-owned NumPy arrays with the table's dtypes
and complete `[E,2,...]` shapes; `python/owl/rs.pyi` carries their typed signature.
`headers` is a JSON array of E full engine `TraceHeader` objects, not live-step
JSON. The writer prepares every environment before the first destination write,
then writes infallibly while detached from Python. Typed borrow guards remain
alive for the call. Supplied-data errors raise `ValueError` with field and,
where applicable, environment context; all output bytes stay unchanged. The
call returns `None`, retains no caller array and allocates no replacement output
arrays. Parsing and prepared snapshots still allocate scratch storage.

`ObservationGame` owns its immutable checked config, forwards native stepping
and prepares both seat views from one public snapshot. Exact count/rank tensors,
strict engine-shaped tiles and wide intermediate arithmetic preserve admitted
facts; positive hire costs use exact integer Fibonacci products before floating
conversion. The one-shot writer always sets `still_playing=true`; Task 1.4
uses it for live and synchronously reset games while publishing completed-game
rewards, dones, transition counters and terminal records separately.
Native `check_row` is a diagnostic validator outside the write hot path. The
Python schema's range checks do not replace semantic/privacy assertions.

Snapshot acquisition and output allocation are separate costs. The engine's
public snapshot clones both private inventories internally; only the requesting
seat's private state may enter its row. A counter test proves exactly one
acquisition per both-seat encode and stable output allocations; an intentional
two-acquisition mutation fails. The fat-LTO release timing build was stopped at
53.81 seconds after sampled process-group RSS reached 1,052,393,472 bytes.
No phase costs were obtained; the optimized command is handed off for the pod in
`ops/rebuild-2026-09-29/1.3/timing.json`. Debug timings do not justify changing
the approved snapshot path.

The frozen oracle (`tests/fixtures/kaggriculture/observation-v3/`) holds 512
states and 1,024 seat rows. The pinned reference `encode_invest` recorded those
rows, and the tensor-only reconstruction matches them bitwise at all 8,176
offsets. Seeded states use policy `observation-corpus-v2`, the Claude R1
correction, and 6 non-synthetic states exceed 16 actors against quota four.
See the Task 1.3 receipt for actual checks and unresolved limits.

### Stateless native replay export (Task 7.3)

The existing `owl.rs` extension exposes two diagnostic JSON-text functions:

```python
export_kaggriculture_episode(seed_header_json: str, tape_json: str) -> str
verify_kaggriculture_episode(
    episode_json: str, captured_json: str | None = None
) -> str
```

Invalid data and divergence raise `ValueError`, naming the first JSON pointer
and transition where one applies. These functions retain no environment, seed
counter or caller buffer. JSON text preserves non-negative, arbitrary-width
integer seeds without passing through a Python/Rust float. Missing, negative,
fractional and float-typed seeds are rejected.

Every public PyO3 call checks the installed source hashes and compares the
supplied specification, full configuration and envelope against the pinned
framework. The pure Rust adapter accepts an already verified specification from
its caller; it does not load Python framework files itself.

The seed header carries `seed`, resolved `configuration`, `provenance`, the
pinned framework's `specification` and its envelope metadata. Configuration
admission reuses the v4 native validator. Provenance identifies source, engine,
schema 1 and target framework `1.32.7`. Replay constructs the engine only through
`Game::from_seed_header`; required initial/schedule/terminal-bank placeholders
are diagnostic scaffolding, never oracle answers. The tape has `complete` and
`transitions`, each with exactly two seat-ordered engine `actions` and optional
two-seat `tokens` records (`tokens`, `length`). Native `grammar::plan` and
`grammar::decode` validate tokens against the pre-transition actor count and
order limit, then compare their decoded action to the recorded action. The
grammar hire cap is derived as `turnsPerDay * maxMarketOrdersPerTurn + 1`: hands
clear at the end of each day, so this is the day's maximum `actors + hires` and
never rejects a reachable program.

Episode `steps[0]` is initial state. Submitted action `t` appears beside its
post-action observation in `steps[t+1][seat].action`. Each observation has only
that seat's `player` and `private`; the pinned framework omits shared `step`
from seat 1, while Kaggriculture's interpreter populates both copies of the five
game-shared fields. Import restores only specification-marked shared fields
from seat 0. Payload order, zero quantities, empty action entries and float64
banks survive serialization. Top-level rewards/statuses mirror the last row;
rewards are raw Kaggle rewards in Kaggle's representation: the schema default
(integer `0`) while a seat is ACTIVE, `float(money)` once it is DONE (pinned
`kaggriculture.py:963`). `info.seed` retains the consumed seed and
`configuration.seed` is null. `info.v3_native_replay` explicitly identifies
native production rather than Python-framework execution.

Verification imports, replays from the seed and re-exports. Exporter-produced
episodes require byte equality after canonical JSON reserialization. Foreign
episodes use value equality followed by object-key insertion-order comparison.
Value equality normalizes decimal spelling (`1e-05` equals `1e-5`) but keeps the
JSON number kind: an integer never equals a float (`0` differs from `0.0`).
Only these host/runtime fields are omitted from foreign comparison:

- top-level `info` other than `seed`, including the native provenance added on
  import; the framework's `toJSON` copies arbitrary host metadata;
- per-seat `info`, populated by the framework rather than the game kernel;
- per-seat `observation.remainingOverageTime`, charged by the framework from
  measured agent execution duration;
- `configuration.actTimeout` and `configuration.runTimeout`, framework time
  budgets that do not affect native game rules.

Schema-shared omission is normalized, and the observation wrapper uses schema
order; nested payload order stays significant. No public/private state, action,
seed, status or raw reward is allowlisted. A separate captured-evidence check
accepts native `StepSnapshot` evidence, whose f64 `rewards` are parsed as
floats; everything else compares strictly. It takes an `initial` snapshot, `terminal` snapshot, one post-transition bank
pair per transition (`banks`), and indexed full successor snapshots
(`snapshots: [{transition, snapshot}]`). Its report counts the evidence actually
provided; replay agreement with itself is not independent certification.

`python/owl/kaggriculture/replay_export.py` loads the specification from the
installed framework after checking its version and four pinned source hashes.
`ReplayRecorder` selects reproducibly from an explicit evaluation identity,
takes its default count from `cfg.rl.eval_replay_games`, and stratifies against
the supplied model-seat schedule. Only selected game ordinals perform recording
work. Start/transition/terminal evidence is copied before later caller writes;
completed exports have seed/action/checkpoint/seat/version custody sidecars and
episode SHA-256. Truncation and errors remain explicit non-success records.
Seeds and checkpoint identity are host metadata, never model inputs.

The native `KaggricultureEnv` binding and `_evaluate_games` recorder wiring are
not implemented here. The planned binding-dependent cases remain explicit
`pytest.skip("needs Task 1.4 binding")` tests; the two- and four-rank configs already
request eight replays, but no live eight-game evaluation is qualified by this
diagnostic API.

### Native grammar boundary (Task 1.2)

`src/kaggriculture/grammar.rs` is the single C3 implementation. It uses `std`
and `serde_json` only. `plan(actors: i64, order_limit: i64, hire_limit: i64)`
returns `Result<GrammarPlan, String>` after validating actors/hire limit in
1–241 and order limit in 1–10. A hire limit below the actor count is valid and
masks HIRE. The cursor retains its originating shape; cross-shape calls fail,
and equal-shape plans are interchangeable. It is reset for every observation.

`State::{allows, advance, write_mask}` expose checked prefix support;
`advance` returns `None` only at final STOP. `write_can_act` overwrites exactly
252 booleans with `frame < actors + order_limit + 1`, independent of early STOP.
Caller slices of the wrong size fail before writing.

`decode(&plan, &[i64], length: i64)` requires exactly 3,024 values. It validates
signed length before conversion, scans all values against their named slot
width (including tails), requires zero padding, then walks the active prefix
and renders canonical JSON. It never narrows tokens, repairs them or retains
input storage. Length 0 belongs only to synthetic inactive model rows; native
seats are live after auto-reset. `encode(&plan, &Value, &mut [i64])` accepts
exactly the three action keys and every actor in order, preserves omitted unit
quantities, explicit market zero and EMPTY positions, writes zero padding and
returns the active frame count. Encoding uses the same cursor and leaves the
output untouched on every rejection. These functions return field/frame/slot
errors; native environment step admission adds env/seat context.

Actor order is an integration invariant: frame `i`, `unit_actor i`, own
observation `actor_slot i` and engine unit `i` all mean farmer first, followed
by hands in stored order. HIRE capacity counts submitted requests at frame
completion, regardless of cash or later execution success.

`grammar_tables() -> Result<GrammarTables, String>` derives eight arrays from
admitted cursor prefixes, totaling 964 booleans:

| Field | Shape |
| --- | --- |
| `unit_kind` | `[20]` |
| `unit_item` | `[20,16]` |
| `unit_quantity_present` | `[20,2]` |
| `unit_quantity_high` | `[2,32]` |
| `unit_quantity_low` | `[2,2,32]` |
| `market_kind` | `[8]` |
| `market_item` | `[8,16]` |
| `market_quantity` | `[8,32]` |

`unit_quantity_low[present][high_is_zero]` uses true = index 1. Invalid unit
kind rows have singleton-zero placeholders; callers still apply `unit_kind`.
`market_quantity` serves both digits; extraction fails in release builds if
any admitted high digit yields a different low support. Runtime actor order,
phase, queue availability, HIRE prefix and STOP overlays remain necessary.

Task 1.4 implements `kaggriculture_grammar_tables()` returning exactly these
keys as independent C-contiguous `numpy.bool_` arrays on every call, plus
`kaggriculture_grammar_constants()` returning a tuple of
`(GRAMMAR_TABLES_VERSION=1, SLOT_NAMES, SLOT_WIDTHS)` with tuple names and widths.
Task 1.5 will check that metadata and map same-name arrays to device
`torch.bool` tensors once per model/device. Task 2.3 consumes these shapes
directly. The Python codec and `native_grammar_tables(device)` integration
remain Task 1.5; no Python grammar reconstruction or binary DFA runtime layer
is introduced here.

The root builds this source under edition 2021. Task 1.3 created the first
production root → engine dependency, unified the Serde feature graph (L4 repair
in `docs/rules-engine.md`) and retired the temporary engine include: kernel
acceptance and replay-state tests run in root integration
(`src/kaggriculture/grammar_kernel_tests.rs`). CPU checked token admission
addresses a separate indexing hazard; it does not qualify the L6 Inductor
GEMM overflow fix or CUDA/BF16 replay. Native lifecycle transactions and CPU
buffer admission have separate Task 1.4 checks below; the pinned CUDA reuse
fence remains a Task 1.5 obligation.

### Native environment lifecycle (Task 1.4)

`owl.rs.KaggricultureEnv` is implemented in root `src/kaggriculture/env.rs`,
`reward.rs`, `admission.rs` and `bindings.rs`. The merged module root remains
`src/kaggriculture/mod.rs`; it registers the class and four grammar functions
through the existing `add_to_module` route and preserves the Task 1.3 header
encoder. Vendored kernel bytes are unchanged. There is no public lifecycle
state-import API.

```python
KaggricultureEnv(
    n_envs: int, seed: int, seed_stride: int, config: str,
    reward_config: KaggricultureRewardDict, native_threads: int,
    *, hire_limit: int,
)
```

Every constructor argument is required. `config` is validated game-envelope
JSON text. `n_envs` and `native_threads` are positive; even one thread uses an
explicit native Rayon pool. `hire_limit` is the action-spec capacity, never a
cash estimate. Native has no `pin_memory` argument: Python owns and pins its
Torch allocations in Task 1.5. This constructor refinement was approved as Q1
of the Task 1.4 brief review, and contract v4.2
(`docs/kaggriculture-contract.md`) incorporates it.

The following are the 35 required keyword-only NumPy output parameters, in ABI
order. The first 29 use the observation table above. The six transition outputs
complete the common argument list:

```text
tile_kind, tile_crop, tile_animal, tile_cell, tile_role,
tiles_int, tiles_float, actor_slot, actor_cell, actor_role,
actor_mask, actor_inventory, actor_inventory_rank, actors_float,
player_features, storage_counts, storage_rank, banks,
shop_type, shop_slot, shop_mask, market_product, market_float, market_int,
global_features, globals_int, still_playing, order_limits, can_act,
rewards, dones, transition_banks_before, transition_banks_after,
transition_econ_before, transition_econ_after
```

| Transition output | NumPy dtype | Full shape |
| --- | --- | --- |
| `rewards` | float32 | `[E,2]` |
| `dones` | bool | `[E,2]` |
| `transition_banks_before`, `transition_banks_after` | float64 | `[E,2]` |
| `transition_econ_before`, `transition_econ_after` | int64 | `[E,2,32]` |

There is no `econ_counters` alias. Each method below takes exactly those 35
keywords; `buffers` in this call notation is the caller's dictionary expanded
into them, not an additional ABI argument. `python/owl/rs.pyi` lists every typed
parameter explicitly.

```python
env.observe(**buffers) -> None
env.reset(**buffers) -> None
env.step(tokens, lengths, **buffers) -> dict[str, list[float]]
env.truncate_envs(mask, **buffers) -> None
```

`tokens` is native-endian int64 `[E,2,252,12]`, `lengths` is int64 `[E,2]`, and
`mask` is bool `[E]`. Inputs may be read-only. Every supplied array must be
aligned and C-contiguous; outputs must also be writable, with exact shape and
dtype. All supplied input and output byte regions must be pairwise disjoint.
Fortran-only, strided, internally overlapping and foreign-endian storage is
rejected. Disjoint contiguous views of caller storage are accepted. Native
retains borrow guards until the call returns, keeps no array references, and
never substitutes output arrays. Callers must not access outputs or
mutate/resize inputs during a call.
They must copy values they retain across later writes.

Construction consumes exactly E seeds and prepares live games. `observe`
publishes cached current observations and transition values without consuming
seeds or taking a step; the adapter will use it immediately after allocation.
`reset` consumes E new seeds, clears terminal records, and writes all rows and
padding. It publishes zero rewards/dones, current banks in both bank outputs,
and zero counters in both counter outputs.

`step` validates every action and buffer before advancing candidates. Games,
seed reservations, terminal records, observations and transition values remain
unpublished until every worker succeeds and the return dictionary is allocated.
Expected input/engine failures raise `ValueError`; checked i64 seed extraction
or counter advancement raises `OverflowError`; caught worker panics raise
`RuntimeError`. Failure preserves every supplied output byte and every live
game, seed and terminal record. Native work releases the GIL.

A terminal transition captures completed-game rewards, dones, banks and counters
before synchronously constructing one new game for that environment. Observation
rows then describe the reset game: clock zero and live masks. Default
`episodeSteps=720` ends on transition 719; settings 1 and 2 end after one
transition. The returned dictionary always has exactly `total_games_played`,
`terminal_bank_0`, `terminal_bank_1`, `terminal_margin_0`. Its lists are empty
without completions; each completed environment contributes one entry in
ascending environment order, with `1.0` in `total_games_played`.

`truncate_envs` reserves seeds only for true mask entries, in ascending
environment order. Its separate infallible commit copies only those environments'
29 observation rows, including padding. Unselected observation bytes and all
six transition buffers remain byte-identical, including actual terminal dones.
Only selected terminal records are cleared. An all-false mask changes nothing.
There is no `clear_transition` option or artificial terminal winner. The trainer
must evaluate/copy the pre-reset bootstrap observation before truncating and
retain its economic reward; native tests establish buffer behavior, not trainer
or GPU integration.

The Task 1.5 adapter will own one persistent Torch allocation and NumPy view per
output. Before `reset`, `step` or `truncate_envs`, it must call
`torch.cuda.current_stream(transfer_device).synchronize()` exactly when the
buffers are pinned and the transfer device is CUDA. CPU or unpinned use makes
no CUDA call. Every asynchronous reader must finish or join that stream before
reuse. This fence and the adapter are not implemented by Task 1.4.

Cold diagnostics are `terminal_metrics(i)`, `state_snapshot(i)` and
`seed_state()`. A terminal record contains finite float64-valued Python floats
`bank_0`, `bank_1`, `margin_0`, Python int `episode_steps`, winner 0/1/-1 from raw
bank comparison, and independent C-contiguous int64 `[32]` arrays `econ_0` and
`econ_1`. Each read returns fresh containers. The record is `None` before a
terminal transition, after the next successful nonterminal step, or after a
selected explicit reset; failed calls preserve it. `state_snapshot(i)` returns
JSON text for diagnosis, never policy input. `seed_state()` returns the nested
tuple `(next_seed, (current_env_seed, ...))`.

Seeds, next counter and stride are i64: seed in `[0, 2**63-1]`, stride in
`[1, 2**63-1]`. Booleans, floats and strings are rejected; negative seeds raise
`ValueError`. Every consumption must also leave a representable successor, so
consuming `2**63-1` fails even at stride one. The native class never adds rank.
The factory rule is initial seed `base_seed + rank`, stride `world_size`, giving
`base_seed + rank + k * world_size`. Native tests exercise world sizes 2 and 8,
67 seeds per rank across construction and all reset routes. Rank batches run
sequentially on the Mac; this is a stream arithmetic test, not a distributed run.

`reward_config` must be a plain dict with exactly `reward_mode="win_loss"`,
`econ_shaping`, `econ_starvation_weight`, `econ_drought_weight`, `econ_cap`,
`econ_ineffective_weight`, `econ_ineffective_cap`; all six numeric coefficients
are explicit, finite, nonnegative float64 values. Active caps must sum below
one. With `W=econ_shaping`, `s=econ_starvation_weight`,
`d=econ_drought_weight`, the additional admission predicate is evaluated with
separate IEEE-754 binary64 multiplications: if `W>0`, require `econ_cap>0` and
`(W*s>0 or W*d>0)`; if `econ_ineffective_weight>0`, require
`econ_ineffective_cap>0`. Inactive components impose no positive-cap condition.
Thus `W=s=d=1e-300` is rejected when both products underflow to zero. This
per-component rule intentionally strengthens the pinned reference's admission.

For each seat, only its cumulative starvation, drought and ineffective counters
(indices 0, 1 and 2 of the 32-counter vector) enter the penalty:

```text
P(c) = min(econ_cap, W * (s*S + d*D))
     + min(econ_ineffective_cap, econ_ineffective_weight * I)
delta = P(after) - P(before)
economic_reward = delta_opponent - delta_self
terminal_scale = 1 - (econ_cap if W>0 else 0)
                   - (econ_ineffective_cap if econ_ineffective_weight>0 else 0)
```

Disabled components short-circuit. Economic reward is computed in float64 and
rounded to float32 first; on terminal steps that float32 value is promoted to
float64, the raw-bank sign times `terminal_scale` is added, then the result is
rounded to float32 again. Counter monotonicity and representability, finite
banks and final rewards are checked before publication. Full undiscounted
untruncated real-arithmetic returns telescope within [-1,1]; this does not bound
bootstrap-augmented partial returns. Python `rewards.py` and its independent
oracle belong to Task 1.5.

The root release profile enables overflow checks for `kaggriculture-engine`;
caught arithmetic panics roll the candidate batch back. The cast audit identifies
two reachable unbounded HIRE cash casts, guarded using exact Fibonacci costs
for executed hires before commit. No additional game-envelope caps are imposed.
On the pod, the release overflow test passes with the override and fails with
`--config 'profile.release.package.kaggriculture-engine.overflow-checks=false'`.
Optimized phase medians (fat LTO, one env, one thread) for a dense 241-actor
state:

| Phase | Median |
| --- | --- |
| outer clone | 159.5 µs |
| kernel step including its inner clone | 270.3 µs |
| snapshot | 171.7 µs |
| composed clone/step/prepare/write | 747.6 µs |

The same composed work on an early state takes 54.8 µs. Engine overflow checks
cost no measurable time. These are component timings, not complete-update
throughput. Receipts are in `ops/rebuild-2026-09-29/1.4/claude-review/pod-oracle/`.
Codex's Mac attempts, which stopped at the memory watchdog, are in
`ops/rebuild-2026-09-29/1.4/timing.json`.

The four cold grammar functions are also exported through `owl.rs`:

```python
kaggriculture_encode(action_json: str, actors: int, order_limit: int,
                     hire_limit: int, out: numpy.ndarray[int64]) -> int
kaggriculture_decode(tokens: numpy.ndarray[int64], length: int, actors: int,
                     order_limit: int, hire_limit: int) -> str
kaggriculture_grammar_tables() -> dict[str, numpy.ndarray[bool]]
kaggriculture_grammar_constants() -> tuple[int, tuple[str, ...], tuple[int, ...]]
```

Codec arrays have exact shape `[252,12]`; encode requires writable storage and
leaves all 3,024 cells unchanged on failure. Decode permits read-only input,
checks the full row including zero padding, and returns canonical action JSON.
Both call the existing root grammar. Constants are version 1, names
`("unit_actor", "unit_kind", "unit_target", "unit_item", "unit_quantity_present",
"unit_quantity_high", "unit_quantity", "market_kind", "market_item",
"market_quantity_high", "market_quantity", "stop")`, and widths
`(241,20,128,16,2,32,32,8,16,32,32,2)`. Cold JSON does not enter live `step`.
Qualification is tracked in `ops/rebuild-2026-09-29/1.4/`: Codex's `results.md`
and Claude's `claude-review/review.md`. The pinned reference TrainingBatch was
recorded on the pod: 16 games, seeds 17000–17015, 719 transitions each, fixture
`tests/fixtures/kaggriculture_env_reference_v1.{npz,json}`.
`tests/kaggriculture/test_env_reference.py` replays it through this native env
and requires every reward, done, bank, counter, seed and terminal record to be
bit-identical over all 11,504 transitions. It passes on the pod and on the Mac.
Inverting native `dones` makes it fail at game 0, step 0. Native environment and
grammar suites pass 387 cases (344 + 43). Full `just prepare` passes.
