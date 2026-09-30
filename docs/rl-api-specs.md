# RL API Specs

This document describes the currently available RL observation and action specs.
The Python config API uses pydantic discriminator fields so future specs can add
different options without changing the outer `VectorizedEnv` constructor shape.
Orbit's `EnvConfig.n_envs` defaults to `2` and must be even so built-in checkpoint
evaluation can split evaluation games across 2-player and 4-player batches.
Kaggriculture accepts one or more environments and has exactly two seats.
All tensor shapes in this document are local to one vectorized environment instance.
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
- **Environment**: seeds are `base_seed + rank` with stride `world_size`. Auto-reset is synchronous: on a terminal step, the observation belongs to the new game while rewards, dones and transition banks belong to the completed one. Evaluation decides winners from raw final banks (equal banks draw), and truncation keeps the transition's economic reward and bootstraps from the critic. Each evaluation seeds its games with `_evaluation_seed(base_seed, env_steps)` (`scripts/run_ppo.py`), a reproducible mix placed in `[2**62, 2**62 + 2**61)` that is distinct per evaluation step for a fixed base seed (and per base seed for a fixed step) and leaves int64 headroom for the seeds one evaluation consumes. It is not injective over `(base_seed, env_steps)` pairs, and consecutive seeds consumed by different evaluations or runs may overlap. The trainer admits a rollout seed budget at startup so `base_seed + rank + k * world_size < 2**62` throughout the launch, and a resumed launch's `base_seed` starts past the seeds its checkpoint trained on (see the trainer seam below); this is not a native admission cap. The approved Task 1.4 native ABI accepts nonnegative i64 seeds for both training and evaluation, with checked consumption as specified below.

### Python adapter, factory and reward configuration (Task 1.5 Stage 2)

Optional Python rollout optimizations (all default off) preserve the existing
action/observation tensor schemas and reward arithmetic:

- `rl.compile_actor_heads`: compile the grammar policy core independently of
  `rl.model_compile`, using `rl.model_compile_mode` (`default` or
  `max-autotune-no-cudagraphs` only). Eager exponential draws preserve the
  original Gumbel RNG order; the compiled region receives the draws. Startup
  and every actor dispatch enforce the cuBLAS-only GEMM claim.
- `rl.rollout_packing`: the trainer scopes current host observation masks to
  each sampling forward and the final value bootstrap. Packed indices and
  lengths are computed and validated on CPU, then transferred to the device.
  This avoids GPU `nonzero`, truth and scalar readbacks while retaining exactly
  the live tokens. The host mask is cleared on context exit, including errors;
  it adds no between-turn policy state. Learner-only batches preserve flattened
  seat order. Update, teacher and truncation-bootstrap packing keep the existing
  checked path. CPU padded attention is unchanged.
- `rl.pinned_action_d2h`: preallocate pinned int64 tokens/lengths, issue both
  copies on the producer's current CUDA stream, record one event and wait for it
  immediately before the native step consumes the arrays. Storage is reused
  only after the synchronous native call returns. CPU uses the existing
  contiguous copy path. The environment's observation-buffer fence stays.
- `env.skip_reward_telemetry_validation`: skip only the duplicate Python finite
  scans in `reward_bank_mean` and `reward_margin_abs_mean`. Native reward
  admission still checks finite banks, and public reward helpers validate by
  default. Shape/dtype checks, FP64 operation order and logged values remain
  unchanged. Leave this false for the original debug validation.

False values are omitted from config serialization, preserving old config
hashes. The three RL flags are rejected for Orbit configs. No preset changes;
CPU checks do not establish CUDA parity, allocator cost or complete-update SPS.

**Stage 2 status: real native binding wired.** The adapter and cold codec call
the merged Task 1.4 extension directly. Real-binding CPU tests cover construction,
reset, legal steps, terminal metrics, diagnostics, selected-row truncation,
stable buffer identities, rollback of all 35 outputs, live seat isolation,
reference codec replay, reward admission/oracle agreement and rank seed streams.
Exact-signature fake tests retain fault injection and fence-order coverage.
The model now loads native grammar tables with strict metadata/array validation.
`run_ppo` uses the game factory for Kaggriculture rollouts and independent
evaluation environments. Both use the canonical typed observation/action mapping;
evaluation seeds are reproducible and winners follow raw terminal banks.
Kaggriculture replay export remains Task 7.3: `rl.eval_replay_games > 0` fails
at startup before any run directory, environment or model is created.
The pod DMA fence test and early 2-rank smoke remain pending; CPU tests do not
qualify either. No training or GPU run is part of Stage 2.

`owl.train.config.GameEnvConfig` remains the single config union, discriminated
by `env.obs_spec.obs_spec`; absent observation tags select Isaiah's Orbit schema.
There is no additional `game` field. `owl.game.create_env(env_config, *, n_envs,
base_seed, rank, world_size, pin_memory, transfer_device)` dispatches by validated
config class. It checks strict integer rank/world/base values and i64 addition;
Kaggriculture receives `seed=base_seed+rank`, `seed_stride=world_size`. It does
not reset after construction. Orbit receives exactly its original constructor's
`n_envs`, specs, `two_player_weight`, `reward_mode` and `pin_memory` keywords.
This factory constructs neither a model nor a trainer.

`KaggricultureEnvConfig` remains in `owl.kaggriculture.config`; `n_envs` may be
one, seed defaults to zero in `0..2**63-1`, `native_threads` is required, and
all three are strict integers excluding booleans. `config` defaults to a
`KaggricultureGameConfig`: board size 10, episode steps 720, money 3000, ten
orders, 24 turns/day, shed 100, weed chance .005, shop unlock/sell intervals 3/4,
town sell interval 24, hire multiplier 1 and empty market params. Integer counts
admit finite integral JSON floats and canonicalize them to integers; strings and
booleans reject. Counts fit i64, orders are 1..10 and orders times turns/day must
not exceed 240. Weed chance is finite/nonnegative with no upper bound of one.
Unknown fields and nonempty market params reject. Native JSON uses camel-case
aliases; only supplied non-null `actTimeout`, `runTimeout` and `seed` framework
metadata is preserved. Framework seed metadata has no lifecycle effect.
`hire_limit` is owned only by `action_spec` and is passed to native unchanged.

The twelve reward coefficients have a single definition in
`owl.kaggriculture.rewards.KaggricultureRewardConfig`; all are required, finite
and nonnegative. The existing `reward_shaping` field holds them, while the
single top-level `reward_mode` remains `win_loss`. A config written before the
own-bank term (without the three `econ_bank_*` fields, such as a pre-change
run's or BC checkpoint's sibling `config.yaml`) no longer validates, including
as `rl.teacher_init` or for a full resume; adding `econ_bank_weight: 0.0`,
`econ_bank_scale: 100000.0` and `econ_bank_cap: 0.0` migrates it and
reproduces its rewards exactly. Likewise a config written before the margin
term (without the three `econ_margin_*` fields, such as J/2's or
`hz4bpjnq`'s run-dir `config.yaml`) needs `econ_margin_weight: 0.0`,
`econ_margin_scale: 50000.0` and `econ_margin_cap: 0.0`, which reproduce its
rewards exactly. A `--load-model-weights` warm start does not read the
source's sibling config. A complete example is:

```yaml
env:
  n_envs: 2
  seed: 0
  obs_spec: {obs_spec: kaggriculture}
  action_spec: {action_spec: kaggriculture, hire_limit: 241}
  config: {episodeSteps: 720}
  native_threads: 1
  pin_memory: false
  reward_mode: win_loss
  reward_shaping:
    econ_shaping: 0.2
    econ_starvation_weight: 4.0
    econ_drought_weight: 1.0
    econ_cap: 0.25
    econ_ineffective_weight: 0.0
    econ_ineffective_cap: 0.1
    econ_bank_weight: 0.0
    econ_bank_scale: 100000.0
    econ_bank_cap: 0.0
    econ_margin_weight: 0.0
    econ_margin_scale: 50000.0
    econ_margin_cap: 0.0
```

Inactive caps may be zero; enabled caps sum strictly below one. Positive
`econ_shaping` requires positive `econ_cap` and at least one positive binary64
product `econ_shaping * econ_starvation_weight` or
`econ_shaping * econ_drought_weight`. Positive ineffective weight requires a
positive ineffective cap. Positive `econ_bank_weight` requires positive
`econ_bank_scale` and `econ_bank_cap`; with the bank weight zero, scale and cap
are only finite and nonnegative. Positive `econ_margin_weight` likewise
requires positive `econ_margin_scale` and `econ_margin_cap`. `terminal_scale`
is one minus enabled caps, including the bank and margin caps. This
per-component predicate intentionally strengthens the reference, including
underflow cases. `to_native_dict(reward_mode)` produces the thirteen exact
native keys. `bank_score` and `bank_rewards` are the float64 oracle of the
own-bank term, and `margin_score` and `margin_rewards` that of the margin term
(see the native reward below); the adapter uses them for its
`reward_bank_mean` and `reward_margin_abs_mean` step metrics. The float64 economic/terminal oracles validate shapes, finite banks and
monotonic int64 cumulative counters; only S0/D1/I2 contribute. The transition
oracle casts the economic difference to f32, promotes to f64 for terminal
addition, then casts to f32. It preserves native binary64 operation order:
overflow of the inner death sum saturates at the cap even for tiny positive
shaping, and disabled components short-circuit. Live extreme-coefficient tests
compare native rewards exactly; the recorded 16-game fixture checks the
independent oracle within its existing one-f32-ULP allowance.
Rust remains the live reward authority. Complete
719-transition synthetic paths test the telescoping bound with a sum of output
rounding ULP allowances; this does not bound bootstrapped partial returns.

`allocate_observation_buffers(n_envs, *, pin_memory)` allocates the 29 buffers
listed below, nesting `can_act` under `obs.action_mask`. The adapter allocates
six more outputs: rewards f32 `[E,2]`, dones bool `[E,2]`, transition banks before
and after f64 `[E,2]`, and transition economic counters before and after int64
`[E,2,32]`. Each has one retained C-contiguous NumPy view in `NativeArrays`.
Every native lifecycle call receives all 35 keywords explicitly. Construction
calls `observe`, not `reset`. Returned tensor/batch identities stay stable.
`truncate_envs` updates selected observation rows in place while all transition
outputs and unselected rows stay unchanged under the native contract.

`reset`, `step` and `truncate_envs` fence first: CUDA plus pinned storage calls
`torch.cuda.current_stream(transfer_device).synchronize()` before any native
write. CPU or unpinned reuse makes no CUDA call. Other-stream readers must join
the current stream before reuse; CPU readers retaining values must copy them.
Requested unavailable pinning fails explicitly. Actions must contain exact
C-contiguous CPU int64 tokens `[E,2,252,12]` and lengths `[E,2]`; truncate masks
must be C-contiguous CPU bool `[E]`. Each fresh input gets one zero-copy NumPy
view per call. There is no casting, device transfer, repair or live JSON codec.
Native failures propagate without adapter writes; the real-binding adapter test
checks byte-identical outputs after an invalid native action, complementing
Task 1.4's injected native transaction tests.

Diagnostics delegate lazily: `terminal_metrics(i)` is the stub's typed dict or
None (float banks/margin, integer steps/winner, two int64 `[32]` counter arrays),
`state_snapshot(i)` parses a JSON object and `seed_state()` returns
`tuple[int, tuple[int, ...]]`. Snapshots are never model inputs. The four cold
codec functions call native encode/decode only; actor counts come from this
seat's first 241 mask entries, order limits from its observation and hire limit
from the action spec. Batch encoding publishes a newly allocated result only
when every row succeeds. Table constants/provenance are the Task 1.2 grammar
and Task 1.4 exports, with no Python grammar fallback.

### Shared PPO trainer and policy evaluation (Task 3.1)

`PPOTrainer` keeps one collector and one PPO update loop. Observation helpers
iterate `type(obs).model_fields`, rebuild the same batch type and preserve
optional `None` fields; nested masks and action bundles dispatch through typed
unions and `isinstance` narrowing. Both `ObsBatch` and `KaggricultureObsBatch`
use those helpers for device transfer, indexing, time flattening, segment-major
batches and in-place rollout copies. Unsupported field types, action-bundle
types or incompatible batch/mask types fail explicitly, as does a rollout buffer
pairing one game's observation spec with the other game's action spec.

Kaggriculture storage preserves `KaggricultureActionMask.can_act` as bool
`[H,E,2,252]`, action tokens as int64 `[H,E,2,252,12]` and lengths as int64
`[H,E,2]`, where `H` is the rollout horizon. Observation tensors retain their
native dtypes and per-step shapes. The trainer materializes C-contiguous CPU
int64 action tensors before `env.step`; the adapter performs no implicit cast,
layout repair or device transfer. Orbit storage and action-family layouts retain
Isaiah's shapes and dtypes.

For Kaggriculture, `train/max_entities` is the largest live-actor count of one
seat (`actor_mask` per `[env, seat]`); tiles, shops and market tokens have fixed
counts and are not included. Orbit keeps its `entity_mask` count. The
`train/{n}p_rate` metrics run up to the game's seat count, so Kaggriculture logs
only `1p` and `2p`. `PPOTrainer` rejects `value_loss='winner_ce'` for a
Kaggriculture environment, because the update keeps its per-seat winner
log-probabilities unreshaped; `FullConfig` also rejects it.

Kaggriculture rollouts call `owl.game.create_env` with
`base_seed=_kaggriculture_rollout_base_seed(cfg.env.seed, start_env_steps)`
(`cfg.env.seed + 4 * start_env_steps`), the distributed rank and world size, and
`transfer_device=distributed.device`. A fresh launch starts at `cfg.env.seed`. A
resume, and a `--load-model-weights` fresh launch in any mode (which keeps
the checkpoint's `env_steps`), read that global `env_steps` before allocation
with a memory-mapped `torch.load`, so the stream starts past every seed of the
launches the checkpoint trained on under the same `cfg.env.seed`. If the
trainer's full load then returns different `env_steps`, startup raises
`RuntimeError` ("changed during startup"). A launch from step `S0` to `S1` draws
below `base + 2 * global_envs + 2 * (S1 - S0)`, and one update gives
`S1 - S0 >= global_envs`. The rule holds across repeated resumes and world-size
changes. Orbit retains its original `VectorizedEnv` constructor call. The trainer accepts Kaggriculture base seeds only in `[0, 2**61)` and
computes a conservative step budget before allocating a run. From the launch's
base, construction plus the trainer reset, up to two new seeds per global
environment step (auto-reset and truncation), and a full update of
stopping-point overshoot must keep all rollout seeds below `2**62`. A launch
that continues a checkpoint's step and leaves no budget fails at startup. The
admitted environment-step counter also remains below `2**61` for evaluation seed
hashing.
An explicit step limit beyond the safe budget fails at startup; a launch without
one uses the computed ceiling. Native/factory seed admission remains the wider
nonnegative i64 contract.

`run_ppo._evaluate_games` maps both games through those same helpers and mixes
the candidate and last-best actions by seat. Its Kaggriculture branch drives
`_create_eval_env` with `_evaluation_seed(cfg.env.seed, env_steps)`, rank 0,
world size 1 and the evaluation transfer device. The adapter's entry fence
therefore covers evaluation copies too. Raw final banks determine wins and
draws; the returned reset observation's banks never score the completed game.
Last-best teacher load/refresh uses the shared model and checkpoint path, and
teacher caches remain outside checkpoints. Replay export is Task 7.3.

#### Learner-perspective bank telemetry (W&B)

Kaggriculture PPO logs raw final banks (money, not normalized) from the
learner's side, beside and without changing the seat-ordered
`train/terminal_bank_0`, `train/terminal_bank_1`, `train/terminal_margin_0` and
the evaluation's `eval/candidate_bank`, `eval/last_best_bank`,
`eval/candidate_bank_margin`. `owl.kaggriculture.telemetry` computes them;
percentiles interpolate linearly between order statistics (`torch.quantile`).
They are telemetry only: they never enter model inputs, rewards, losses,
advantage or return normalization, or checkpoint selection (promotion keeps the
raw-bank win rate). Orbit logs none of these keys.

Training, per update over the games completed in that rollout. `PPOTrainer`
reads the native step's per-game `terminal_bank_0`/`terminal_bank_1` lists and
gathers them from every rank, so the percentiles cover the global interval.
Truncated games are not completed and contribute nothing.

| Key | Meaning |
| --- | --- |
| `train/bank_games` | Completed games in the update (the denominator); always logged, `0` when none completed. |
| `train/own_bank_{mean,p10,p50,p90}` | Each learner seat's raw final bank, both seats of every game (two values per game). |
| `train/margin_abs_{mean,p50}` | Per game, `abs(bank_0 - bank_1)`. |
| `train/winner_bank_mean`, `train/loser_bank_mean` | Higher and lower bank over decisive games; omitted when every game drew. |
| `train/draw_rate` | Fraction of games with equal banks. |

Every key but `train/bank_games` is omitted when no game completed; none is NaN.
The training margin is absolute because both seats of a self-play game are the
learner. One game contributes `+m` from one seat and `-m` from the other, so a
signed learner margin is identically zero, and `train/terminal_margin_0` only
measures seat asymmetry. `abs` measures how decisive games are.

Evaluation against last-best, per evaluation over its games, with the candidate
as "own": `eval/bank_games` (games, equal to `eval/games`) and the mean and
p10/p50/p90 of `eval/own_bank_*` (candidate bank), `eval/opponent_bank_*`
(last-best bank) and the signed `eval/margin_*` (own minus opponent). The means
equal the existing candidate metrics. Do not read `eval/margin_mean` as
`eval/margin_0`: the latter is the seat-0 mean `bank_0 - bank_1`, whichever
model held seat 0, and says nothing about the candidate. The fixed-opponent
collection below logs its own `*_vs_bot` keys; the bot's name is never in a key
or a model input, only the W&B summary label `opponent_mix/bot`.

#### Fixed-opponent collection (`env.opponent_mix`)

Owner, 2026-09-30: "OK, for fixed bot, we can use cha22 (check
~/kaggriculture-v2)." and "implement the new rewrad first before we revisit
the cha22 anchor setup." PPO can train against a fixed scripted bot instead of
mirror self-play. The feature is bot-agnostic: any key of the `opponents_rs`
registry (`owl.rs.kaggriculture_opponent_bots()`, today `starter`, `r04`,
`ecobot`, `e776`, `cha22`) is accepted. The cha22 anchor presets
`configs/kaggriculture_4rank_vs_cha22.yaml` and
`configs/kaggriculture_2rank_vs_cha22.yaml` set `{bot: cha22, fraction: 1.0}`
on the term M margin preset.

```yaml
env:
  opponent_mix: {bot: r04, fraction: 1.0}   # absent/None: pure self-play
```

- **Default.** Without `opponent_mix` the trainer, adapter and native env run
  the pre-mix path byte for byte, and the config dumps exactly as before, so
  `config.yaml` and `v3/config_sha256` are unchanged. A golden digest recorded
  on the pre-mix tree (`25412a7`) pins this
  (`tests/kaggriculture/test_opponent_mix.py`).
- **Which envs.** `fraction * env.n_envs` must be a whole number of at least
  one (no silent rounding). The first that many envs of each rank host the
  bot; the rest stay self-play.
- **Which seat.** In env `e`'s `k`-th game (`k = 0` at construction, +1 at
  every reset, truncation and auto-reset) the learner plays seat
  `(e + k) mod 2` and the bot the other. The adapter's `learner_mask`
  (`bool [E,2]`, all true without a mix) says which seats the learner plays on
  the current observation.
- **Native step.** The bot seat's transport must be the absent program:
  length 0 and all-zero tokens. Anything else fails the whole batch, with no
  state published. Inside the step's worker, a cloned `HostedSeat` controller
  computes the bot's official JSON from the pre-step snapshot, so a failed
  batch keeps the committed controller. The learner's decoded program is
  executed unchanged. A new controller starts with every new game.
  Controllers are deterministic, and no seed is consumed beyond the self-play
  stream. Observations are written exactly as in self-play and contain no
  bot identity.
- **Trainer.** The rollout forward runs on the learner's seat rows only
  (`forward_learner_rows`), so scripted seats cost no rollout forward pass.
  They are stored with the absent program, zero log-probabilities and zero
  values. The rollout buffer stores the learner mask per step. `train_iteration`
  ANDs it into the value, policy and entity masks (`_apply_learner_mask`), so
  scripted rows contribute nothing to any of these:
  - the policy, entropy and teacher-KL terms (policy weight);
  - the value and teacher-value terms (value weight);
  - advantage normalization and every denominator, including
    `train/player_step_total` and `policy_active_ratio`.

  GAE runs per seat column on that seat's own rewards. A seat switch happens
  only across a `done`, so a learned seat never bootstraps from a scripted
  row. Replay and teacher inputs mark scripted rows not playing
  (`_learner_model_view`); rows are encoded independently, so learner rows are
  unaffected. The update still encodes scripted rows: in a fraction-1.0 batch,
  half the update's rows are masked work. Stateless models only.
- **Telemetry**, per update, gathered over ranks. The self-play keys above
  cover self-play games only. Every step of a mixed batch returns
  `_terminal_learner_seat` (-1 for a self-play game), which the reducer skips.
  The fixed-opponent keys are:
  - `train/bank_games_vs_bot` (always logged);
  - with at least one game: `train/win_rate_vs_bot` (a draw scores one half,
    as in the last-best evaluation), `train/own_bank_mean_vs_bot`,
    `train/opponent_bank_mean_vs_bot` and `train/margin_mean_vs_bot`
    (own minus bot).

  Some existing keys are degenerate or include the bot's seat under a mix.
  Their logged values are unchanged; read them as follows:
  - `train/return_common_mean` and `train/return_zero_sum_abs_mean` average
    over segments where both seats are learned. At fraction 1.0 there are
    none, so they log the empty-mask mean: finite but meaningless. At a
    fraction below 1.0 they cover the self-play envs only.
  - `perf/tokens_per_second` counts the non-masked tokens of every row,
    scripted rows included. The rollout forward never encodes those rows, so
    at fraction 1.0 it overstates learner tokens (about 2x); the update does
    still encode them.
  - `train/{1,2}p_rate` counts every row's live seats, the bot's included.
    It describes games, not learner rows.
  - `train/reward_bank_mean` averages both seats' own-bank reward, the bot's
    included. It is exactly 0 while `econ_bank_weight` is 0, as under term M.
    `train/reward_margin_abs_mean` also averages both seats; the margin
    reward is antisymmetric, so the bot's rows equal the learner's.
- **Evaluation.** At every `checkpoint_freq`, after the last-best evaluation,
  `run_ppo._evaluate_against_bot` plays one game per env on the same
  evaluation worlds with every env hosting the bot. The learned seat is seat 1
  in even envs and seat 0 in odd ones, so an even `env.n_envs` covers both
  seats equally. It logs:
  - `eval/{bank_games,win_rate,own_bank_mean,opponent_bank_mean,margin_mean}_vs_bot`;
  - `eval/{bank_games,win_rate}_vs_bot_seat_{0,1}`;
  - `time/eval_vs_bot_seconds` and `perf/eval_vs_bot_sps`.

  Promotion still reads `eval/win_rate_against_last_best` alone, and the
  last-best evaluation env never hosts the bot. The attribution (own bank =
  the learned seat's terminal bank, in both seats) is pinned by
  `test_fixed_bot_evaluation_attributes_banks_to_the_learned_seat`.

  Both evaluations run on rank 0 before `broadcast_object`, while the other
  ranks wait in that collective under the default NCCL process-group timeout
  (10 minutes; `python/owl/train/distributed.py` sets none). The self-play
  run's last-best evaluation alone paused training about 33 s at 10M steps.
  The fixed-bot evaluation (`env.n_envs` full games against the bot) adds an
  unmeasured amount. Check `time/eval_seconds + time/eval_vs_bot_seconds`
  at the first checkpoint interval against that timeout.
- **Stateless policy.** The bot key reaches only collection bookkeeping and the
  W&B summary labels `opponent_mix/bot` and `opponent_mix/fraction`. It never
  reaches observations, embeddings, heads, losses, rewards, normalization,
  checkpoints or checkpoint selection. The bot keeps its own scripted state.
- **Build.** The controllers are the root crate's default Cargo feature
  `fixed-opponents`. The Kaggle submission build (`--no-default-features`)
  drops them, because EcoBot and E776 carry no redistribution license. There
  `kaggriculture_opponent_bots()` is empty and every bot key is refused.
- **Limits.** Bot behaviour is qualified against the original Python
  submissions at the default game configuration only; shorter test
  configurations run the same controllers unqualified. Nothing has been
  trained with a mix yet. Cha22 is a registry key since the Track B import
  merged; it is parity-qualified by a light three-game corpus against Starter
  (`docs/rules-parity-coverage.md`), and its stepping throughput in training
  is unmeasured.

#### Staggered game phases (`rl.initial_stagger`) and the credit window

Owner, 2026-09-30, of what a per-player critic does not fix: "- Lockstep game
phases and the short credit window for long-payback investments. for sure."
Every Kaggriculture game lasts `episodeSteps - 1` = 719 transitions and all
envs reset together, so without a stagger each 64-step rollout trains one game
phase and only about one rollout in eleven contains a game end.

```yaml
rl:
  initial_stagger: true   # default false; omitted from the dump when false
```

- **Offsets.** `owl.train.ppo.initial_stagger_steps(seed, rank, n_envs,
  episode_steps)` gives env `i` of `rank` one uniform draw `u` from
  `1..episode_steps - 1`, seeded by `(env.seed, tag, rank * n_envs + i)`. It
  depends only on the global env index, not on the world size, so ranks draw
  different offsets. `run_ppo` passes the result to `PPOTrainer` as
  `initial_stagger=` (an `InitialStagger`). The trainer refuses the config
  without the offsets and the offsets without the config.
- **Seam.** The stagger reuses the stateless truncation path
  (`_apply_truncation`). Every env's first game is flagged, and its cut step is
  `u`, not `rl.truncation_step`. The critic evaluates the cut state, and
  `_cut_truncated_envs_` ends the trajectory there (done, `truncated`,
  `bootstrap_values`; the transition's economic reward is kept). Then
  `truncate_envs` resets the env. A new game is never flagged, so later games
  run to their natural end. `u = 719` coincides with the natural end, so that
  env's first game completes normally; the phase is uniform over all 719
  residues.
- **Metrics.** A cut publishes no transition metrics (the native truncate path
  commits no `TransitionCache`), so a cut game adds nothing to `train/bank_games`,
  `train/own_bank_*` or `train/total_games_played`. Per update, gathered over
  ranks, and only under the stagger:
  - `train/game_phase_frac_{0..5}`: the fraction of acted env observations whose
    game step lies in each sixth of a game (120 turns, five in-game days);
  - `train/game_ends`: completed games;
  - `train/stagger_cuts`: first games cut.
- **Rules.** Kaggriculture only (`FullConfig` refuses Orbit), `episodeSteps >= 2`,
  a stateless model, and no combination with `rl.truncation_prob` or
  `rl.truncation_step`.
- **Resume.** A resumed launch restarts every env at step 0, and the same
  offsets desynchronize it again. The step counters are not checkpointed.
- **Stateless policy.** Offsets and step counters live in the trainer. They
  never reach observations, embeddings, heads, losses, rewards or checkpoints.
  A cut resets the env like any truncation.

The credit window is configuration only: with `rl.gamma: 1.0` and
`rl.gae_lambda: 1.0`, `compute_gae` returns the Monte Carlo return to the first
`done` in the segment. At a cut it adds the cut's bootstrap; without a `done` it
adds the segment-end critic value (`last_values`). The presets
`configs/kaggriculture_{4,2}rank_bank_critic_credit.yaml` set `horizon: 256`,
`gae_lambda: 1.0` and `initial_stagger: true` with
`model: kaggriculture_critic_offset` (the per-seat critic offset below), since
at λ = 1 the critic enters only through the segment-end and cut bootstraps. They scale per-rank `env.n_envs`
to 16 (4 ranks) or 32 (2 ranks) and `rl.segments_per_minibatch` to 1 or 2.
This keeps 16,384 global env steps and 16 optimizer steps per iteration, and
the same per-rank minibatch and teacher forward rows. The rollout forward runs
256 times per iteration on a quarter of the rows. Rank 0's last-best evaluation
plays `env.n_envs` games, a quarter of before. Both costs are unmeasured.

### Per-seat critic offset (`model.critic_offset`)

Owner, 2026-09-30: "per-player critic might be the way out?". With
`model.critic_offset: true` the Kaggriculture value of seat row `s` is
`V_s = 2 p_s(self) − 1 + o_s`. `o_s` is a scalar from `critic_offset_head`, an
`OutputProjectionMLP(trunk, 1)` on the row's own (self) critic-value token only:
no opponent token, no other row and no opponent identity. Its output layer is
zero-initialized, so at step 0 every value, action and log-probability equals
the model without the head bit for bit. A non-live row's offset is exactly 0.
The flag defaults to false: no head is built, `ModelOutput.value_offsets` is
`None`, and the config dumps without the two fields, so the config hash, native
env, trainer and model outputs are byte-identical to 3e89425
(`ops/critic-offset-2026-09-30/`).

- **Where the sum is used.** `_values` returns the sum, so the rollout values
  for GAE, the truncation bootstrap, `last_values`, the value loss and
  `train/explained_variance` all use it. Kaggriculture's value loss is MSE
  (`winner_ce` is refused), and MSE fits the sum against the GAE return, so the
  offset is not bounded to (−1, 1).
- **Teacher.** Value distillation (`teacher_value_coef`) stays a CE between the
  student's and the teacher's winner softmaxes; the offset is outside it and is
  trained by the value loss alone. The last-best teacher is built from the
  student config (head included), and its targets are still winner
  probabilities, so the CE stays well-defined whether or not either side has a
  head. MSE sees only the sum; the winner part is additionally pulled toward
  the teacher's (zero-sum) winner distribution, and the offset takes the rest.
- **`critic_offset_detach_trunk: true`** (requires `critic_offset`): the head
  reads the critic token detached, so the offset's value gradient reaches the
  head only; the trunk still gets the winner part's gradient. The head's
  gradient still enters the global `max_grad_norm` clip, so when clipping binds
  it shrinks the shared (trunk, actor, winner) update; a detached arm is
  therefore not free of the clip-share part of gradient interference (measured
  on a tiny model at clip 1e-3: shared Σ|Δ| −21%).
- **RNG stream.** Building and initialising the head draws from the global
  RNG, so a head run and a flag-off run at the same seed do not share sampled
  trajectories. The step-0 identity is of deterministic outputs on fixed
  weights.
- **Telemetry** (only with the head): the rollout also stores each value's
  offset. `train/value_offset_mean` and `train/value_offset_abs_mean` are masked
  over the value mask. `train/ev_common` is the explained variance of the
  common-mode return (the mean of both seats' GAE returns) by the mean of both
  seats' offsets, over steps where both seats are trained. It is omitted when
  there is no such step (`env.opponent_mix` at fraction 1.0). Fixed-opponent
  learner rows scatter their offsets like their values, with zero on scripted
  rows.
- **Checkpoint loading.** `load_model_state_dict_allowing_lora` (every
  `--load-model-weights` mode, `rl.teacher_init`, last-best and the resume's
  last-best) accepts a
  checkpoint that omits exactly all `critic_offset_head.*` keys. It then zeroes
  the head's output layer; the hidden layer keeps its fresh initialization. Any
  other missing key, a partial head, or an unexpected key still fails. A head
  checkpoint loaded into a model without the head fails on its unexpected
  `critic_offset_head.*` keys. `model_and_optimizer` from a headless checkpoint
  fails, because the head adds optimizer parameters, so warm starts use
  `model_only`. The resume model itself loads strictly, so a headless resume
  into a head config fails.
- **Kaggle agent.** The policy never reads the head (actions and
  log-probabilities are unchanged by its weights). The packaging must build the
  model from the checkpoint's own `config.yaml`, which records
  `critic_offset`, so the head's keys load strictly; the Orbit `Agent` does not
  load Kaggriculture checkpoints yet (Task 7.4).
- **Preset.** `configs/kaggriculture_4rank_bank_critic.yaml` is the margin preset
  with the owner's reward (term A .25 / 150,000 / .25, term M .25 / 100,000 /
  .25, `econ_shaping` 0, so `terminal_scale` .5) and
  `model: kaggriculture_critic_offset`.

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

The native lifecycle reuses its private `ObsStaging` across successful steps.
Each worker clears and rewrites its two seat rows, including all padding. Raw
transport admission, terminal prediction and grammar decoding run per environment
in the existing Rayon pool. Error precedence remains raw transport, seed
reservation, grammar, then engine work, with errors selected in environment order.
The engine returns one transactional game clone through
`stepped_with_market_metrics`; the committed game is unchanged until publication.
After all workers and Python return allocation succeed, commit checks dimensions,
copies the 29 fields into disjoint caller-owned environment slices in parallel,
and replaces/drops each old game and snapshot on its worker. It returns staging
to the next step. A discarded pending batch causes a replacement scratch allocation
on the next step, with no committed state change. Reset/truncation keep selected-row
publication and transition semantics. Preparation and commit retain their GIL
release, and the pinned-buffer entry fence and NumPy borrow guards are unchanged.
This is the sole native path; no runtime selector or tensor/config change is added.

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
Task 1.5's `native_grammar_tables(device="cpu")` checks constants first:
exact version 1, slot names and widths, then exactly eight keys with the specified
shapes, NumPy bool dtype and C layout. It maps same-name arrays through
`grammar_tables_from_arrays(arrays, device=device)` once per model construction;
registered buffers follow model device transfers. Task 2.3 consumes these shapes
directly. The model default uses native tables; explicit table injection remains,
and `expected_grammar_tables` is only a test oracle. Missing bindings or malformed
metadata/arrays raise, with no fallback or Python grammar reconstruction.

The root builds this source under edition 2021. Task 1.3 created the first
production root → engine dependency, unified the Serde feature graph (L4 repair
in `docs/rules-engine.md`) and retired the temporary engine include: kernel
acceptance and replay-state tests run in root integration
(`src/kaggriculture/grammar_kernel_tests.rs`). CPU checked token admission
addresses a separate indexing hazard; it does not qualify the L6 Inductor
GEMM overflow fix or CUDA/BF16 replay. Native lifecycle transactions and CPU
buffer admission have separate Task 1.4 checks below; the pinned CUDA reuse
fence is implemented in Task 1.5; its real DMA proof remains pod-only.

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
    *, hire_limit: int, opponent_bot: str | None = None, opponent_envs: int = 0,
)
```

Every constructor argument but the fixed-opponent pair is required;
`opponent_bot`/`opponent_envs` are given together or not at all (see
"Fixed-opponent collection" below). `config` is validated game-envelope
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
`terminal_bank_0`, `terminal_bank_1`, `terminal_margin_0`, plus
`terminal_learner_seat` exactly when the env hosts a fixed opponent. Its lists
are empty without completions; each completed environment contributes one entry
in ascending environment order, with `1.0` in `total_games_played`.

`truncate_envs` reserves seeds only for true mask entries, in ascending
environment order. Its separate infallible commit copies only those environments'
29 observation rows, including padding. Unselected observation bytes and all
six transition buffers remain byte-identical, including actual terminal dones.
Only selected terminal records are cleared. An all-false mask changes nothing.
There is no `clear_transition` option or artificial terminal winner. The trainer
must evaluate/copy the pre-reset bootstrap observation before truncating and
retain its economic reward; native tests establish buffer behavior, not trainer
or GPU integration.

The Task 1.5 adapter owns one persistent Torch allocation and NumPy view per
output (see the adapter section above). Before `reset`, `step` or `truncate_envs`, it must call
`torch.cuda.current_stream(transfer_device).synchronize()` exactly when the
buffers are pinned and the transfer device is CUDA. CPU or unpinned use makes
no CUDA call. Every asynchronous reader must finish or join that stream before
reuse. Task 1.5 implements this fence and the adapter; Task 1.4 does not.

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
`econ_ineffective_weight`, `econ_ineffective_cap`, `econ_bank_weight`,
`econ_bank_scale`, `econ_bank_cap`, `econ_margin_weight`, `econ_margin_scale`,
`econ_margin_cap`; all twelve numeric coefficients are explicit, finite,
nonnegative float64 values. Active caps, including the bank cap when
`econ_bank_weight>0` and the margin cap when `econ_margin_weight>0`, must sum
below one (summed as death, then ineffective, then bank, then margin). Positive
`econ_bank_weight` requires `econ_bank_scale>0` and `econ_bank_cap>0`; positive
`econ_margin_weight` requires `econ_margin_scale>0` and `econ_margin_cap>0`.
With `W=econ_shaping`, `s=econ_starvation_weight`,
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
B(bank) = min(econ_bank_cap, econ_bank_weight * max(0, bank) / econ_bank_scale)
M(d) = clamp(econ_margin_weight * d / econ_margin_scale,
             -econ_margin_cap, econ_margin_cap)
economic_reward = delta_opponent - delta_self
                + (B(bank_self_after) - B(bank_self_before) if econ_bank_weight>0)
                + (M(bank_self_after - bank_opp_after)
                   - M(bank_self_before - bank_opp_before) if econ_margin_weight>0)
terminal_scale = 1 - (econ_cap if W>0 else 0)
                   - (econ_ineffective_cap if econ_ineffective_weight>0 else 0)
                   - (econ_bank_cap if econ_bank_weight>0 else 0)
                   - (econ_margin_cap if econ_margin_weight>0 else 0)
```

The own-bank term (owner decision 2026-09-30, "term A") pays each seat the
change of its own capped bank score and nothing to the rival, so it is not
zero-sum. `bank_before` is the bank of the state the action was taken in (the
reset bank on a game's first transition, read before the step) and
`bank_after` the completed transition's, before any auto-reset, so no delta
spans two games and a complete game's term telescopes to `B(final) − B(reset)`.
`B` is evaluated as product, then quotient, then cap; an overflowing product
saturates at the cap. With `econ_bank_weight = 0` nothing is added, so rewards
are bit-identical to the relative-only reward (the recorded 16-game fixture
replays bit-exactly). The bank presets use weight .25, scale 100,000 and cap
.25, so `B = .25 · min(1, bank / 100,000)`: 3,000 scores .0075, 70,000 scores
.175 and `B` saturates at a bank of 100,000. The values were proposed by the
agent, not given by the owner (see the term A Decision).
`configs/kaggriculture_4rank_bank_critic.yaml` (owner, 2026-09-30) uses weight
.25, scale 150,000 and cap .25 (saturating at a bank of 150,000) beside term M
at .25 / 100,000 / .25, with the per-seat critic offset on; the numbers were
proposed by the agent and approved by the owner ("OK go ahead.").

The cash-difference (margin) term (owner decision 2026-09-30, "term M": "0.5
Cash Diff (add this in) + 0.5 (Terminal loss 1/-1/0)") pays each seat the
change of its clamped margin score `M(bank_self − bank_opp)` over the
transition. It uses the same before/after banks as term A, so no delta spans
two games. `M` is evaluated as product, then quotient, then clamp to
`[−c_m, c_m]`; an overflowing product or quotient saturates at the cap, and
`M` is odd in the margin bit for bit, so the two seats' increments are exact
negations: the term is zero-sum, so unlike term A it has no common-mode part
the zero-sum winner critic cannot represent (with term A off,
`train/return_common_mean` stays 0). Both farms reset to the same
`startingMoney`, so a complete game's term telescopes to `M(final margin)`, and
with `terminal_scale = 1 − c_m` (every other term off) a whole game's return
from the reset state is `M(final margin) + (1 − c_m) · sign(final margin)`,
within [−1, 1]. The magnitude is not bounded that way mid-game. The critic
predicts return-to-go, `M(final margin) − M(margin_t) + terminal_scale ·
sign(final margin)`, and because `M` is signed that spans
`[−(2 c_m + terminal_scale), 2 c_m + terminal_scale]`, which is [−1.5, 1.5] for
the margin preset. That is outside the critic's `2p − 1 ∈ (−1, 1)`: for example
a seat trailing by 30k that wins by 25k has a return-to-go of +1.05. The value
fit is therefore biased in states with a large lead or deficit that later
reverses. Term M is the first term with a signed potential, so it is the first
to break the bound; the effect is unmeasured (independent review r1,
`ops/reward-margin/review-r1.md`). With
`econ_margin_weight = 0` nothing is added, so rewards are bit-identical to the
reward without it. In the economic sum it follows the bank term (relative,
then bank, then margin, in float64) before the single f32 rounding. Besides
the bank-critic preset above (term M at .25 / 100,000 / .25), only
`configs/kaggriculture_4rank_margin.yaml` and the cha22 anchor presets
(`configs/kaggriculture_{4,2}rank_vs_cha22.yaml`, the same reward against the
fixed bot) enable it: weight .5, scale 50,000
and cap .5 with `econ_shaping` 0 and term A off, so `terminal_scale` is .5 and
`M = .5 · clamp(margin / 50,000, −1, 1)`, linear up to a 50k margin. The owner
gave the .5/.5 split; the 50,000 scale was proposed by the agent from the live
run `hz4bpjnq`'s self-play margins (mean |margin| 10–18k), not given by the
owner or measured optimal (see the term M Decision). The adapter's per-step
`reward_margin_abs_mean` (the mean absolute margin component over every seat;
its signed mean is zero) recomputes the float64 oracle when the term is on and
is exactly 0 with it off.

Disabled components short-circuit. Economic reward (relative penalties plus
any bank and margin terms) is computed in float64 and rounded to float32 first; on terminal
steps that float32 value is promoted to float64, the raw-bank sign times
`terminal_scale` is added, then the result is rounded to float32 again.
Non-terminal steps publish the float32 economic value unchanged, including a
`-0.0` from an underflowing negative sum; the Python oracle adds the terminal
term only on terminal steps, so its bits, including the sign of zero, match
native. Counter monotonicity and representability, finite banks and final
rewards are checked before publication. Full undiscounted
untruncated real-arithmetic returns telescope within [-1,1] because the active
caps sum below one; this does not bound bootstrap-augmented partial returns.
The winner critic (`2p − 1` per seat row) is a softmax over (self, opponent)
within each row. The two seats' rows are separate softmaxes, so their values
sum to zero only when the critic is consistent across the two views, as winner
cross-entropy and teacher value distillation train it; its hard limits are the
(−1, 1) range and that winner semantic. It therefore cannot represent the
common-mode (mean over both seats) part of the bank term's return without
distorting its winner probabilities; `model.critic_offset` (above) adds a
per-seat offset for that part. Without the head this is a known, unmeasured
limit:
the trainer logs `train/reward_bank_mean` (the per-update mean own-bank
component per seat-step), `train/return_common_mean` (the mean over segments
of both seats' mean segment return, exactly 0 with the term off) and its
zero-sum counterpart `train/return_zero_sum_abs_mean` (the mean over the same
segments of `|R_0 − R_1| / 2`, the part the critic can represent) to measure it.
The adapter's per-step `reward_bank_mean` recomputes the float64 oracle over
`[n_envs, 2]` whenever the term is on; that cost is unmeasured against the
rollout step time (it is skipped, and exactly 0, with the term off). Python `rewards.py` and its independent
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
