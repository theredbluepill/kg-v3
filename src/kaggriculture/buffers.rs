//! Named, reusable storage for the v4 / observation-v3 tensor contract.
//!
//! Native mutable references guarantee dtype, contiguity and disjointness.
//! NumPy admission must establish those properties before creating these slices.
use rayon::prelude::*;

use super::{ObserveError, ObserveErrorKind};

pub struct ObsRowMut<'a> {
    pub tile_kind: &'a mut [i64; 200],
    pub tile_crop: &'a mut [i64; 200],
    pub tile_animal: &'a mut [i64; 200],
    pub tile_cell: &'a mut [i64; 200],
    pub tile_role: &'a mut [i64; 200],
    pub tiles_int: &'a mut [[i64; 7]; 200],
    pub tiles_float: &'a mut [[f32; 15]; 200],
    pub actor_slot: &'a mut [i64; 482],
    pub actor_cell: &'a mut [i64; 482],
    pub actor_role: &'a mut [i64; 482],
    pub actor_mask: &'a mut [bool; 482],
    pub actor_inventory: &'a mut [[i64; 12]; 241],
    pub actor_inventory_rank: &'a mut [[i64; 12]; 241],
    pub actors_float: &'a mut [[f32; 26]; 482],
    pub player_features: &'a mut [[f32; 44]; 2],
    pub storage_counts: &'a mut [i64; 17],
    pub storage_rank: &'a mut [i64; 12],
    pub banks: &'a mut [f64; 2],
    pub shop_type: &'a mut [i64; 8],
    pub shop_slot: &'a mut [i64; 8],
    pub shop_mask: &'a mut [bool; 8],
    pub market_product: &'a mut [i64; 9],
    pub market_float: &'a mut [[f32; 2]; 9],
    pub market_int: &'a mut [[i64; 2]; 9],
    pub global_features: &'a mut [f32; 15],
    pub globals_int: &'a mut [i64; 16],
    pub still_playing: &'a mut bool,
    pub order_limits: &'a mut i64,
    pub can_act: &'a mut [bool; 252],
}

impl ObsRowMut<'_> {
    /// Clear every field, including padding, before writing one prepared seat.
    pub fn clear(&mut self) {
        self.tile_kind.fill(0);
        self.tile_crop.fill(0);
        self.tile_animal.fill(0);
        self.tile_cell.fill(0);
        self.tile_role.fill(0);
        self.tiles_int.fill([0; 7]);
        self.tiles_float.fill([0.0; 15]);
        self.actor_slot.fill(0);
        self.actor_cell.fill(0);
        self.actor_role.fill(0);
        self.actor_mask.fill(false);
        self.actor_inventory.fill([0; 12]);
        self.actor_inventory_rank.fill([0; 12]);
        self.actors_float.fill([0.0; 26]);
        self.player_features.fill([0.0; 44]);
        self.storage_counts.fill(0);
        self.storage_rank.fill(0);
        self.banks.fill(0.0);
        self.shop_type.fill(0);
        self.shop_slot.fill(0);
        self.shop_mask.fill(false);
        self.market_product.fill(0);
        self.market_float.fill([0.0; 2]);
        self.market_int.fill([0; 2]);
        self.global_features.fill(0.0);
        self.globals_int.fill(0);
        *self.still_playing = false;
        *self.order_limits = 0;
        self.can_act.fill(false);
    }
}

pub struct ObsEnvMut<'a> {
    pub seats: [ObsRowMut<'a>; 2],
}

pub struct ObsBuffersMut<'a> {
    pub tile_kind: &'a mut [i64],
    pub tile_crop: &'a mut [i64],
    pub tile_animal: &'a mut [i64],
    pub tile_cell: &'a mut [i64],
    pub tile_role: &'a mut [i64],
    pub tiles_int: &'a mut [i64],
    pub tiles_float: &'a mut [f32],
    pub actor_slot: &'a mut [i64],
    pub actor_cell: &'a mut [i64],
    pub actor_role: &'a mut [i64],
    pub actor_mask: &'a mut [bool],
    pub actor_inventory: &'a mut [i64],
    pub actor_inventory_rank: &'a mut [i64],
    pub actors_float: &'a mut [f32],
    pub player_features: &'a mut [f32],
    pub storage_counts: &'a mut [i64],
    pub storage_rank: &'a mut [i64],
    pub banks: &'a mut [f64],
    pub shop_type: &'a mut [i64],
    pub shop_slot: &'a mut [i64],
    pub shop_mask: &'a mut [bool],
    pub market_product: &'a mut [i64],
    pub market_float: &'a mut [f32],
    pub market_int: &'a mut [i64],
    pub global_features: &'a mut [f32],
    pub globals_int: &'a mut [i64],
    pub still_playing: &'a mut [bool],
    pub order_limits: &'a mut [i64],
    pub can_act: &'a mut [bool],
}

pub struct ValidatedObsBuffersMut<'a> {
    buffers: ObsBuffersMut<'a>,
    n_envs: usize,
}

pub struct ObsStaging {
    n_envs: usize,
    tile_kind: Vec<i64>,
    tile_crop: Vec<i64>,
    tile_animal: Vec<i64>,
    tile_cell: Vec<i64>,
    tile_role: Vec<i64>,
    tiles_int: Vec<i64>,
    tiles_float: Vec<f32>,
    actor_slot: Vec<i64>,
    actor_cell: Vec<i64>,
    actor_role: Vec<i64>,
    actor_mask: Vec<bool>,
    actor_inventory: Vec<i64>,
    actor_inventory_rank: Vec<i64>,
    actors_float: Vec<f32>,
    player_features: Vec<f32>,
    storage_counts: Vec<i64>,
    storage_rank: Vec<i64>,
    banks: Vec<f64>,
    shop_type: Vec<i64>,
    shop_slot: Vec<i64>,
    shop_mask: Vec<bool>,
    market_product: Vec<i64>,
    market_float: Vec<f32>,
    market_int: Vec<i64>,
    global_features: Vec<f32>,
    globals_int: Vec<i64>,
    still_playing: Vec<bool>,
    order_limits: Vec<i64>,
    can_act: Vec<bool>,
}

#[derive(Clone, Copy)]
pub(super) struct FieldShape {
    pub(super) name: &'static str,
    pub(super) row_shape: &'static [usize],
    pub(super) element_bytes: usize,
}

// The only dynamic shape table. Scalars have an empty trailing shape.
pub(super) const FIELD_SHAPES: [FieldShape; 29] = [
    FieldShape {
        name: "tile_kind",
        row_shape: &[200],
        element_bytes: size_of::<i64>(),
    },
    FieldShape {
        name: "tile_crop",
        row_shape: &[200],
        element_bytes: size_of::<i64>(),
    },
    FieldShape {
        name: "tile_animal",
        row_shape: &[200],
        element_bytes: size_of::<i64>(),
    },
    FieldShape {
        name: "tile_cell",
        row_shape: &[200],
        element_bytes: size_of::<i64>(),
    },
    FieldShape {
        name: "tile_role",
        row_shape: &[200],
        element_bytes: size_of::<i64>(),
    },
    FieldShape {
        name: "tiles_int",
        row_shape: &[200, 7],
        element_bytes: size_of::<i64>(),
    },
    FieldShape {
        name: "tiles_float",
        row_shape: &[200, 15],
        element_bytes: size_of::<f32>(),
    },
    FieldShape {
        name: "actor_slot",
        row_shape: &[482],
        element_bytes: size_of::<i64>(),
    },
    FieldShape {
        name: "actor_cell",
        row_shape: &[482],
        element_bytes: size_of::<i64>(),
    },
    FieldShape {
        name: "actor_role",
        row_shape: &[482],
        element_bytes: size_of::<i64>(),
    },
    FieldShape {
        name: "actor_mask",
        row_shape: &[482],
        element_bytes: size_of::<bool>(),
    },
    FieldShape {
        name: "actor_inventory",
        row_shape: &[241, 12],
        element_bytes: size_of::<i64>(),
    },
    FieldShape {
        name: "actor_inventory_rank",
        row_shape: &[241, 12],
        element_bytes: size_of::<i64>(),
    },
    FieldShape {
        name: "actors_float",
        row_shape: &[482, 26],
        element_bytes: size_of::<f32>(),
    },
    FieldShape {
        name: "player_features",
        row_shape: &[2, 44],
        element_bytes: size_of::<f32>(),
    },
    FieldShape {
        name: "storage_counts",
        row_shape: &[17],
        element_bytes: size_of::<i64>(),
    },
    FieldShape {
        name: "storage_rank",
        row_shape: &[12],
        element_bytes: size_of::<i64>(),
    },
    FieldShape {
        name: "banks",
        row_shape: &[2],
        element_bytes: size_of::<f64>(),
    },
    FieldShape {
        name: "shop_type",
        row_shape: &[8],
        element_bytes: size_of::<i64>(),
    },
    FieldShape {
        name: "shop_slot",
        row_shape: &[8],
        element_bytes: size_of::<i64>(),
    },
    FieldShape {
        name: "shop_mask",
        row_shape: &[8],
        element_bytes: size_of::<bool>(),
    },
    FieldShape {
        name: "market_product",
        row_shape: &[9],
        element_bytes: size_of::<i64>(),
    },
    FieldShape {
        name: "market_float",
        row_shape: &[9, 2],
        element_bytes: size_of::<f32>(),
    },
    FieldShape {
        name: "market_int",
        row_shape: &[9, 2],
        element_bytes: size_of::<i64>(),
    },
    FieldShape {
        name: "global_features",
        row_shape: &[15],
        element_bytes: size_of::<f32>(),
    },
    FieldShape {
        name: "globals_int",
        row_shape: &[16],
        element_bytes: size_of::<i64>(),
    },
    FieldShape {
        name: "still_playing",
        row_shape: &[],
        element_bytes: size_of::<bool>(),
    },
    FieldShape {
        name: "order_limits",
        row_shape: &[],
        element_bytes: size_of::<i64>(),
    },
    FieldShape {
        name: "action_mask.can_act",
        row_shape: &[252],
        element_bytes: size_of::<bool>(),
    },
];

fn shape_error(field: &str, detail: impl Into<String>) -> ObserveError {
    ObserveError {
        kind: ObserveErrorKind::Shape,
        env: None,
        seat: None,
        field: field.into(),
        detail: detail.into(),
    }
}

pub(super) fn checked_lengths(n_envs: usize) -> Result<[usize; 29], ObserveError> {
    if n_envs == 0 {
        return Err(shape_error("n_envs", "expected at least one environment"));
    }
    let n_rows = n_envs
        .checked_mul(2)
        .ok_or_else(|| shape_error("n_envs", "two-seat row count overflows usize"))?;
    let mut lengths = [0; 29];
    let mut total_bytes = 0_usize;
    for (length, shape) in lengths.iter_mut().zip(FIELD_SHAPES) {
        let row_length = shape
            .row_shape
            .iter()
            .try_fold(1_usize, |n, dim| n.checked_mul(*dim))
            .ok_or_else(|| shape_error(shape.name, "row element count overflows usize"))?;
        *length = n_rows
            .checked_mul(row_length)
            .ok_or_else(|| shape_error(shape.name, "batch element count overflows usize"))?;
        let bytes = length
            .checked_mul(shape.element_bytes)
            .filter(|bytes| *bytes <= isize::MAX as usize)
            .ok_or_else(|| shape_error(shape.name, "allocation byte size exceeds isize::MAX"))?;
        total_bytes = total_bytes
            .checked_add(bytes)
            .filter(|bytes| *bytes <= isize::MAX as usize)
            .ok_or_else(|| {
                shape_error(
                    shape.name,
                    "combined allocation byte size exceeds isize::MAX",
                )
            })?;
    }
    Ok(lengths)
}

fn check_length(field_index: usize, actual: usize, expected: usize) -> Result<(), ObserveError> {
    if actual != expected {
        return Err(shape_error(
            FIELD_SHAPES[field_index].name,
            format!("expected {expected} elements, received {actual}"),
        ));
    }
    Ok(())
}

fn allocate<T: Clone + Default>(length: usize, field_index: usize) -> Result<Vec<T>, ObserveError> {
    let mut values = Vec::new();
    values.try_reserve_exact(length).map_err(|error| {
        shape_error(
            FIELD_SHAPES[field_index].name,
            format!("cannot allocate {length} elements: {error}"),
        )
    })?;
    values.resize(length, T::default());
    Ok(values)
}

fn row_field<T, const N: usize>(
    values: &mut [T],
    field_index: usize,
) -> Result<&mut [T; N], ObserveError> {
    check_length(field_index, values.len(), N)?;
    Ok(values
        .try_into()
        .expect("length checked against the fixed row width"))
}

fn row_matrix<T, const C: usize, const N: usize>(
    values: &mut [T],
    field_index: usize,
) -> Result<&mut [[T; C]; N], ObserveError> {
    check_length(field_index, values.len(), C * N)?;
    let (rows, remainder) = values.as_chunks_mut::<C>();
    assert!(remainder.is_empty(), "length checked against C * N");
    Ok(rows
        .try_into()
        .expect("length checked against the fixed row height"))
}

fn row_scalar<T>(values: &mut [T], field_index: usize) -> Result<&mut T, ObserveError> {
    check_length(field_index, values.len(), 1)?;
    Ok(&mut values[0])
}

impl<'a> ObsRowMut<'a> {
    /// View 29 flat buffers holding exactly one seat row (leading shape
    /// `[1, 1]`) as one row. Every length is checked before any write.
    pub fn from_single_row(buffers: ObsBuffersMut<'a>) -> Result<Self, ObserveError> {
        Ok(Self {
            tile_kind: row_field(buffers.tile_kind, 0)?,
            tile_crop: row_field(buffers.tile_crop, 1)?,
            tile_animal: row_field(buffers.tile_animal, 2)?,
            tile_cell: row_field(buffers.tile_cell, 3)?,
            tile_role: row_field(buffers.tile_role, 4)?,
            tiles_int: row_matrix(buffers.tiles_int, 5)?,
            tiles_float: row_matrix(buffers.tiles_float, 6)?,
            actor_slot: row_field(buffers.actor_slot, 7)?,
            actor_cell: row_field(buffers.actor_cell, 8)?,
            actor_role: row_field(buffers.actor_role, 9)?,
            actor_mask: row_field(buffers.actor_mask, 10)?,
            actor_inventory: row_matrix(buffers.actor_inventory, 11)?,
            actor_inventory_rank: row_matrix(buffers.actor_inventory_rank, 12)?,
            actors_float: row_matrix(buffers.actors_float, 13)?,
            player_features: row_matrix(buffers.player_features, 14)?,
            storage_counts: row_field(buffers.storage_counts, 15)?,
            storage_rank: row_field(buffers.storage_rank, 16)?,
            banks: row_field(buffers.banks, 17)?,
            shop_type: row_field(buffers.shop_type, 18)?,
            shop_slot: row_field(buffers.shop_slot, 19)?,
            shop_mask: row_field(buffers.shop_mask, 20)?,
            market_product: row_field(buffers.market_product, 21)?,
            market_float: row_matrix(buffers.market_float, 22)?,
            market_int: row_matrix(buffers.market_int, 23)?,
            global_features: row_field(buffers.global_features, 24)?,
            globals_int: row_field(buffers.globals_int, 25)?,
            still_playing: row_scalar(buffers.still_playing, 26)?,
            order_limits: row_scalar(buffers.order_limits, 27)?,
            can_act: row_field(buffers.can_act, 28)?,
        })
    }
}

impl<'a> ObsBuffersMut<'a> {
    pub fn validate(self, n_envs: usize) -> Result<ValidatedObsBuffersMut<'a>, ObserveError> {
        let lengths = checked_lengths(n_envs)?;
        check_length(0, self.tile_kind.len(), lengths[0])?;
        check_length(1, self.tile_crop.len(), lengths[1])?;
        check_length(2, self.tile_animal.len(), lengths[2])?;
        check_length(3, self.tile_cell.len(), lengths[3])?;
        check_length(4, self.tile_role.len(), lengths[4])?;
        check_length(5, self.tiles_int.len(), lengths[5])?;
        check_length(6, self.tiles_float.len(), lengths[6])?;
        check_length(7, self.actor_slot.len(), lengths[7])?;
        check_length(8, self.actor_cell.len(), lengths[8])?;
        check_length(9, self.actor_role.len(), lengths[9])?;
        check_length(10, self.actor_mask.len(), lengths[10])?;
        check_length(11, self.actor_inventory.len(), lengths[11])?;
        check_length(12, self.actor_inventory_rank.len(), lengths[12])?;
        check_length(13, self.actors_float.len(), lengths[13])?;
        check_length(14, self.player_features.len(), lengths[14])?;
        check_length(15, self.storage_counts.len(), lengths[15])?;
        check_length(16, self.storage_rank.len(), lengths[16])?;
        check_length(17, self.banks.len(), lengths[17])?;
        check_length(18, self.shop_type.len(), lengths[18])?;
        check_length(19, self.shop_slot.len(), lengths[19])?;
        check_length(20, self.shop_mask.len(), lengths[20])?;
        check_length(21, self.market_product.len(), lengths[21])?;
        check_length(22, self.market_float.len(), lengths[22])?;
        check_length(23, self.market_int.len(), lengths[23])?;
        check_length(24, self.global_features.len(), lengths[24])?;
        check_length(25, self.globals_int.len(), lengths[25])?;
        check_length(26, self.still_playing.len(), lengths[26])?;
        check_length(27, self.order_limits.len(), lengths[27])?;
        check_length(28, self.can_act.len(), lengths[28])?;
        Ok(ValidatedObsBuffersMut {
            buffers: self,
            n_envs,
        })
    }
}

// These checks are internal invariant checks, not fallible boundary admission.
// Every admitted field is divisible into the exact number of two-seat rows.
fn chunks<T, const N: usize>(values: &mut [T]) -> &mut [[T; N]] {
    let (rows, remainder) = values.as_chunks_mut::<N>();
    assert!(
        remainder.is_empty(),
        "validated observation chunk has a remainder"
    );
    rows
}

fn zip_exact<A: ExactSizeIterator, B: ExactSizeIterator>(a: A, b: B) -> std::iter::Zip<A, B> {
    assert_eq!(
        a.len(),
        b.len(),
        "validated observation fields have unequal row counts"
    );
    a.zip(b)
}

impl ValidatedObsBuffersMut<'_> {
    pub fn envs_mut(&mut self) -> impl Iterator<Item = ObsEnvMut<'_>> {
        let group_0 = zip_exact(
            zip_exact(
                zip_exact(
                    zip_exact(
                        zip_exact(
                            chunks::<_, 2>(chunks::<_, 200>(&mut *self.buffers.tile_kind))
                                .iter_mut(),
                            chunks::<_, 2>(chunks::<_, 200>(&mut *self.buffers.tile_crop))
                                .iter_mut(),
                        ),
                        chunks::<_, 2>(chunks::<_, 200>(&mut *self.buffers.tile_animal)).iter_mut(),
                    ),
                    chunks::<_, 2>(chunks::<_, 200>(&mut *self.buffers.tile_cell)).iter_mut(),
                ),
                chunks::<_, 2>(chunks::<_, 200>(&mut *self.buffers.tile_role)).iter_mut(),
            ),
            chunks::<_, 2>(chunks::<_, 200>(chunks::<_, 7>(
                &mut *self.buffers.tiles_int,
            )))
            .iter_mut(),
        )
        .map(
            |(((((tile_kind, tile_crop), tile_animal), tile_cell), tile_role), tiles_int)| {
                (
                    tile_kind,
                    tile_crop,
                    tile_animal,
                    tile_cell,
                    tile_role,
                    tiles_int,
                )
            },
        );
        let group_1 = zip_exact(
            zip_exact(
                zip_exact(
                    zip_exact(
                        zip_exact(
                            chunks::<_, 2>(chunks::<_, 200>(chunks::<_, 15>(
                                &mut *self.buffers.tiles_float,
                            )))
                            .iter_mut(),
                            chunks::<_, 2>(chunks::<_, 482>(&mut *self.buffers.actor_slot))
                                .iter_mut(),
                        ),
                        chunks::<_, 2>(chunks::<_, 482>(&mut *self.buffers.actor_cell)).iter_mut(),
                    ),
                    chunks::<_, 2>(chunks::<_, 482>(&mut *self.buffers.actor_role)).iter_mut(),
                ),
                chunks::<_, 2>(chunks::<_, 482>(&mut *self.buffers.actor_mask)).iter_mut(),
            ),
            chunks::<_, 2>(chunks::<_, 241>(chunks::<_, 12>(
                &mut *self.buffers.actor_inventory,
            )))
            .iter_mut(),
        )
        .map(
            |(
                ((((tiles_float, actor_slot), actor_cell), actor_role), actor_mask),
                actor_inventory,
            )| {
                (
                    tiles_float,
                    actor_slot,
                    actor_cell,
                    actor_role,
                    actor_mask,
                    actor_inventory,
                )
            },
        );
        let group_2 = zip_exact(
            zip_exact(
                zip_exact(
                    zip_exact(
                        zip_exact(
                            chunks::<_, 2>(chunks::<_, 241>(chunks::<_, 12>(
                                &mut *self.buffers.actor_inventory_rank,
                            )))
                            .iter_mut(),
                            chunks::<_, 2>(chunks::<_, 482>(chunks::<_, 26>(
                                &mut *self.buffers.actors_float,
                            )))
                            .iter_mut(),
                        ),
                        chunks::<_, 2>(chunks::<_, 2>(chunks::<_, 44>(
                            &mut *self.buffers.player_features,
                        )))
                        .iter_mut(),
                    ),
                    chunks::<_, 2>(chunks::<_, 17>(&mut *self.buffers.storage_counts)).iter_mut(),
                ),
                chunks::<_, 2>(chunks::<_, 12>(&mut *self.buffers.storage_rank)).iter_mut(),
            ),
            chunks::<_, 2>(chunks::<_, 2>(&mut *self.buffers.banks)).iter_mut(),
        )
        .map(
            |(
                (
                    (((actor_inventory_rank, actors_float), player_features), storage_counts),
                    storage_rank,
                ),
                banks,
            )| {
                (
                    actor_inventory_rank,
                    actors_float,
                    player_features,
                    storage_counts,
                    storage_rank,
                    banks,
                )
            },
        );
        let group_3 = zip_exact(
            zip_exact(
                zip_exact(
                    zip_exact(
                        zip_exact(
                            chunks::<_, 2>(chunks::<_, 8>(&mut *self.buffers.shop_type)).iter_mut(),
                            chunks::<_, 2>(chunks::<_, 8>(&mut *self.buffers.shop_slot)).iter_mut(),
                        ),
                        chunks::<_, 2>(chunks::<_, 8>(&mut *self.buffers.shop_mask)).iter_mut(),
                    ),
                    chunks::<_, 2>(chunks::<_, 9>(&mut *self.buffers.market_product)).iter_mut(),
                ),
                chunks::<_, 2>(chunks::<_, 9>(chunks::<_, 2>(
                    &mut *self.buffers.market_float,
                )))
                .iter_mut(),
            ),
            chunks::<_, 2>(chunks::<_, 9>(chunks::<_, 2>(
                &mut *self.buffers.market_int,
            )))
            .iter_mut(),
        )
        .map(
            |(
                ((((shop_type, shop_slot), shop_mask), market_product), market_float),
                market_int,
            )| {
                (
                    shop_type,
                    shop_slot,
                    shop_mask,
                    market_product,
                    market_float,
                    market_int,
                )
            },
        );
        let group_4 = zip_exact(
            zip_exact(
                zip_exact(
                    zip_exact(
                        chunks::<_, 2>(chunks::<_, 15>(&mut *self.buffers.global_features))
                            .iter_mut(),
                        chunks::<_, 2>(chunks::<_, 16>(&mut *self.buffers.globals_int)).iter_mut(),
                    ),
                    chunks::<_, 2>(&mut *self.buffers.still_playing).iter_mut(),
                ),
                chunks::<_, 2>(&mut *self.buffers.order_limits).iter_mut(),
            ),
            chunks::<_, 2>(chunks::<_, 252>(&mut *self.buffers.can_act)).iter_mut(),
        )
        .map(
            |((((global_features, globals_int), still_playing), order_limits), can_act)| {
                (
                    global_features,
                    globals_int,
                    still_playing,
                    order_limits,
                    can_act,
                )
            },
        );
        zip_exact(
            zip_exact(zip_exact(zip_exact(group_0, group_1), group_2), group_3),
            group_4,
        )
        .map(
            |(
                (
                    (
                        (
                            (tile_kind, tile_crop, tile_animal, tile_cell, tile_role, tiles_int),
                            (
                                tiles_float,
                                actor_slot,
                                actor_cell,
                                actor_role,
                                actor_mask,
                                actor_inventory,
                            ),
                        ),
                        (
                            actor_inventory_rank,
                            actors_float,
                            player_features,
                            storage_counts,
                            storage_rank,
                            banks,
                        ),
                    ),
                    (shop_type, shop_slot, shop_mask, market_product, market_float, market_int),
                ),
                (global_features, globals_int, still_playing, order_limits, can_act),
            )| {
                let [tile_kind_0, tile_kind_1] = tile_kind;
                let [tile_crop_0, tile_crop_1] = tile_crop;
                let [tile_animal_0, tile_animal_1] = tile_animal;
                let [tile_cell_0, tile_cell_1] = tile_cell;
                let [tile_role_0, tile_role_1] = tile_role;
                let [tiles_int_0, tiles_int_1] = tiles_int;
                let [tiles_float_0, tiles_float_1] = tiles_float;
                let [actor_slot_0, actor_slot_1] = actor_slot;
                let [actor_cell_0, actor_cell_1] = actor_cell;
                let [actor_role_0, actor_role_1] = actor_role;
                let [actor_mask_0, actor_mask_1] = actor_mask;
                let [actor_inventory_0, actor_inventory_1] = actor_inventory;
                let [actor_inventory_rank_0, actor_inventory_rank_1] = actor_inventory_rank;
                let [actors_float_0, actors_float_1] = actors_float;
                let [player_features_0, player_features_1] = player_features;
                let [storage_counts_0, storage_counts_1] = storage_counts;
                let [storage_rank_0, storage_rank_1] = storage_rank;
                let [banks_0, banks_1] = banks;
                let [shop_type_0, shop_type_1] = shop_type;
                let [shop_slot_0, shop_slot_1] = shop_slot;
                let [shop_mask_0, shop_mask_1] = shop_mask;
                let [market_product_0, market_product_1] = market_product;
                let [market_float_0, market_float_1] = market_float;
                let [market_int_0, market_int_1] = market_int;
                let [global_features_0, global_features_1] = global_features;
                let [globals_int_0, globals_int_1] = globals_int;
                let [still_playing_0, still_playing_1] = still_playing;
                let [order_limits_0, order_limits_1] = order_limits;
                let [can_act_0, can_act_1] = can_act;
                ObsEnvMut {
                    seats: [
                        ObsRowMut {
                            tile_kind: tile_kind_0,
                            tile_crop: tile_crop_0,
                            tile_animal: tile_animal_0,
                            tile_cell: tile_cell_0,
                            tile_role: tile_role_0,
                            tiles_int: tiles_int_0,
                            tiles_float: tiles_float_0,
                            actor_slot: actor_slot_0,
                            actor_cell: actor_cell_0,
                            actor_role: actor_role_0,
                            actor_mask: actor_mask_0,
                            actor_inventory: actor_inventory_0,
                            actor_inventory_rank: actor_inventory_rank_0,
                            actors_float: actors_float_0,
                            player_features: player_features_0,
                            storage_counts: storage_counts_0,
                            storage_rank: storage_rank_0,
                            banks: banks_0,
                            shop_type: shop_type_0,
                            shop_slot: shop_slot_0,
                            shop_mask: shop_mask_0,
                            market_product: market_product_0,
                            market_float: market_float_0,
                            market_int: market_int_0,
                            global_features: global_features_0,
                            globals_int: globals_int_0,
                            still_playing: still_playing_0,
                            order_limits: order_limits_0,
                            can_act: can_act_0,
                        },
                        ObsRowMut {
                            tile_kind: tile_kind_1,
                            tile_crop: tile_crop_1,
                            tile_animal: tile_animal_1,
                            tile_cell: tile_cell_1,
                            tile_role: tile_role_1,
                            tiles_int: tiles_int_1,
                            tiles_float: tiles_float_1,
                            actor_slot: actor_slot_1,
                            actor_cell: actor_cell_1,
                            actor_role: actor_role_1,
                            actor_mask: actor_mask_1,
                            actor_inventory: actor_inventory_1,
                            actor_inventory_rank: actor_inventory_rank_1,
                            actors_float: actors_float_1,
                            player_features: player_features_1,
                            storage_counts: storage_counts_1,
                            storage_rank: storage_rank_1,
                            banks: banks_1,
                            shop_type: shop_type_1,
                            shop_slot: shop_slot_1,
                            shop_mask: shop_mask_1,
                            market_product: market_product_1,
                            market_float: market_float_1,
                            market_int: market_int_1,
                            global_features: global_features_1,
                            globals_int: globals_int_1,
                            still_playing: still_playing_1,
                            order_limits: order_limits_1,
                            can_act: can_act_1,
                        },
                    ],
                }
            },
        )
    }
    pub fn par_envs_mut(&mut self) -> impl IndexedParallelIterator<Item = ObsEnvMut<'_>> {
        let group_0 = chunks::<_, 2>(chunks::<_, 200>(&mut *self.buffers.tile_kind))
            .par_iter_mut()
            .zip_eq(chunks::<_, 2>(chunks::<_, 200>(&mut *self.buffers.tile_crop)).par_iter_mut())
            .zip_eq(chunks::<_, 2>(chunks::<_, 200>(&mut *self.buffers.tile_animal)).par_iter_mut())
            .zip_eq(chunks::<_, 2>(chunks::<_, 200>(&mut *self.buffers.tile_cell)).par_iter_mut())
            .zip_eq(chunks::<_, 2>(chunks::<_, 200>(&mut *self.buffers.tile_role)).par_iter_mut())
            .zip_eq(
                chunks::<_, 2>(chunks::<_, 200>(chunks::<_, 7>(
                    &mut *self.buffers.tiles_int,
                )))
                .par_iter_mut(),
            )
            .map(
                |(((((tile_kind, tile_crop), tile_animal), tile_cell), tile_role), tiles_int)| {
                    (
                        tile_kind,
                        tile_crop,
                        tile_animal,
                        tile_cell,
                        tile_role,
                        tiles_int,
                    )
                },
            );
        let group_1 = chunks::<_, 2>(chunks::<_, 200>(chunks::<_, 15>(
            &mut *self.buffers.tiles_float,
        )))
        .par_iter_mut()
        .zip_eq(chunks::<_, 2>(chunks::<_, 482>(&mut *self.buffers.actor_slot)).par_iter_mut())
        .zip_eq(chunks::<_, 2>(chunks::<_, 482>(&mut *self.buffers.actor_cell)).par_iter_mut())
        .zip_eq(chunks::<_, 2>(chunks::<_, 482>(&mut *self.buffers.actor_role)).par_iter_mut())
        .zip_eq(chunks::<_, 2>(chunks::<_, 482>(&mut *self.buffers.actor_mask)).par_iter_mut())
        .zip_eq(
            chunks::<_, 2>(chunks::<_, 241>(chunks::<_, 12>(
                &mut *self.buffers.actor_inventory,
            )))
            .par_iter_mut(),
        )
        .map(
            |(
                ((((tiles_float, actor_slot), actor_cell), actor_role), actor_mask),
                actor_inventory,
            )| {
                (
                    tiles_float,
                    actor_slot,
                    actor_cell,
                    actor_role,
                    actor_mask,
                    actor_inventory,
                )
            },
        );
        let group_2 = chunks::<_, 2>(chunks::<_, 241>(chunks::<_, 12>(
            &mut *self.buffers.actor_inventory_rank,
        )))
        .par_iter_mut()
        .zip_eq(
            chunks::<_, 2>(chunks::<_, 482>(chunks::<_, 26>(
                &mut *self.buffers.actors_float,
            )))
            .par_iter_mut(),
        )
        .zip_eq(
            chunks::<_, 2>(chunks::<_, 2>(chunks::<_, 44>(
                &mut *self.buffers.player_features,
            )))
            .par_iter_mut(),
        )
        .zip_eq(chunks::<_, 2>(chunks::<_, 17>(&mut *self.buffers.storage_counts)).par_iter_mut())
        .zip_eq(chunks::<_, 2>(chunks::<_, 12>(&mut *self.buffers.storage_rank)).par_iter_mut())
        .zip_eq(chunks::<_, 2>(chunks::<_, 2>(&mut *self.buffers.banks)).par_iter_mut())
        .map(
            |(
                (
                    (((actor_inventory_rank, actors_float), player_features), storage_counts),
                    storage_rank,
                ),
                banks,
            )| {
                (
                    actor_inventory_rank,
                    actors_float,
                    player_features,
                    storage_counts,
                    storage_rank,
                    banks,
                )
            },
        );
        let group_3 = chunks::<_, 2>(chunks::<_, 8>(&mut *self.buffers.shop_type))
            .par_iter_mut()
            .zip_eq(chunks::<_, 2>(chunks::<_, 8>(&mut *self.buffers.shop_slot)).par_iter_mut())
            .zip_eq(chunks::<_, 2>(chunks::<_, 8>(&mut *self.buffers.shop_mask)).par_iter_mut())
            .zip_eq(
                chunks::<_, 2>(chunks::<_, 9>(&mut *self.buffers.market_product)).par_iter_mut(),
            )
            .zip_eq(
                chunks::<_, 2>(chunks::<_, 9>(chunks::<_, 2>(
                    &mut *self.buffers.market_float,
                )))
                .par_iter_mut(),
            )
            .zip_eq(
                chunks::<_, 2>(chunks::<_, 9>(chunks::<_, 2>(
                    &mut *self.buffers.market_int,
                )))
                .par_iter_mut(),
            )
            .map(
                |(
                    ((((shop_type, shop_slot), shop_mask), market_product), market_float),
                    market_int,
                )| {
                    (
                        shop_type,
                        shop_slot,
                        shop_mask,
                        market_product,
                        market_float,
                        market_int,
                    )
                },
            );
        let group_4 = chunks::<_, 2>(chunks::<_, 15>(&mut *self.buffers.global_features))
            .par_iter_mut()
            .zip_eq(chunks::<_, 2>(chunks::<_, 16>(&mut *self.buffers.globals_int)).par_iter_mut())
            .zip_eq(chunks::<_, 2>(&mut *self.buffers.still_playing).par_iter_mut())
            .zip_eq(chunks::<_, 2>(&mut *self.buffers.order_limits).par_iter_mut())
            .zip_eq(chunks::<_, 2>(chunks::<_, 252>(&mut *self.buffers.can_act)).par_iter_mut())
            .map(
                |((((global_features, globals_int), still_playing), order_limits), can_act)| {
                    (
                        global_features,
                        globals_int,
                        still_playing,
                        order_limits,
                        can_act,
                    )
                },
            );
        group_0
            .zip_eq(group_1)
            .zip_eq(group_2)
            .zip_eq(group_3)
            .zip_eq(group_4)
            .map(
                |(
                    (
                        (
                            (
                                (
                                    tile_kind,
                                    tile_crop,
                                    tile_animal,
                                    tile_cell,
                                    tile_role,
                                    tiles_int,
                                ),
                                (
                                    tiles_float,
                                    actor_slot,
                                    actor_cell,
                                    actor_role,
                                    actor_mask,
                                    actor_inventory,
                                ),
                            ),
                            (
                                actor_inventory_rank,
                                actors_float,
                                player_features,
                                storage_counts,
                                storage_rank,
                                banks,
                            ),
                        ),
                        (shop_type, shop_slot, shop_mask, market_product, market_float, market_int),
                    ),
                    (global_features, globals_int, still_playing, order_limits, can_act),
                )| {
                    let [tile_kind_0, tile_kind_1] = tile_kind;
                    let [tile_crop_0, tile_crop_1] = tile_crop;
                    let [tile_animal_0, tile_animal_1] = tile_animal;
                    let [tile_cell_0, tile_cell_1] = tile_cell;
                    let [tile_role_0, tile_role_1] = tile_role;
                    let [tiles_int_0, tiles_int_1] = tiles_int;
                    let [tiles_float_0, tiles_float_1] = tiles_float;
                    let [actor_slot_0, actor_slot_1] = actor_slot;
                    let [actor_cell_0, actor_cell_1] = actor_cell;
                    let [actor_role_0, actor_role_1] = actor_role;
                    let [actor_mask_0, actor_mask_1] = actor_mask;
                    let [actor_inventory_0, actor_inventory_1] = actor_inventory;
                    let [actor_inventory_rank_0, actor_inventory_rank_1] = actor_inventory_rank;
                    let [actors_float_0, actors_float_1] = actors_float;
                    let [player_features_0, player_features_1] = player_features;
                    let [storage_counts_0, storage_counts_1] = storage_counts;
                    let [storage_rank_0, storage_rank_1] = storage_rank;
                    let [banks_0, banks_1] = banks;
                    let [shop_type_0, shop_type_1] = shop_type;
                    let [shop_slot_0, shop_slot_1] = shop_slot;
                    let [shop_mask_0, shop_mask_1] = shop_mask;
                    let [market_product_0, market_product_1] = market_product;
                    let [market_float_0, market_float_1] = market_float;
                    let [market_int_0, market_int_1] = market_int;
                    let [global_features_0, global_features_1] = global_features;
                    let [globals_int_0, globals_int_1] = globals_int;
                    let [still_playing_0, still_playing_1] = still_playing;
                    let [order_limits_0, order_limits_1] = order_limits;
                    let [can_act_0, can_act_1] = can_act;
                    ObsEnvMut {
                        seats: [
                            ObsRowMut {
                                tile_kind: tile_kind_0,
                                tile_crop: tile_crop_0,
                                tile_animal: tile_animal_0,
                                tile_cell: tile_cell_0,
                                tile_role: tile_role_0,
                                tiles_int: tiles_int_0,
                                tiles_float: tiles_float_0,
                                actor_slot: actor_slot_0,
                                actor_cell: actor_cell_0,
                                actor_role: actor_role_0,
                                actor_mask: actor_mask_0,
                                actor_inventory: actor_inventory_0,
                                actor_inventory_rank: actor_inventory_rank_0,
                                actors_float: actors_float_0,
                                player_features: player_features_0,
                                storage_counts: storage_counts_0,
                                storage_rank: storage_rank_0,
                                banks: banks_0,
                                shop_type: shop_type_0,
                                shop_slot: shop_slot_0,
                                shop_mask: shop_mask_0,
                                market_product: market_product_0,
                                market_float: market_float_0,
                                market_int: market_int_0,
                                global_features: global_features_0,
                                globals_int: globals_int_0,
                                still_playing: still_playing_0,
                                order_limits: order_limits_0,
                                can_act: can_act_0,
                            },
                            ObsRowMut {
                                tile_kind: tile_kind_1,
                                tile_crop: tile_crop_1,
                                tile_animal: tile_animal_1,
                                tile_cell: tile_cell_1,
                                tile_role: tile_role_1,
                                tiles_int: tiles_int_1,
                                tiles_float: tiles_float_1,
                                actor_slot: actor_slot_1,
                                actor_cell: actor_cell_1,
                                actor_role: actor_role_1,
                                actor_mask: actor_mask_1,
                                actor_inventory: actor_inventory_1,
                                actor_inventory_rank: actor_inventory_rank_1,
                                actors_float: actors_float_1,
                                player_features: player_features_1,
                                storage_counts: storage_counts_1,
                                storage_rank: storage_rank_1,
                                banks: banks_1,
                                shop_type: shop_type_1,
                                shop_slot: shop_slot_1,
                                shop_mask: shop_mask_1,
                                market_product: market_product_1,
                                market_float: market_float_1,
                                market_int: market_int_1,
                                global_features: global_features_1,
                                globals_int: globals_int_1,
                                still_playing: still_playing_1,
                                order_limits: order_limits_1,
                                can_act: can_act_1,
                            },
                        ],
                    }
                },
            )
    }
}

impl ObsStaging {
    pub fn new(n_envs: usize) -> Result<Self, ObserveError> {
        // Check every element count and allocation byte count before allocating.
        let lengths = checked_lengths(n_envs)?;
        Ok(Self {
            n_envs,
            tile_kind: allocate(lengths[0], 0)?,
            tile_crop: allocate(lengths[1], 1)?,
            tile_animal: allocate(lengths[2], 2)?,
            tile_cell: allocate(lengths[3], 3)?,
            tile_role: allocate(lengths[4], 4)?,
            tiles_int: allocate(lengths[5], 5)?,
            tiles_float: allocate(lengths[6], 6)?,
            actor_slot: allocate(lengths[7], 7)?,
            actor_cell: allocate(lengths[8], 8)?,
            actor_role: allocate(lengths[9], 9)?,
            actor_mask: allocate(lengths[10], 10)?,
            actor_inventory: allocate(lengths[11], 11)?,
            actor_inventory_rank: allocate(lengths[12], 12)?,
            actors_float: allocate(lengths[13], 13)?,
            player_features: allocate(lengths[14], 14)?,
            storage_counts: allocate(lengths[15], 15)?,
            storage_rank: allocate(lengths[16], 16)?,
            banks: allocate(lengths[17], 17)?,
            shop_type: allocate(lengths[18], 18)?,
            shop_slot: allocate(lengths[19], 19)?,
            shop_mask: allocate(lengths[20], 20)?,
            market_product: allocate(lengths[21], 21)?,
            market_float: allocate(lengths[22], 22)?,
            market_int: allocate(lengths[23], 23)?,
            global_features: allocate(lengths[24], 24)?,
            globals_int: allocate(lengths[25], 25)?,
            still_playing: allocate(lengths[26], 26)?,
            order_limits: allocate(lengths[27], 27)?,
            can_act: allocate(lengths[28], 28)?,
        })
    }

    pub fn buffers_mut(&mut self) -> ValidatedObsBuffersMut<'_> {
        // Immutable dimensions and private vectors preserve construction admission.
        ValidatedObsBuffersMut {
            n_envs: self.n_envs,
            buffers: ObsBuffersMut {
                tile_kind: &mut self.tile_kind,
                tile_crop: &mut self.tile_crop,
                tile_animal: &mut self.tile_animal,
                tile_cell: &mut self.tile_cell,
                tile_role: &mut self.tile_role,
                tiles_int: &mut self.tiles_int,
                tiles_float: &mut self.tiles_float,
                actor_slot: &mut self.actor_slot,
                actor_cell: &mut self.actor_cell,
                actor_role: &mut self.actor_role,
                actor_mask: &mut self.actor_mask,
                actor_inventory: &mut self.actor_inventory,
                actor_inventory_rank: &mut self.actor_inventory_rank,
                actors_float: &mut self.actors_float,
                player_features: &mut self.player_features,
                storage_counts: &mut self.storage_counts,
                storage_rank: &mut self.storage_rank,
                banks: &mut self.banks,
                shop_type: &mut self.shop_type,
                shop_slot: &mut self.shop_slot,
                shop_mask: &mut self.shop_mask,
                market_product: &mut self.market_product,
                market_float: &mut self.market_float,
                market_int: &mut self.market_int,
                global_features: &mut self.global_features,
                globals_int: &mut self.globals_int,
                still_playing: &mut self.still_playing,
                order_limits: &mut self.order_limits,
                can_act: &mut self.can_act,
            },
        }
    }

    pub fn publish(&self, out: &mut ValidatedObsBuffersMut<'_>) -> Result<(), ObserveError> {
        if self.n_envs != out.n_envs {
            return Err(shape_error(
                "n_envs",
                format!(
                    "source n_envs={} differs from destination n_envs={}",
                    self.n_envs, out.n_envs
                ),
            ));
        }
        // No fallible operation after the first copy. Both sides were admitted.
        out.buffers.tile_kind.copy_from_slice(&self.tile_kind);
        out.buffers.tile_crop.copy_from_slice(&self.tile_crop);
        out.buffers.tile_animal.copy_from_slice(&self.tile_animal);
        out.buffers.tile_cell.copy_from_slice(&self.tile_cell);
        out.buffers.tile_role.copy_from_slice(&self.tile_role);
        out.buffers.tiles_int.copy_from_slice(&self.tiles_int);
        out.buffers.tiles_float.copy_from_slice(&self.tiles_float);
        out.buffers.actor_slot.copy_from_slice(&self.actor_slot);
        out.buffers.actor_cell.copy_from_slice(&self.actor_cell);
        out.buffers.actor_role.copy_from_slice(&self.actor_role);
        out.buffers.actor_mask.copy_from_slice(&self.actor_mask);
        out.buffers
            .actor_inventory
            .copy_from_slice(&self.actor_inventory);
        out.buffers
            .actor_inventory_rank
            .copy_from_slice(&self.actor_inventory_rank);
        out.buffers.actors_float.copy_from_slice(&self.actors_float);
        out.buffers
            .player_features
            .copy_from_slice(&self.player_features);
        out.buffers
            .storage_counts
            .copy_from_slice(&self.storage_counts);
        out.buffers.storage_rank.copy_from_slice(&self.storage_rank);
        out.buffers.banks.copy_from_slice(&self.banks);
        out.buffers.shop_type.copy_from_slice(&self.shop_type);
        out.buffers.shop_slot.copy_from_slice(&self.shop_slot);
        out.buffers.shop_mask.copy_from_slice(&self.shop_mask);
        out.buffers
            .market_product
            .copy_from_slice(&self.market_product);
        out.buffers.market_float.copy_from_slice(&self.market_float);
        out.buffers.market_int.copy_from_slice(&self.market_int);
        out.buffers
            .global_features
            .copy_from_slice(&self.global_features);
        out.buffers.globals_int.copy_from_slice(&self.globals_int);
        out.buffers
            .still_playing
            .copy_from_slice(&self.still_playing);
        out.buffers.order_limits.copy_from_slice(&self.order_limits);
        out.buffers.can_act.copy_from_slice(&self.can_act);
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn allocation_layout(&self) -> [(usize, usize); 29] {
        [
            (self.tile_kind.as_ptr() as usize, self.tile_kind.capacity()),
            (self.tile_crop.as_ptr() as usize, self.tile_crop.capacity()),
            (
                self.tile_animal.as_ptr() as usize,
                self.tile_animal.capacity(),
            ),
            (self.tile_cell.as_ptr() as usize, self.tile_cell.capacity()),
            (self.tile_role.as_ptr() as usize, self.tile_role.capacity()),
            (self.tiles_int.as_ptr() as usize, self.tiles_int.capacity()),
            (
                self.tiles_float.as_ptr() as usize,
                self.tiles_float.capacity(),
            ),
            (
                self.actor_slot.as_ptr() as usize,
                self.actor_slot.capacity(),
            ),
            (
                self.actor_cell.as_ptr() as usize,
                self.actor_cell.capacity(),
            ),
            (
                self.actor_role.as_ptr() as usize,
                self.actor_role.capacity(),
            ),
            (
                self.actor_mask.as_ptr() as usize,
                self.actor_mask.capacity(),
            ),
            (
                self.actor_inventory.as_ptr() as usize,
                self.actor_inventory.capacity(),
            ),
            (
                self.actor_inventory_rank.as_ptr() as usize,
                self.actor_inventory_rank.capacity(),
            ),
            (
                self.actors_float.as_ptr() as usize,
                self.actors_float.capacity(),
            ),
            (
                self.player_features.as_ptr() as usize,
                self.player_features.capacity(),
            ),
            (
                self.storage_counts.as_ptr() as usize,
                self.storage_counts.capacity(),
            ),
            (
                self.storage_rank.as_ptr() as usize,
                self.storage_rank.capacity(),
            ),
            (self.banks.as_ptr() as usize, self.banks.capacity()),
            (self.shop_type.as_ptr() as usize, self.shop_type.capacity()),
            (self.shop_slot.as_ptr() as usize, self.shop_slot.capacity()),
            (self.shop_mask.as_ptr() as usize, self.shop_mask.capacity()),
            (
                self.market_product.as_ptr() as usize,
                self.market_product.capacity(),
            ),
            (
                self.market_float.as_ptr() as usize,
                self.market_float.capacity(),
            ),
            (
                self.market_int.as_ptr() as usize,
                self.market_int.capacity(),
            ),
            (
                self.global_features.as_ptr() as usize,
                self.global_features.capacity(),
            ),
            (
                self.globals_int.as_ptr() as usize,
                self.globals_int.capacity(),
            ),
            (
                self.still_playing.as_ptr() as usize,
                self.still_playing.capacity(),
            ),
            (
                self.order_limits.as_ptr() as usize,
                self.order_limits.capacity(),
            ),
            (self.can_act.as_ptr() as usize, self.can_act.capacity()),
        ]
    }
}
