//! Transactional Python/NumPy boundary. No Python objects enter detached work.
use kaggriculture_engine::Config;
use numpy::{
    Element, IntoPyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyReadwriteArrayDyn,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyOverflowError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyInt, PyTuple};

use super::buffers::{ObsBuffersMut, ValidatedObsBuffersMut};
use super::env::{EnvError, NativeEnv, TransitionBuffersMut};
use super::grammar;
use super::reward::{RewardConfig, RewardMode};

fn python_error(error: EnvError) -> PyErr {
    match error {
        EnvError::Value(message) => PyValueError::new_err(message),
        EnvError::Overflow(message) => PyOverflowError::new_err(message),
        EnvError::Panic(message) => PyRuntimeError::new_err(message),
    }
}

const NAMES: [&str; 39] = [
    "tile_kind",
    "tile_crop",
    "tile_animal",
    "tile_cell",
    "tile_role",
    "tiles_int",
    "tiles_float",
    "actor_slot",
    "actor_cell",
    "actor_role",
    "actor_mask",
    "actor_inventory",
    "actor_inventory_rank",
    "actors_float",
    "player_features",
    "storage_counts",
    "storage_rank",
    "banks",
    "shop_type",
    "shop_slot",
    "shop_mask",
    "market_product",
    "market_float",
    "market_int",
    "global_features",
    "globals_int",
    "still_playing",
    "order_limits",
    "can_act",
    "rewards",
    "dones",
    "transition_banks_before",
    "transition_banks_after",
    "transition_econ_before",
    "transition_econ_after",
    "tokens",
    "lengths",
    "mask",
    "out",
];

fn extract_output<'py, T: Element, const FIELD: usize>(
    value: &Bound<'py, PyAny>,
) -> PyResult<PyReadwriteArrayDyn<'py, T>> {
    let array = value
        .cast::<PyArrayDyn<T>>()
        .map_err(|error| PyValueError::new_err(format!("{}: {error}", NAMES[FIELD])))?;
    array.try_readwrite().map_err(|error| {
        PyValueError::new_err(format!("{}: writable borrow failed: {error}", NAMES[FIELD]))
    })
}
fn extract_input<'py, T: Element, const FIELD: usize>(
    value: &Bound<'py, PyAny>,
) -> PyResult<PyReadonlyArrayDyn<'py, T>> {
    let array = value
        .cast::<PyArrayDyn<T>>()
        .map_err(|error| PyValueError::new_err(format!("{}: {error}", NAMES[FIELD])))?;
    array.try_readonly().map_err(|error| {
        PyValueError::new_err(format!("{}: readonly borrow failed: {error}", NAMES[FIELD]))
    })
}

#[derive(Clone, Copy)]
struct Region {
    name: &'static str,
    start: usize,
    end: usize,
}
fn region<T: Element>(
    array: &Bound<'_, PyArrayDyn<T>>,
    name: &'static str,
    shape: &[usize],
) -> PyResult<Region> {
    if array.shape() != shape {
        return Err(PyValueError::new_err(format!(
            "{name}: expected shape {shape:?}, received {:?}",
            array.shape()
        )));
    }
    if !array.is_c_contiguous() || !array.is_aligned() {
        return Err(PyValueError::new_err(format!(
            "{name}: expected aligned C-contiguous storage"
        )));
    }
    let bytes = shape
        .iter()
        .try_fold(std::mem::size_of::<T>(), |n, d| n.checked_mul(*d))
        .ok_or_else(|| PyValueError::new_err(format!("{name}: byte size overflow")))?;
    let start = array.data() as usize;
    let end = start
        .checked_add(bytes)
        .ok_or_else(|| PyValueError::new_err(format!("{name}: byte address overflow")))?;
    Ok(Region { name, start, end })
}
fn check_disjoint(regions: &[Region]) -> PyResult<()> {
    for (i, right) in regions.iter().enumerate() {
        for left in &regions[..i] {
            if left.start < right.end && right.start < left.end {
                return Err(PyValueError::new_err(format!(
                    "{}: byte region overlaps {}",
                    right.name, left.name
                )));
            }
        }
    }
    Ok(())
}
fn output_slice<'a, T: Element>(
    array: &'a mut PyReadwriteArrayDyn<'_, T>,
    name: &str,
) -> PyResult<&'a mut [T]> {
    array
        .as_slice_mut()
        .map_err(|error| PyValueError::new_err(format!("{name}: {error}")))
}

struct OutputArrays<'py> {
    tile_kind: PyReadwriteArrayDyn<'py, i64>,
    tile_crop: PyReadwriteArrayDyn<'py, i64>,
    tile_animal: PyReadwriteArrayDyn<'py, i64>,
    tile_cell: PyReadwriteArrayDyn<'py, i64>,
    tile_role: PyReadwriteArrayDyn<'py, i64>,
    tiles_int: PyReadwriteArrayDyn<'py, i64>,
    tiles_float: PyReadwriteArrayDyn<'py, f32>,
    actor_slot: PyReadwriteArrayDyn<'py, i64>,
    actor_cell: PyReadwriteArrayDyn<'py, i64>,
    actor_role: PyReadwriteArrayDyn<'py, i64>,
    actor_mask: PyReadwriteArrayDyn<'py, bool>,
    actor_inventory: PyReadwriteArrayDyn<'py, i64>,
    actor_inventory_rank: PyReadwriteArrayDyn<'py, i64>,
    actors_float: PyReadwriteArrayDyn<'py, f32>,
    player_features: PyReadwriteArrayDyn<'py, f32>,
    storage_counts: PyReadwriteArrayDyn<'py, i64>,
    storage_rank: PyReadwriteArrayDyn<'py, i64>,
    banks: PyReadwriteArrayDyn<'py, f64>,
    shop_type: PyReadwriteArrayDyn<'py, i64>,
    shop_slot: PyReadwriteArrayDyn<'py, i64>,
    shop_mask: PyReadwriteArrayDyn<'py, bool>,
    market_product: PyReadwriteArrayDyn<'py, i64>,
    market_float: PyReadwriteArrayDyn<'py, f32>,
    market_int: PyReadwriteArrayDyn<'py, i64>,
    global_features: PyReadwriteArrayDyn<'py, f32>,
    globals_int: PyReadwriteArrayDyn<'py, i64>,
    still_playing: PyReadwriteArrayDyn<'py, bool>,
    order_limits: PyReadwriteArrayDyn<'py, i64>,
    can_act: PyReadwriteArrayDyn<'py, bool>,
    rewards: PyReadwriteArrayDyn<'py, f32>,
    dones: PyReadwriteArrayDyn<'py, bool>,
    transition_banks_before: PyReadwriteArrayDyn<'py, f64>,
    transition_banks_after: PyReadwriteArrayDyn<'py, f64>,
    transition_econ_before: PyReadwriteArrayDyn<'py, i64>,
    transition_econ_after: PyReadwriteArrayDyn<'py, i64>,
}
impl OutputArrays<'_> {
    fn preflight(&self, n: usize, inputs: &[Region]) -> PyResult<()> {
        let mut regions = Vec::with_capacity(35 + inputs.len());
        regions.push(region(&self.tile_kind, "tile_kind", &[n, 2, 200])?);
        regions.push(region(&self.tile_crop, "tile_crop", &[n, 2, 200])?);
        regions.push(region(&self.tile_animal, "tile_animal", &[n, 2, 200])?);
        regions.push(region(&self.tile_cell, "tile_cell", &[n, 2, 200])?);
        regions.push(region(&self.tile_role, "tile_role", &[n, 2, 200])?);
        regions.push(region(&self.tiles_int, "tiles_int", &[n, 2, 200, 7])?);
        regions.push(region(&self.tiles_float, "tiles_float", &[n, 2, 200, 15])?);
        regions.push(region(&self.actor_slot, "actor_slot", &[n, 2, 482])?);
        regions.push(region(&self.actor_cell, "actor_cell", &[n, 2, 482])?);
        regions.push(region(&self.actor_role, "actor_role", &[n, 2, 482])?);
        regions.push(region(&self.actor_mask, "actor_mask", &[n, 2, 482])?);
        regions.push(region(
            &self.actor_inventory,
            "actor_inventory",
            &[n, 2, 241, 12],
        )?);
        regions.push(region(
            &self.actor_inventory_rank,
            "actor_inventory_rank",
            &[n, 2, 241, 12],
        )?);
        regions.push(region(
            &self.actors_float,
            "actors_float",
            &[n, 2, 482, 26],
        )?);
        regions.push(region(
            &self.player_features,
            "player_features",
            &[n, 2, 2, 44],
        )?);
        regions.push(region(&self.storage_counts, "storage_counts", &[n, 2, 17])?);
        regions.push(region(&self.storage_rank, "storage_rank", &[n, 2, 12])?);
        regions.push(region(&self.banks, "banks", &[n, 2, 2])?);
        regions.push(region(&self.shop_type, "shop_type", &[n, 2, 8])?);
        regions.push(region(&self.shop_slot, "shop_slot", &[n, 2, 8])?);
        regions.push(region(&self.shop_mask, "shop_mask", &[n, 2, 8])?);
        regions.push(region(&self.market_product, "market_product", &[n, 2, 9])?);
        regions.push(region(&self.market_float, "market_float", &[n, 2, 9, 2])?);
        regions.push(region(&self.market_int, "market_int", &[n, 2, 9, 2])?);
        regions.push(region(
            &self.global_features,
            "global_features",
            &[n, 2, 15],
        )?);
        regions.push(region(&self.globals_int, "globals_int", &[n, 2, 16])?);
        regions.push(region(&self.still_playing, "still_playing", &[n, 2])?);
        regions.push(region(&self.order_limits, "order_limits", &[n, 2])?);
        regions.push(region(&self.can_act, "can_act", &[n, 2, 252])?);
        regions.push(region(&self.rewards, "rewards", &[n, 2])?);
        regions.push(region(&self.dones, "dones", &[n, 2])?);
        regions.push(region(
            &self.transition_banks_before,
            "transition_banks_before",
            &[n, 2],
        )?);
        regions.push(region(
            &self.transition_banks_after,
            "transition_banks_after",
            &[n, 2],
        )?);
        regions.push(region(
            &self.transition_econ_before,
            "transition_econ_before",
            &[n, 2, 32],
        )?);
        regions.push(region(
            &self.transition_econ_after,
            "transition_econ_after",
            &[n, 2, 32],
        )?);
        regions.extend_from_slice(inputs);
        // Complete address preflight precedes the first Rust mutable slice.
        check_disjoint(&regions)
    }
    fn slices(
        &mut self,
        n: usize,
    ) -> PyResult<(ValidatedObsBuffersMut<'_>, TransitionBuffersMut<'_>)> {
        let obs = ObsBuffersMut {
            tile_kind: output_slice(&mut self.tile_kind, "tile_kind")?,
            tile_crop: output_slice(&mut self.tile_crop, "tile_crop")?,
            tile_animal: output_slice(&mut self.tile_animal, "tile_animal")?,
            tile_cell: output_slice(&mut self.tile_cell, "tile_cell")?,
            tile_role: output_slice(&mut self.tile_role, "tile_role")?,
            tiles_int: output_slice(&mut self.tiles_int, "tiles_int")?,
            tiles_float: output_slice(&mut self.tiles_float, "tiles_float")?,
            actor_slot: output_slice(&mut self.actor_slot, "actor_slot")?,
            actor_cell: output_slice(&mut self.actor_cell, "actor_cell")?,
            actor_role: output_slice(&mut self.actor_role, "actor_role")?,
            actor_mask: output_slice(&mut self.actor_mask, "actor_mask")?,
            actor_inventory: output_slice(&mut self.actor_inventory, "actor_inventory")?,
            actor_inventory_rank: output_slice(
                &mut self.actor_inventory_rank,
                "actor_inventory_rank",
            )?,
            actors_float: output_slice(&mut self.actors_float, "actors_float")?,
            player_features: output_slice(&mut self.player_features, "player_features")?,
            storage_counts: output_slice(&mut self.storage_counts, "storage_counts")?,
            storage_rank: output_slice(&mut self.storage_rank, "storage_rank")?,
            banks: output_slice(&mut self.banks, "banks")?,
            shop_type: output_slice(&mut self.shop_type, "shop_type")?,
            shop_slot: output_slice(&mut self.shop_slot, "shop_slot")?,
            shop_mask: output_slice(&mut self.shop_mask, "shop_mask")?,
            market_product: output_slice(&mut self.market_product, "market_product")?,
            market_float: output_slice(&mut self.market_float, "market_float")?,
            market_int: output_slice(&mut self.market_int, "market_int")?,
            global_features: output_slice(&mut self.global_features, "global_features")?,
            globals_int: output_slice(&mut self.globals_int, "globals_int")?,
            still_playing: output_slice(&mut self.still_playing, "still_playing")?,
            order_limits: output_slice(&mut self.order_limits, "order_limits")?,
            can_act: output_slice(&mut self.can_act, "can_act")?,
        }
        .validate(n)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let transition = TransitionBuffersMut {
            rewards: output_slice(&mut self.rewards, "rewards")?,
            dones: output_slice(&mut self.dones, "dones")?,
            transition_banks_before: output_slice(
                &mut self.transition_banks_before,
                "transition_banks_before",
            )?,
            transition_banks_after: output_slice(
                &mut self.transition_banks_after,
                "transition_banks_after",
            )?,
            transition_econ_before: output_slice(
                &mut self.transition_econ_before,
                "transition_econ_before",
            )?,
            transition_econ_after: output_slice(
                &mut self.transition_econ_after,
                "transition_econ_after",
            )?,
        };
        Ok((obs, transition))
    }
}

fn integer_seed(value: &Bound<'_, PyAny>, name: &str) -> PyResult<i64> {
    if value.is_instance_of::<PyBool>() || !value.is_instance_of::<PyInt>() {
        return Err(PyValueError::new_err(format!(
            "{name}: expected Python integer, not bool, float or string"
        )));
    }
    value
        .extract::<i64>()
        .map_err(|error| PyOverflowError::new_err(format!("{name}: {error}")))
}
fn reward_config(value: &Bound<'_, PyDict>) -> PyResult<RewardConfig> {
    const KEYS: [&str; 7] = [
        "reward_mode",
        "econ_shaping",
        "econ_starvation_weight",
        "econ_drought_weight",
        "econ_cap",
        "econ_ineffective_weight",
        "econ_ineffective_cap",
    ];
    if !value.is_exact_instance_of::<PyDict>() || value.len() != KEYS.len() {
        return Err(PyValueError::new_err(
            "reward_config: expected plain dict with exactly the seven required keys",
        ));
    }
    for key in KEYS {
        if !value.contains(key)? {
            return Err(PyValueError::new_err(format!(
                "reward_config: missing required key {key}"
            )));
        }
    }
    let mode = value
        .get_item("reward_mode")?
        .ok_or_else(|| PyValueError::new_err("reward_mode missing"))?;
    if mode
        .extract::<&str>()
        .map_err(|error| PyValueError::new_err(format!("reward_mode: {error}")))?
        != "win_loss"
    {
        return Err(PyValueError::new_err(
            "reward_mode: only win_loss is supported",
        ));
    }
    let coefficient = |name: &str| -> PyResult<f64> {
        let item = value
            .get_item(name)?
            .ok_or_else(|| PyValueError::new_err(format!("{name}: missing")))?;
        if item.is_instance_of::<PyBool>() {
            return Err(PyValueError::new_err(format!(
                "{name}: bool is not a coefficient"
            )));
        }
        item.extract::<f64>()
            .map_err(|error| PyValueError::new_err(format!("{name}: {error}")))
    };
    let config = RewardConfig {
        reward_mode: RewardMode::WinLoss,
        econ_shaping: coefficient("econ_shaping")?,
        econ_starvation_weight: coefficient("econ_starvation_weight")?,
        econ_drought_weight: coefficient("econ_drought_weight")?,
        econ_cap: coefficient("econ_cap")?,
        econ_ineffective_weight: coefficient("econ_ineffective_weight")?,
        econ_ineffective_cap: coefficient("econ_ineffective_cap")?,
    };
    config.validate().map_err(PyValueError::new_err)?;
    Ok(config)
}

// Only Send work/results cross the GIL boundary; Python guards stay in callers.
fn native_work<T: Send, F: FnOnce() -> T + Send>(py: Python<'_>, work: F) -> T {
    py.detach(|| {
        #[cfg(test)]
        detached_test::await_python_progress();
        work()
    })
}

#[pyclass(name = "KaggricultureEnv", module = "owl.rs")]
pub struct PyKaggricultureEnv {
    native: NativeEnv,
}

#[allow(clippy::too_many_arguments)]
#[pymethods]
impl PyKaggricultureEnv {
    #[new]
    #[pyo3(signature = (n_envs, seed, seed_stride, config, reward_config, native_threads, *, hire_limit))]
    fn new(
        py: Python<'_>,
        n_envs: usize,
        seed: &Bound<'_, PyAny>,
        seed_stride: &Bound<'_, PyAny>,
        config: &str,
        reward_config: &Bound<'_, PyDict>,
        native_threads: usize,
        hire_limit: i64,
    ) -> PyResult<Self> {
        let seed = integer_seed(seed, "seed")?;
        let stride = integer_seed(seed_stride, "seed_stride")?;
        let config: Config = serde_json::from_str(config)
            .map_err(|error| PyValueError::new_err(format!("config: {error}")))?;
        let reward = self::reward_config(reward_config)?;
        py.detach(move || {
            NativeEnv::new(
                n_envs,
                seed,
                stride,
                config,
                reward,
                native_threads,
                hire_limit,
            )
        })
        .map(|native| Self { native })
        .map_err(python_error)
    }
    #[pyo3(signature = (*, tile_kind, tile_crop, tile_animal, tile_cell, tile_role, tiles_int, tiles_float, actor_slot, actor_cell, actor_role, actor_mask, actor_inventory, actor_inventory_rank, actors_float, player_features, storage_counts, storage_rank, banks, shop_type, shop_slot, shop_mask, market_product, market_float, market_int, global_features, globals_int, still_playing, order_limits, can_act, rewards, dones, transition_banks_before, transition_banks_after, transition_econ_before, transition_econ_after))]
    fn observe(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = extract_output::<i64, 0>)] tile_kind: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<i64, 1>)] tile_crop: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<i64, 2>)] tile_animal: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<i64, 3>)] tile_cell: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<i64, 4>)] tile_role: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<i64, 5>)] tiles_int: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<f32, 6>)] tiles_float: PyReadwriteArrayDyn<'_, f32>,
        #[pyo3(from_py_with = extract_output::<i64, 7>)] actor_slot: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<i64, 8>)] actor_cell: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<i64, 9>)] actor_role: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<bool, 10>)] actor_mask: PyReadwriteArrayDyn<
            '_,
            bool,
        >,
        #[pyo3(from_py_with = extract_output::<i64, 11>)] actor_inventory: PyReadwriteArrayDyn<
            '_,
            i64,
        >,
        #[pyo3(from_py_with = extract_output::<i64, 12>)] actor_inventory_rank: PyReadwriteArrayDyn<
            '_,
            i64,
        >,
        #[pyo3(from_py_with = extract_output::<f32, 13>)] actors_float: PyReadwriteArrayDyn<
            '_,
            f32,
        >,
        #[pyo3(from_py_with = extract_output::<f32, 14>)] player_features: PyReadwriteArrayDyn<
            '_,
            f32,
        >,
        #[pyo3(from_py_with = extract_output::<i64, 15>)] storage_counts: PyReadwriteArrayDyn<
            '_,
            i64,
        >,
        #[pyo3(from_py_with = extract_output::<i64, 16>)] storage_rank: PyReadwriteArrayDyn<
            '_,
            i64,
        >,
        #[pyo3(from_py_with = extract_output::<f64, 17>)] banks: PyReadwriteArrayDyn<'_, f64>,
        #[pyo3(from_py_with = extract_output::<i64, 18>)] shop_type: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<i64, 19>)] shop_slot: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<bool, 20>)] shop_mask: PyReadwriteArrayDyn<'_, bool>,
        #[pyo3(from_py_with = extract_output::<i64, 21>)] market_product: PyReadwriteArrayDyn<
            '_,
            i64,
        >,
        #[pyo3(from_py_with = extract_output::<f32, 22>)] market_float: PyReadwriteArrayDyn<
            '_,
            f32,
        >,
        #[pyo3(from_py_with = extract_output::<i64, 23>)] market_int: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<f32, 24>)] global_features: PyReadwriteArrayDyn<
            '_,
            f32,
        >,
        #[pyo3(from_py_with = extract_output::<i64, 25>)] globals_int: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<bool, 26>)] still_playing: PyReadwriteArrayDyn<
            '_,
            bool,
        >,
        #[pyo3(from_py_with = extract_output::<i64, 27>)] order_limits: PyReadwriteArrayDyn<
            '_,
            i64,
        >,
        #[pyo3(from_py_with = extract_output::<bool, 28>)] can_act: PyReadwriteArrayDyn<'_, bool>,
        #[pyo3(from_py_with = extract_output::<f32, 29>)] rewards: PyReadwriteArrayDyn<'_, f32>,
        #[pyo3(from_py_with = extract_output::<bool, 30>)] dones: PyReadwriteArrayDyn<'_, bool>,
        #[pyo3(from_py_with = extract_output::<f64, 31>)]
        transition_banks_before: PyReadwriteArrayDyn<'_, f64>,
        #[pyo3(from_py_with = extract_output::<f64, 32>)]
        transition_banks_after: PyReadwriteArrayDyn<'_, f64>,
        #[pyo3(from_py_with = extract_output::<i64, 33>)]
        transition_econ_before: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<i64, 34>)]
        transition_econ_after: PyReadwriteArrayDyn<'_, i64>,
    ) -> PyResult<()> {
        let n = self.native.n_envs();
        let mut arrays = OutputArrays {
            tile_kind,
            tile_crop,
            tile_animal,
            tile_cell,
            tile_role,
            tiles_int,
            tiles_float,
            actor_slot,
            actor_cell,
            actor_role,
            actor_mask,
            actor_inventory,
            actor_inventory_rank,
            actors_float,
            player_features,
            storage_counts,
            storage_rank,
            banks,
            shop_type,
            shop_slot,
            shop_mask,
            market_product,
            market_float,
            market_int,
            global_features,
            globals_int,
            still_playing,
            order_limits,
            can_act,
            rewards,
            dones,
            transition_banks_before,
            transition_banks_after,
            transition_econ_before,
            transition_econ_after,
        };
        let inputs = [];
        arrays.preflight(n, &inputs)?;
        let (mut out, transitions) = arrays.slices(n)?;
        let pending = native_work(py, || self.native.prepare_observe()).map_err(python_error)?;
        py.detach(|| self.native.publish_observe(pending, &mut out, transitions))
            .map_err(python_error)?;
        // Typed NumPy guards remain alive across both detached phases.
        Ok(())
    }
    #[pyo3(signature = (*, tile_kind, tile_crop, tile_animal, tile_cell, tile_role, tiles_int, tiles_float, actor_slot, actor_cell, actor_role, actor_mask, actor_inventory, actor_inventory_rank, actors_float, player_features, storage_counts, storage_rank, banks, shop_type, shop_slot, shop_mask, market_product, market_float, market_int, global_features, globals_int, still_playing, order_limits, can_act, rewards, dones, transition_banks_before, transition_banks_after, transition_econ_before, transition_econ_after))]
    fn reset(
        &mut self,
        py: Python<'_>,
        #[pyo3(from_py_with = extract_output::<i64, 0>)] tile_kind: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<i64, 1>)] tile_crop: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<i64, 2>)] tile_animal: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<i64, 3>)] tile_cell: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<i64, 4>)] tile_role: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<i64, 5>)] tiles_int: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<f32, 6>)] tiles_float: PyReadwriteArrayDyn<'_, f32>,
        #[pyo3(from_py_with = extract_output::<i64, 7>)] actor_slot: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<i64, 8>)] actor_cell: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<i64, 9>)] actor_role: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<bool, 10>)] actor_mask: PyReadwriteArrayDyn<
            '_,
            bool,
        >,
        #[pyo3(from_py_with = extract_output::<i64, 11>)] actor_inventory: PyReadwriteArrayDyn<
            '_,
            i64,
        >,
        #[pyo3(from_py_with = extract_output::<i64, 12>)] actor_inventory_rank: PyReadwriteArrayDyn<
            '_,
            i64,
        >,
        #[pyo3(from_py_with = extract_output::<f32, 13>)] actors_float: PyReadwriteArrayDyn<
            '_,
            f32,
        >,
        #[pyo3(from_py_with = extract_output::<f32, 14>)] player_features: PyReadwriteArrayDyn<
            '_,
            f32,
        >,
        #[pyo3(from_py_with = extract_output::<i64, 15>)] storage_counts: PyReadwriteArrayDyn<
            '_,
            i64,
        >,
        #[pyo3(from_py_with = extract_output::<i64, 16>)] storage_rank: PyReadwriteArrayDyn<
            '_,
            i64,
        >,
        #[pyo3(from_py_with = extract_output::<f64, 17>)] banks: PyReadwriteArrayDyn<'_, f64>,
        #[pyo3(from_py_with = extract_output::<i64, 18>)] shop_type: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<i64, 19>)] shop_slot: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<bool, 20>)] shop_mask: PyReadwriteArrayDyn<'_, bool>,
        #[pyo3(from_py_with = extract_output::<i64, 21>)] market_product: PyReadwriteArrayDyn<
            '_,
            i64,
        >,
        #[pyo3(from_py_with = extract_output::<f32, 22>)] market_float: PyReadwriteArrayDyn<
            '_,
            f32,
        >,
        #[pyo3(from_py_with = extract_output::<i64, 23>)] market_int: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<f32, 24>)] global_features: PyReadwriteArrayDyn<
            '_,
            f32,
        >,
        #[pyo3(from_py_with = extract_output::<i64, 25>)] globals_int: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<bool, 26>)] still_playing: PyReadwriteArrayDyn<
            '_,
            bool,
        >,
        #[pyo3(from_py_with = extract_output::<i64, 27>)] order_limits: PyReadwriteArrayDyn<
            '_,
            i64,
        >,
        #[pyo3(from_py_with = extract_output::<bool, 28>)] can_act: PyReadwriteArrayDyn<'_, bool>,
        #[pyo3(from_py_with = extract_output::<f32, 29>)] rewards: PyReadwriteArrayDyn<'_, f32>,
        #[pyo3(from_py_with = extract_output::<bool, 30>)] dones: PyReadwriteArrayDyn<'_, bool>,
        #[pyo3(from_py_with = extract_output::<f64, 31>)]
        transition_banks_before: PyReadwriteArrayDyn<'_, f64>,
        #[pyo3(from_py_with = extract_output::<f64, 32>)]
        transition_banks_after: PyReadwriteArrayDyn<'_, f64>,
        #[pyo3(from_py_with = extract_output::<i64, 33>)]
        transition_econ_before: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<i64, 34>)]
        transition_econ_after: PyReadwriteArrayDyn<'_, i64>,
    ) -> PyResult<()> {
        let n = self.native.n_envs();
        let mut arrays = OutputArrays {
            tile_kind,
            tile_crop,
            tile_animal,
            tile_cell,
            tile_role,
            tiles_int,
            tiles_float,
            actor_slot,
            actor_cell,
            actor_role,
            actor_mask,
            actor_inventory,
            actor_inventory_rank,
            actors_float,
            player_features,
            storage_counts,
            storage_rank,
            banks,
            shop_type,
            shop_slot,
            shop_mask,
            market_product,
            market_float,
            market_int,
            global_features,
            globals_int,
            still_playing,
            order_limits,
            can_act,
            rewards,
            dones,
            transition_banks_before,
            transition_banks_after,
            transition_econ_before,
            transition_econ_after,
        };
        let inputs = [];
        arrays.preflight(n, &inputs)?;
        let (mut out, transitions) = arrays.slices(n)?;
        let mask = vec![true; n];
        let pending =
            native_work(py, || self.native.prepare_reset(&mask, false)).map_err(python_error)?;
        py.detach(|| self.native.commit(pending, &mut out, transitions))
            .map_err(python_error)?;
        // Typed NumPy guards remain alive across both detached phases.
        Ok(())
    }
    #[pyo3(signature = (tokens, lengths, *, tile_kind, tile_crop, tile_animal, tile_cell, tile_role, tiles_int, tiles_float, actor_slot, actor_cell, actor_role, actor_mask, actor_inventory, actor_inventory_rank, actors_float, player_features, storage_counts, storage_rank, banks, shop_type, shop_slot, shop_mask, market_product, market_float, market_int, global_features, globals_int, still_playing, order_limits, can_act, rewards, dones, transition_banks_before, transition_banks_after, transition_econ_before, transition_econ_after))]
    fn step(
        &mut self,
        py: Python<'_>,
        #[pyo3(from_py_with = extract_input::<i64, 35>)] tokens: PyReadonlyArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_input::<i64, 36>)] lengths: PyReadonlyArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<i64, 0>)] tile_kind: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<i64, 1>)] tile_crop: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<i64, 2>)] tile_animal: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<i64, 3>)] tile_cell: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<i64, 4>)] tile_role: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<i64, 5>)] tiles_int: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<f32, 6>)] tiles_float: PyReadwriteArrayDyn<'_, f32>,
        #[pyo3(from_py_with = extract_output::<i64, 7>)] actor_slot: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<i64, 8>)] actor_cell: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<i64, 9>)] actor_role: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<bool, 10>)] actor_mask: PyReadwriteArrayDyn<
            '_,
            bool,
        >,
        #[pyo3(from_py_with = extract_output::<i64, 11>)] actor_inventory: PyReadwriteArrayDyn<
            '_,
            i64,
        >,
        #[pyo3(from_py_with = extract_output::<i64, 12>)] actor_inventory_rank: PyReadwriteArrayDyn<
            '_,
            i64,
        >,
        #[pyo3(from_py_with = extract_output::<f32, 13>)] actors_float: PyReadwriteArrayDyn<
            '_,
            f32,
        >,
        #[pyo3(from_py_with = extract_output::<f32, 14>)] player_features: PyReadwriteArrayDyn<
            '_,
            f32,
        >,
        #[pyo3(from_py_with = extract_output::<i64, 15>)] storage_counts: PyReadwriteArrayDyn<
            '_,
            i64,
        >,
        #[pyo3(from_py_with = extract_output::<i64, 16>)] storage_rank: PyReadwriteArrayDyn<
            '_,
            i64,
        >,
        #[pyo3(from_py_with = extract_output::<f64, 17>)] banks: PyReadwriteArrayDyn<'_, f64>,
        #[pyo3(from_py_with = extract_output::<i64, 18>)] shop_type: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<i64, 19>)] shop_slot: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<bool, 20>)] shop_mask: PyReadwriteArrayDyn<'_, bool>,
        #[pyo3(from_py_with = extract_output::<i64, 21>)] market_product: PyReadwriteArrayDyn<
            '_,
            i64,
        >,
        #[pyo3(from_py_with = extract_output::<f32, 22>)] market_float: PyReadwriteArrayDyn<
            '_,
            f32,
        >,
        #[pyo3(from_py_with = extract_output::<i64, 23>)] market_int: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<f32, 24>)] global_features: PyReadwriteArrayDyn<
            '_,
            f32,
        >,
        #[pyo3(from_py_with = extract_output::<i64, 25>)] globals_int: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<bool, 26>)] still_playing: PyReadwriteArrayDyn<
            '_,
            bool,
        >,
        #[pyo3(from_py_with = extract_output::<i64, 27>)] order_limits: PyReadwriteArrayDyn<
            '_,
            i64,
        >,
        #[pyo3(from_py_with = extract_output::<bool, 28>)] can_act: PyReadwriteArrayDyn<'_, bool>,
        #[pyo3(from_py_with = extract_output::<f32, 29>)] rewards: PyReadwriteArrayDyn<'_, f32>,
        #[pyo3(from_py_with = extract_output::<bool, 30>)] dones: PyReadwriteArrayDyn<'_, bool>,
        #[pyo3(from_py_with = extract_output::<f64, 31>)]
        transition_banks_before: PyReadwriteArrayDyn<'_, f64>,
        #[pyo3(from_py_with = extract_output::<f64, 32>)]
        transition_banks_after: PyReadwriteArrayDyn<'_, f64>,
        #[pyo3(from_py_with = extract_output::<i64, 33>)]
        transition_econ_before: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<i64, 34>)]
        transition_econ_after: PyReadwriteArrayDyn<'_, i64>,
    ) -> PyResult<Py<PyDict>> {
        let n = self.native.n_envs();
        let mut arrays = OutputArrays {
            tile_kind,
            tile_crop,
            tile_animal,
            tile_cell,
            tile_role,
            tiles_int,
            tiles_float,
            actor_slot,
            actor_cell,
            actor_role,
            actor_mask,
            actor_inventory,
            actor_inventory_rank,
            actors_float,
            player_features,
            storage_counts,
            storage_rank,
            banks,
            shop_type,
            shop_slot,
            shop_mask,
            market_product,
            market_float,
            market_int,
            global_features,
            globals_int,
            still_playing,
            order_limits,
            can_act,
            rewards,
            dones,
            transition_banks_before,
            transition_banks_after,
            transition_econ_before,
            transition_econ_after,
        };
        let inputs = [
            region(&tokens, "tokens", &[n, 2, 252, 12])?,
            region(&lengths, "lengths", &[n, 2])?,
        ];
        arrays.preflight(n, &inputs)?;
        let tokens_slice = tokens
            .as_slice()
            .map_err(|error| PyValueError::new_err(format!("tokens: {error}")))?;
        let lengths_slice = lengths
            .as_slice()
            .map_err(|error| PyValueError::new_err(format!("lengths: {error}")))?;
        let (mut out, transitions) = arrays.slices(n)?;
        let pending = native_work(py, || self.native.prepare_step(tokens_slice, lengths_slice))
            .map_err(python_error)?;
        // Python allocation/conversion completes before the native commit point.
        let result = PyDict::new(py);
        let metrics = pending.metrics();
        result.set_item("total_games_played", vec![1.0; metrics.len()])?;
        result.set_item(
            "terminal_bank_0",
            metrics.iter().map(|m| m.0).collect::<Vec<_>>(),
        )?;
        result.set_item(
            "terminal_bank_1",
            metrics.iter().map(|m| m.1).collect::<Vec<_>>(),
        )?;
        result.set_item(
            "terminal_margin_0",
            metrics.iter().map(|m| m.2).collect::<Vec<_>>(),
        )?;
        let result = result.unbind();
        py.detach(|| self.native.commit(pending, &mut out, transitions))
            .map_err(python_error)?;
        // Typed NumPy guards remain alive across both detached phases.
        Ok(result)
    }
    #[pyo3(signature = (mask, *, tile_kind, tile_crop, tile_animal, tile_cell, tile_role, tiles_int, tiles_float, actor_slot, actor_cell, actor_role, actor_mask, actor_inventory, actor_inventory_rank, actors_float, player_features, storage_counts, storage_rank, banks, shop_type, shop_slot, shop_mask, market_product, market_float, market_int, global_features, globals_int, still_playing, order_limits, can_act, rewards, dones, transition_banks_before, transition_banks_after, transition_econ_before, transition_econ_after))]
    fn truncate_envs(
        &mut self,
        py: Python<'_>,
        #[pyo3(from_py_with = extract_input::<bool, 37>)] mask: PyReadonlyArrayDyn<'_, bool>,
        #[pyo3(from_py_with = extract_output::<i64, 0>)] tile_kind: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<i64, 1>)] tile_crop: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<i64, 2>)] tile_animal: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<i64, 3>)] tile_cell: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<i64, 4>)] tile_role: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<i64, 5>)] tiles_int: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<f32, 6>)] tiles_float: PyReadwriteArrayDyn<'_, f32>,
        #[pyo3(from_py_with = extract_output::<i64, 7>)] actor_slot: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<i64, 8>)] actor_cell: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<i64, 9>)] actor_role: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<bool, 10>)] actor_mask: PyReadwriteArrayDyn<
            '_,
            bool,
        >,
        #[pyo3(from_py_with = extract_output::<i64, 11>)] actor_inventory: PyReadwriteArrayDyn<
            '_,
            i64,
        >,
        #[pyo3(from_py_with = extract_output::<i64, 12>)] actor_inventory_rank: PyReadwriteArrayDyn<
            '_,
            i64,
        >,
        #[pyo3(from_py_with = extract_output::<f32, 13>)] actors_float: PyReadwriteArrayDyn<
            '_,
            f32,
        >,
        #[pyo3(from_py_with = extract_output::<f32, 14>)] player_features: PyReadwriteArrayDyn<
            '_,
            f32,
        >,
        #[pyo3(from_py_with = extract_output::<i64, 15>)] storage_counts: PyReadwriteArrayDyn<
            '_,
            i64,
        >,
        #[pyo3(from_py_with = extract_output::<i64, 16>)] storage_rank: PyReadwriteArrayDyn<
            '_,
            i64,
        >,
        #[pyo3(from_py_with = extract_output::<f64, 17>)] banks: PyReadwriteArrayDyn<'_, f64>,
        #[pyo3(from_py_with = extract_output::<i64, 18>)] shop_type: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<i64, 19>)] shop_slot: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<bool, 20>)] shop_mask: PyReadwriteArrayDyn<'_, bool>,
        #[pyo3(from_py_with = extract_output::<i64, 21>)] market_product: PyReadwriteArrayDyn<
            '_,
            i64,
        >,
        #[pyo3(from_py_with = extract_output::<f32, 22>)] market_float: PyReadwriteArrayDyn<
            '_,
            f32,
        >,
        #[pyo3(from_py_with = extract_output::<i64, 23>)] market_int: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<f32, 24>)] global_features: PyReadwriteArrayDyn<
            '_,
            f32,
        >,
        #[pyo3(from_py_with = extract_output::<i64, 25>)] globals_int: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<bool, 26>)] still_playing: PyReadwriteArrayDyn<
            '_,
            bool,
        >,
        #[pyo3(from_py_with = extract_output::<i64, 27>)] order_limits: PyReadwriteArrayDyn<
            '_,
            i64,
        >,
        #[pyo3(from_py_with = extract_output::<bool, 28>)] can_act: PyReadwriteArrayDyn<'_, bool>,
        #[pyo3(from_py_with = extract_output::<f32, 29>)] rewards: PyReadwriteArrayDyn<'_, f32>,
        #[pyo3(from_py_with = extract_output::<bool, 30>)] dones: PyReadwriteArrayDyn<'_, bool>,
        #[pyo3(from_py_with = extract_output::<f64, 31>)]
        transition_banks_before: PyReadwriteArrayDyn<'_, f64>,
        #[pyo3(from_py_with = extract_output::<f64, 32>)]
        transition_banks_after: PyReadwriteArrayDyn<'_, f64>,
        #[pyo3(from_py_with = extract_output::<i64, 33>)]
        transition_econ_before: PyReadwriteArrayDyn<'_, i64>,
        #[pyo3(from_py_with = extract_output::<i64, 34>)]
        transition_econ_after: PyReadwriteArrayDyn<'_, i64>,
    ) -> PyResult<()> {
        let n = self.native.n_envs();
        let mut arrays = OutputArrays {
            tile_kind,
            tile_crop,
            tile_animal,
            tile_cell,
            tile_role,
            tiles_int,
            tiles_float,
            actor_slot,
            actor_cell,
            actor_role,
            actor_mask,
            actor_inventory,
            actor_inventory_rank,
            actors_float,
            player_features,
            storage_counts,
            storage_rank,
            banks,
            shop_type,
            shop_slot,
            shop_mask,
            market_product,
            market_float,
            market_int,
            global_features,
            globals_int,
            still_playing,
            order_limits,
            can_act,
            rewards,
            dones,
            transition_banks_before,
            transition_banks_after,
            transition_econ_before,
            transition_econ_after,
        };
        let inputs = [region(&mask, "mask", &[n])?];
        arrays.preflight(n, &inputs)?;
        let mask_slice = mask
            .as_slice()
            .map_err(|error| PyValueError::new_err(format!("mask: {error}")))?;
        let (mut out, transitions) = arrays.slices(n)?;
        let pending = native_work(py, || self.native.prepare_reset(mask_slice, true))
            .map_err(python_error)?;
        py.detach(|| self.native.commit(pending, &mut out, transitions))
            .map_err(python_error)?;
        // Typed NumPy guards remain alive across both detached phases.
        Ok(())
    }
    fn terminal_metrics(&self, py: Python<'_>, env_index: usize) -> PyResult<Option<Py<PyDict>>> {
        let Some(record) = self
            .native
            .terminal_metrics(env_index)
            .map_err(python_error)?
        else {
            return Ok(None);
        };
        let result = PyDict::new(py);
        result.set_item("bank_0", record.banks[0])?;
        result.set_item("bank_1", record.banks[1])?;
        result.set_item("margin_0", record.margin)?;
        result.set_item("episode_steps", record.episode_steps)?;
        result.set_item("winner", record.winner)?;
        result.set_item("econ_0", record.econ[0].to_vec().into_pyarray(py))?;
        result.set_item("econ_1", record.econ[1].to_vec().into_pyarray(py))?;
        Ok(Some(result.unbind()))
    }
    fn state_snapshot(&self, env_index: usize) -> PyResult<String> {
        self.native.state_snapshot(env_index).map_err(python_error)
    }
    fn seed_state(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let (next, seeds) = self.native.seed_state();
        let current = PyTuple::new(py, seeds)?;
        Ok(PyTuple::new(py, [next.into_pyobject(py)?.into_any(), current.into_any()])?.unbind())
    }
}

#[cfg(test)]
mod detached_test {
    use super::*;
    use std::cell::RefCell;
    use std::sync::{Arc, Condvar, Mutex};
    use std::time::Duration;

    #[derive(Default)]
    struct Latch {
        entered: bool,
        progressed: bool,
    }
    type SharedLatch = Arc<(Mutex<Latch>, Condvar)>;
    thread_local! {
        static LATCH: RefCell<Option<SharedLatch>> = const { RefCell::new(None) };
    }
    pub(super) fn await_python_progress() {
        LATCH.with(|slot| {
            let Some(latch) = slot.borrow().clone() else {
                return;
            };
            let (lock, ready) = &*latch;
            let mut state = lock.lock().unwrap();
            state.entered = true;
            ready.notify_all();
            let (state, timeout) = ready
                .wait_timeout_while(state, Duration::from_secs(2), |state| !state.progressed)
                .unwrap();
            let progressed = state.progressed;
            drop(state);
            assert!(
                !timeout.timed_out() && progressed,
                "Python thread could not progress while real observe work held the test latch"
            );
        });
    }

    #[test]
    fn native_observe_releases_gil_with_controlled_latch() {
        Python::initialize();
        let latch = Arc::new((Mutex::new(Latch::default()), Condvar::new()));
        let other = Arc::clone(&latch);
        let peer = std::thread::spawn(move || {
            let (lock, ready) = &*other;
            let state = lock.lock().unwrap();
            let (state, _) = ready
                .wait_timeout_while(state, Duration::from_secs(5), |state| !state.entered)
                .unwrap();
            if !state.entered {
                return;
            }
            drop(state);
            Python::attach(|py| {
                // Executing actual Python establishes interpreter ownership.
                assert_eq!(
                    py.eval(pyo3::ffi::c_str!("20 + 22"), None, None)
                        .unwrap()
                        .extract::<i64>()
                        .unwrap(),
                    42
                );
                let mut state = lock.lock().unwrap();
                state.progressed = true;
                ready.notify_all();
            });
        });
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Python::attach(|py| {
                // Embedded cargo tests do not activate the project venv by themselves.
                let site_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".venv/lib");
                if site_root.exists() {
                    for entry in std::fs::read_dir(site_root).unwrap() {
                        let site = entry.unwrap().path().join("site-packages");
                        if site.is_dir() {
                            py.import("sys")
                                .unwrap()
                                .getattr("path")
                                .unwrap()
                                .call_method1("insert", (0, site.to_str().unwrap()))
                                .unwrap();
                        }
                    }
                }
                let module = PyModule::new(py, "native_latch_test").unwrap();
                module.add_class::<PyKaggricultureEnv>().unwrap();
                let np = py.import("numpy").unwrap();
                let kwargs = PyDict::new(py);
                for field in super::super::buffers::FIELD_SHAPES {
                    let mut shape = vec![1usize, 2];
                    shape.extend(field.row_shape);
                    let dtype = match field.name {
                        "actor_mask" | "shop_mask" | "still_playing" | "action_mask.can_act" => {
                            "bool"
                        },
                        "tiles_float" | "actors_float" | "player_features" | "market_float"
                        | "global_features" => "float32",
                        "banks" => "float64",
                        _ => "int64",
                    };
                    let name = if field.name == "action_mask.can_act" {
                        "can_act"
                    } else {
                        field.name
                    };
                    kwargs
                        .set_item(
                            name,
                            np.call_method1("zeros", (PyTuple::new(py, shape).unwrap(), dtype))
                                .unwrap(),
                        )
                        .unwrap();
                }
                for (name, dtype, econ) in [
                    ("rewards", "float32", false),
                    ("dones", "bool", false),
                    ("transition_banks_before", "float64", false),
                    ("transition_banks_after", "float64", false),
                    ("transition_econ_before", "int64", true),
                    ("transition_econ_after", "int64", true),
                ] {
                    let shape = if econ {
                        vec![1usize, 2, 32]
                    } else {
                        vec![1usize, 2]
                    };
                    kwargs
                        .set_item(
                            name,
                            np.call_method1("zeros", (PyTuple::new(py, shape).unwrap(), dtype))
                                .unwrap(),
                        )
                        .unwrap();
                }
                let reward = PyDict::new(py);
                reward.set_item("reward_mode", "win_loss").unwrap();
                for (key, value) in [
                    ("econ_shaping", 0.2),
                    ("econ_starvation_weight", 4.),
                    ("econ_drought_weight", 1.),
                    ("econ_cap", 0.25),
                    ("econ_ineffective_weight", 0.),
                    ("econ_ineffective_cap", 0.1),
                ] {
                    reward.set_item(key, value).unwrap();
                }
                let options = PyDict::new(py);
                options.set_item("hire_limit", 241).unwrap();
                let env = module
                    .getattr("KaggricultureEnv")
                    .unwrap()
                    .call((1, 11, 2, "{}", reward, 1), Some(&options))
                    .unwrap();
                LATCH.with(|slot| *slot.borrow_mut() = Some(Arc::clone(&latch)));
                let observed = env.call_method("observe", (), Some(&kwargs));
                LATCH.with(|slot| *slot.borrow_mut() = None);
                observed.unwrap();
            })
        }));
        peer.join().unwrap();
        result.unwrap();
    }
}

/// Cold codec boundary: the root grammar stages all 3024 cells before writing.
#[pyfunction]
pub(super) fn kaggriculture_encode(
    action_json: &str,
    actors: i64,
    order_limit: i64,
    hire_limit: i64,
    #[pyo3(from_py_with = extract_output::<i64, 38>)] mut out: PyReadwriteArrayDyn<'_, i64>,
) -> PyResult<i64> {
    region(&out, "out", &[grammar::MAX_FRAMES, grammar::SLOTS])?;
    let plan = grammar::plan(actors, order_limit, hire_limit).map_err(PyValueError::new_err)?;
    let action = serde_json::from_str(action_json)
        .map_err(|error| PyValueError::new_err(format!("action_json: {error}")))?;
    grammar::encode(&plan, &action, output_slice(&mut out, "out")?).map_err(PyValueError::new_err)
}

#[pyfunction]
pub(super) fn kaggriculture_decode(
    #[pyo3(from_py_with = extract_input::<i64, 35>)] tokens: PyReadonlyArrayDyn<'_, i64>,
    length: i64,
    actors: i64,
    order_limit: i64,
    hire_limit: i64,
) -> PyResult<String> {
    region(&tokens, "tokens", &[grammar::MAX_FRAMES, grammar::SLOTS])?;
    let plan = grammar::plan(actors, order_limit, hire_limit).map_err(PyValueError::new_err)?;
    let tokens = tokens
        .as_slice()
        .map_err(|error| PyValueError::new_err(format!("tokens: {error}")))?;
    let action = grammar::decode(&plan, tokens, length).map_err(PyValueError::new_err)?;
    serde_json::to_string(&action)
        .map_err(|error| PyValueError::new_err(format!("canonical action JSON: {error}")))
}

fn bool_table<'py>(
    py: Python<'py>,
    shape: &[usize],
    values: impl IntoIterator<Item = bool>,
) -> PyResult<Bound<'py, PyArrayDyn<bool>>> {
    let array = numpy::ndarray::ArrayD::from_shape_vec(
        numpy::ndarray::IxDyn(shape),
        values.into_iter().collect(),
    )
    .map_err(|error| PyRuntimeError::new_err(format!("native grammar table: {error}")))?;
    Ok(array.into_pyarray(py))
}

#[pyfunction]
pub(super) fn kaggriculture_grammar_tables(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let tables = grammar::grammar_tables().map_err(PyRuntimeError::new_err)?;
    let result = PyDict::new(py);
    result.set_item("unit_kind", bool_table(py, &[20], tables.unit_kind)?)?;
    result.set_item(
        "unit_item",
        bool_table(py, &[20, 16], tables.unit_item.into_iter().flatten())?,
    )?;
    result.set_item(
        "unit_quantity_present",
        bool_table(
            py,
            &[20, 2],
            tables.unit_quantity_present.into_iter().flatten(),
        )?,
    )?;
    result.set_item(
        "unit_quantity_high",
        bool_table(
            py,
            &[2, 32],
            tables.unit_quantity_high.into_iter().flatten(),
        )?,
    )?;
    result.set_item(
        "unit_quantity_low",
        bool_table(
            py,
            &[2, 2, 32],
            tables.unit_quantity_low.into_iter().flatten().flatten(),
        )?,
    )?;
    result.set_item("market_kind", bool_table(py, &[8], tables.market_kind)?)?;
    result.set_item(
        "market_item",
        bool_table(py, &[8, 16], tables.market_item.into_iter().flatten())?,
    )?;
    result.set_item(
        "market_quantity",
        bool_table(py, &[8, 32], tables.market_quantity.into_iter().flatten())?,
    )?;
    Ok(result)
}

#[pyfunction]
pub(super) fn kaggriculture_grammar_constants(py: Python<'_>) -> PyResult<Bound<'_, PyTuple>> {
    let names = PyTuple::new(py, grammar::SLOT_NAMES)?;
    let widths = PyTuple::new(py, grammar::SLOT_WIDTHS)?;
    PyTuple::new(
        py,
        [
            grammar::GRAMMAR_TABLES_VERSION
                .into_pyobject(py)?
                .into_any(),
            names.into_any(),
            widths.into_any(),
        ],
    )
}
