//! Kaggriculture's typed observation boundary in the existing Rust extension.
//!
//! Named fields are admitted and written into caller-owned NumPy buffers.
//! Real Python-schema and full-corpus qualification have separate dependencies.
mod buffers;
mod config;
pub mod grammar;
mod observe;
pub mod replay_export;

#[cfg(test)]
mod replay_export_tests;

pub use buffers::{ObsBuffersMut, ObsEnvMut, ObsRowMut, ObsStaging, ValidatedObsBuffersMut};
pub use config::ObservationConfig;
pub use observe::{
    check_row, encode_env, write_env, write_seat, ObservationGame, ObserveError, ObserveErrorKind,
    PreparedObservation, Seat,
};

#[cfg(test)]
mod tests;

#[cfg(test)]
mod grammar_kernel_tests;

#[cfg(test)]
mod oracle_corpus;

use kaggriculture_engine::TraceHeader;
use numpy::{Element, PyArrayDyn, PyArrayMethods, PyReadwriteArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::Value;

use buffers::{checked_lengths, FIELD_SHAPES};

fn python_error(error: ObserveError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn extract_headers<'a>(value: &'a Bound<'_, PyAny>) -> PyResult<&'a str> {
    value
        .extract()
        .map_err(|error| PyValueError::new_err(format!("headers: {error}")))
}

// numpy's default typed extraction calls readwrite(), which panics on failed
// borrowing. Preserve the typed signature but use the fallible API and map all
// supplied-data errors, including dtype and alias failures, to ValueError.
fn extract_array<'py, T: Element, const FIELD: usize>(
    value: &Bound<'py, PyAny>,
) -> PyResult<PyReadwriteArrayDyn<'py, T>> {
    let array = value.cast::<PyArrayDyn<T>>().map_err(|error| {
        python_error(ObserveError::new(
            ObserveErrorKind::Dtype,
            FIELD_SHAPES[FIELD].name,
            error.to_string(),
        ))
    })?;
    array.try_readwrite().map_err(|error| {
        python_error(ObserveError::new(
            ObserveErrorKind::Aliased,
            FIELD_SHAPES[FIELD].name,
            format!("mutable writable borrow failed: {error}"),
        ))
    })
}

#[derive(Clone, Copy)]
struct ArrayRange {
    field: &'static str,
    start: usize,
    end: usize,
}

fn array_range<T: Element, const FIELD: usize>(
    array: &PyReadwriteArrayDyn<'_, T>,
    n_envs: usize,
    length: usize,
) -> Result<ArrayRange, ObserveError> {
    let field = FIELD_SHAPES[FIELD];
    let shape = array.shape();
    if shape.len() != field.row_shape.len() + 2
        || shape[0] != n_envs
        || shape[1] != 2
        || shape[2..] != *field.row_shape
    {
        let expected: Vec<usize> = [n_envs, 2]
            .into_iter()
            .chain(field.row_shape.iter().copied())
            .collect();
        return Err(ObserveError::new(
            ObserveErrorKind::Shape,
            field.name,
            format!("expected shape {expected:?}, received {shape:?}"),
        ));
    }
    if !array.is_c_contiguous() {
        return Err(ObserveError::new(
            ObserveErrorKind::NonContiguous,
            field.name,
            "expected C-contiguous layout",
        ));
    }
    if !array.is_aligned() {
        return Err(ObserveError::new(
            ObserveErrorKind::NonContiguous,
            field.name,
            "expected dtype-aligned storage",
        ));
    }
    let bytes = length.checked_mul(field.element_bytes).ok_or_else(|| {
        ObserveError::new(
            ObserveErrorKind::Shape,
            field.name,
            "byte length overflows usize",
        )
    })?;
    let start = array.data() as usize;
    let end = start.checked_add(bytes).ok_or_else(|| {
        ObserveError::new(
            ObserveErrorKind::Shape,
            field.name,
            "byte range overflows usize",
        )
    })?;
    Ok(ArrayRange {
        field: field.name,
        start,
        end,
    })
}

fn check_disjoint(ranges: &[ArrayRange; 29]) -> Result<(), ObserveError> {
    // Distinct Python/Torch base objects can expose the same allocation, which
    // numpy's base-object borrow registry cannot detect. Check actual addresses
    // before constructing even the first mutable Rust slice.
    for (right_index, right) in ranges.iter().enumerate() {
        for left in &ranges[..right_index] {
            if left.start < right.end && right.start < left.end {
                return Err(ObserveError::new(
                    ObserveErrorKind::Aliased,
                    right.field,
                    format!("byte range overlaps {}", left.field),
                ));
            }
        }
    }
    Ok(())
}

fn array_slice<'a, T: Element, const FIELD: usize>(
    array: &'a mut PyReadwriteArrayDyn<'_, T>,
) -> Result<&'a mut [T], ObserveError> {
    array.as_slice_mut().map_err(|error| {
        ObserveError::new(
            ObserveErrorKind::NonContiguous,
            FIELD_SHAPES[FIELD].name,
            error.to_string(),
        )
    })
}

// The contract deliberately exposes every named buffer instead of packing them
// into an opaque dynamic map. Python signature misuse retains normal TypeError;
// supplied header/array admission errors are ValueError with field/env context.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (headers, *,
    tile_kind, tile_crop, tile_animal, tile_cell, tile_role,
    tiles_int, tiles_float, actor_slot, actor_cell, actor_role,
    actor_mask, actor_inventory, actor_inventory_rank, actors_float, player_features,
    storage_counts, storage_rank, banks, shop_type, shop_slot,
    shop_mask, market_product, market_float, market_int, global_features,
    globals_int, still_playing, order_limits, can_act,
))]
pub fn encode_kaggriculture_headers_into(
    py: Python<'_>,
    #[pyo3(from_py_with = extract_headers)] headers: &str,
    #[pyo3(from_py_with = extract_array::<i64, 0>)] mut tile_kind: PyReadwriteArrayDyn<'_, i64>,
    #[pyo3(from_py_with = extract_array::<i64, 1>)] mut tile_crop: PyReadwriteArrayDyn<'_, i64>,
    #[pyo3(from_py_with = extract_array::<i64, 2>)] mut tile_animal: PyReadwriteArrayDyn<'_, i64>,
    #[pyo3(from_py_with = extract_array::<i64, 3>)] mut tile_cell: PyReadwriteArrayDyn<'_, i64>,
    #[pyo3(from_py_with = extract_array::<i64, 4>)] mut tile_role: PyReadwriteArrayDyn<'_, i64>,
    #[pyo3(from_py_with = extract_array::<i64, 5>)] mut tiles_int: PyReadwriteArrayDyn<'_, i64>,
    #[pyo3(from_py_with = extract_array::<f32, 6>)] mut tiles_float: PyReadwriteArrayDyn<'_, f32>,
    #[pyo3(from_py_with = extract_array::<i64, 7>)] mut actor_slot: PyReadwriteArrayDyn<'_, i64>,
    #[pyo3(from_py_with = extract_array::<i64, 8>)] mut actor_cell: PyReadwriteArrayDyn<'_, i64>,
    #[pyo3(from_py_with = extract_array::<i64, 9>)] mut actor_role: PyReadwriteArrayDyn<'_, i64>,
    #[pyo3(from_py_with = extract_array::<bool, 10>)] mut actor_mask: PyReadwriteArrayDyn<'_, bool>,
    #[pyo3(from_py_with = extract_array::<i64, 11>)] mut actor_inventory: PyReadwriteArrayDyn<
        '_,
        i64,
    >,
    #[pyo3(from_py_with = extract_array::<i64, 12>)] mut actor_inventory_rank: PyReadwriteArrayDyn<
        '_,
        i64,
    >,
    #[pyo3(from_py_with = extract_array::<f32, 13>)] mut actors_float: PyReadwriteArrayDyn<'_, f32>,
    #[pyo3(from_py_with = extract_array::<f32, 14>)] mut player_features: PyReadwriteArrayDyn<
        '_,
        f32,
    >,
    #[pyo3(from_py_with = extract_array::<i64, 15>)] mut storage_counts: PyReadwriteArrayDyn<
        '_,
        i64,
    >,
    #[pyo3(from_py_with = extract_array::<i64, 16>)] mut storage_rank: PyReadwriteArrayDyn<'_, i64>,
    #[pyo3(from_py_with = extract_array::<f64, 17>)] mut banks: PyReadwriteArrayDyn<'_, f64>,
    #[pyo3(from_py_with = extract_array::<i64, 18>)] mut shop_type: PyReadwriteArrayDyn<'_, i64>,
    #[pyo3(from_py_with = extract_array::<i64, 19>)] mut shop_slot: PyReadwriteArrayDyn<'_, i64>,
    #[pyo3(from_py_with = extract_array::<bool, 20>)] mut shop_mask: PyReadwriteArrayDyn<'_, bool>,
    #[pyo3(from_py_with = extract_array::<i64, 21>)] mut market_product: PyReadwriteArrayDyn<
        '_,
        i64,
    >,
    #[pyo3(from_py_with = extract_array::<f32, 22>)] mut market_float: PyReadwriteArrayDyn<'_, f32>,
    #[pyo3(from_py_with = extract_array::<i64, 23>)] mut market_int: PyReadwriteArrayDyn<'_, i64>,
    #[pyo3(from_py_with = extract_array::<f32, 24>)] mut global_features: PyReadwriteArrayDyn<
        '_,
        f32,
    >,
    #[pyo3(from_py_with = extract_array::<i64, 25>)] mut globals_int: PyReadwriteArrayDyn<'_, i64>,
    #[pyo3(from_py_with = extract_array::<bool, 26>)] mut still_playing: PyReadwriteArrayDyn<
        '_,
        bool,
    >,
    #[pyo3(from_py_with = extract_array::<i64, 27>)] mut order_limits: PyReadwriteArrayDyn<'_, i64>,
    #[pyo3(from_py_with = extract_array::<bool, 28>)] mut can_act: PyReadwriteArrayDyn<'_, bool>,
) -> PyResult<()> {
    let json: Value = serde_json::from_str(headers).map_err(|error| {
        python_error(ObserveError::new(
            ObserveErrorKind::State,
            "headers",
            error.to_string(),
        ))
    })?;
    let Value::Array(headers) = json else {
        return Err(python_error(ObserveError::new(
            ObserveErrorKind::Shape,
            "headers",
            "expected a JSON array of full TraceHeaders",
        )));
    };
    if headers.is_empty() {
        return Err(python_error(ObserveError::new(
            ObserveErrorKind::Shape,
            "headers",
            "expected at least one header",
        )));
    }
    let n_envs = headers.len();
    let lengths = checked_lengths(n_envs).map_err(python_error)?;
    let ranges = [
        array_range::<i64, 0>(&tile_kind, n_envs, lengths[0]).map_err(python_error)?,
        array_range::<i64, 1>(&tile_crop, n_envs, lengths[1]).map_err(python_error)?,
        array_range::<i64, 2>(&tile_animal, n_envs, lengths[2]).map_err(python_error)?,
        array_range::<i64, 3>(&tile_cell, n_envs, lengths[3]).map_err(python_error)?,
        array_range::<i64, 4>(&tile_role, n_envs, lengths[4]).map_err(python_error)?,
        array_range::<i64, 5>(&tiles_int, n_envs, lengths[5]).map_err(python_error)?,
        array_range::<f32, 6>(&tiles_float, n_envs, lengths[6]).map_err(python_error)?,
        array_range::<i64, 7>(&actor_slot, n_envs, lengths[7]).map_err(python_error)?,
        array_range::<i64, 8>(&actor_cell, n_envs, lengths[8]).map_err(python_error)?,
        array_range::<i64, 9>(&actor_role, n_envs, lengths[9]).map_err(python_error)?,
        array_range::<bool, 10>(&actor_mask, n_envs, lengths[10]).map_err(python_error)?,
        array_range::<i64, 11>(&actor_inventory, n_envs, lengths[11]).map_err(python_error)?,
        array_range::<i64, 12>(&actor_inventory_rank, n_envs, lengths[12]).map_err(python_error)?,
        array_range::<f32, 13>(&actors_float, n_envs, lengths[13]).map_err(python_error)?,
        array_range::<f32, 14>(&player_features, n_envs, lengths[14]).map_err(python_error)?,
        array_range::<i64, 15>(&storage_counts, n_envs, lengths[15]).map_err(python_error)?,
        array_range::<i64, 16>(&storage_rank, n_envs, lengths[16]).map_err(python_error)?,
        array_range::<f64, 17>(&banks, n_envs, lengths[17]).map_err(python_error)?,
        array_range::<i64, 18>(&shop_type, n_envs, lengths[18]).map_err(python_error)?,
        array_range::<i64, 19>(&shop_slot, n_envs, lengths[19]).map_err(python_error)?,
        array_range::<bool, 20>(&shop_mask, n_envs, lengths[20]).map_err(python_error)?,
        array_range::<i64, 21>(&market_product, n_envs, lengths[21]).map_err(python_error)?,
        array_range::<f32, 22>(&market_float, n_envs, lengths[22]).map_err(python_error)?,
        array_range::<i64, 23>(&market_int, n_envs, lengths[23]).map_err(python_error)?,
        array_range::<f32, 24>(&global_features, n_envs, lengths[24]).map_err(python_error)?,
        array_range::<i64, 25>(&globals_int, n_envs, lengths[25]).map_err(python_error)?,
        array_range::<bool, 26>(&still_playing, n_envs, lengths[26]).map_err(python_error)?,
        array_range::<i64, 27>(&order_limits, n_envs, lengths[27]).map_err(python_error)?,
        array_range::<bool, 28>(&can_act, n_envs, lengths[28]).map_err(python_error)?,
    ];
    check_disjoint(&ranges).map_err(python_error)?;
    let mut output = ObsBuffersMut {
        tile_kind: array_slice::<i64, 0>(&mut tile_kind).map_err(python_error)?,
        tile_crop: array_slice::<i64, 1>(&mut tile_crop).map_err(python_error)?,
        tile_animal: array_slice::<i64, 2>(&mut tile_animal).map_err(python_error)?,
        tile_cell: array_slice::<i64, 3>(&mut tile_cell).map_err(python_error)?,
        tile_role: array_slice::<i64, 4>(&mut tile_role).map_err(python_error)?,
        tiles_int: array_slice::<i64, 5>(&mut tiles_int).map_err(python_error)?,
        tiles_float: array_slice::<f32, 6>(&mut tiles_float).map_err(python_error)?,
        actor_slot: array_slice::<i64, 7>(&mut actor_slot).map_err(python_error)?,
        actor_cell: array_slice::<i64, 8>(&mut actor_cell).map_err(python_error)?,
        actor_role: array_slice::<i64, 9>(&mut actor_role).map_err(python_error)?,
        actor_mask: array_slice::<bool, 10>(&mut actor_mask).map_err(python_error)?,
        actor_inventory: array_slice::<i64, 11>(&mut actor_inventory).map_err(python_error)?,
        actor_inventory_rank: array_slice::<i64, 12>(&mut actor_inventory_rank)
            .map_err(python_error)?,
        actors_float: array_slice::<f32, 13>(&mut actors_float).map_err(python_error)?,
        player_features: array_slice::<f32, 14>(&mut player_features).map_err(python_error)?,
        storage_counts: array_slice::<i64, 15>(&mut storage_counts).map_err(python_error)?,
        storage_rank: array_slice::<i64, 16>(&mut storage_rank).map_err(python_error)?,
        banks: array_slice::<f64, 17>(&mut banks).map_err(python_error)?,
        shop_type: array_slice::<i64, 18>(&mut shop_type).map_err(python_error)?,
        shop_slot: array_slice::<i64, 19>(&mut shop_slot).map_err(python_error)?,
        shop_mask: array_slice::<bool, 20>(&mut shop_mask).map_err(python_error)?,
        market_product: array_slice::<i64, 21>(&mut market_product).map_err(python_error)?,
        market_float: array_slice::<f32, 22>(&mut market_float).map_err(python_error)?,
        market_int: array_slice::<i64, 23>(&mut market_int).map_err(python_error)?,
        global_features: array_slice::<f32, 24>(&mut global_features).map_err(python_error)?,
        globals_int: array_slice::<i64, 25>(&mut globals_int).map_err(python_error)?,
        still_playing: array_slice::<bool, 26>(&mut still_playing).map_err(python_error)?,
        order_limits: array_slice::<i64, 27>(&mut order_limits).map_err(python_error)?,
        can_act: array_slice::<bool, 28>(&mut can_act).map_err(python_error)?,
    }
    .validate(n_envs)
    .map_err(python_error)?;
    // Every typed borrow guard remains alive outside the detached operation.
    // Scratch holds only parsing/config/snapshots, never another output array.
    py.detach(move || -> Result<(), ObserveError> {
        let mut prepared = Vec::new();
        prepared.try_reserve_exact(n_envs).map_err(|error| {
            ObserveError::new(
                ObserveErrorKind::Shape,
                "headers",
                format!("cannot reserve preparation scratch: {error}"),
            )
        })?;
        for (env, value) in headers.into_iter().enumerate() {
            let header: TraceHeader =
                serde_json::from_value(value).map_err(|error| ObserveError {
                    env: Some(env),
                    ..ObserveError::new(
                        ObserveErrorKind::State,
                        format!("headers[{env}]"),
                        error.to_string(),
                    )
                })?;
            let observation = ObservationGame::from_header(&header)
                .and_then(|game| game.prepare())
                .map_err(|mut error| {
                    error.env = Some(env);
                    error
                })?;
            prepared.push(observation);
        }
        // All rows preflighted. Only infallible writes follow the first mutation.
        for (env, mut rows) in output.envs_mut().enumerate() {
            write_env(&prepared[env], &mut rows);
        }
        Ok(())
    })
    .map_err(python_error)
}

mod bindings;

pub(super) fn add_to_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(encode_kaggriculture_headers_into, module)?)?;
    module.add_function(wrap_pyfunction!(export_kaggriculture_episode, module)?)?;
    module.add_function(wrap_pyfunction!(verify_kaggriculture_episode, module)?)?;
    module.add_class::<bindings::PyKaggricultureEnv>()?;
    module.add_function(wrap_pyfunction!(bindings::kaggriculture_encode, module)?)?;
    module.add_function(wrap_pyfunction!(bindings::kaggriculture_decode, module)?)?;
    module.add_function(wrap_pyfunction!(
        bindings::kaggriculture_grammar_tables,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        bindings::kaggriculture_grammar_constants,
        module
    )?)
}

#[pyfunction]
fn export_kaggriculture_episode(
    py: Python<'_>,
    seed_header_json: &str,
    tape_json: &str,
) -> PyResult<String> {
    validate_replay_framework(py, seed_header_json, false)?;
    py.detach(|| {
        let header: Value = serde_json::from_str(seed_header_json).map_err(|e| e.to_string())?;
        let tape: Value = serde_json::from_str(tape_json).map_err(|e| e.to_string())?;
        let header = replay_export::SeedHeader::parse(&header).map_err(|e| e.to_string())?;
        let tape = replay_export::ActionTape::parse(&tape).map_err(|e| e.to_string())?;
        replay_export::export_kaggle_episode(&header, &tape)
            .map(|v| v.to_string())
            .map_err(|e| e.to_string())
    })
    .map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (episode_json, captured_json=None))]
fn verify_kaggriculture_episode(
    py: Python<'_>,
    episode_json: &str,
    captured_json: Option<&str>,
) -> PyResult<String> {
    validate_replay_framework(py, episode_json, true)?;
    py.detach(|| {
        let captured: Option<Value> = captured_json
            .map(serde_json::from_str)
            .transpose()
            .map_err(|e| e.to_string())?;
        replay_export::verify_round_trip(episode_json, captured.as_ref())
            .map(|v| v.to_json().to_string())
            .map_err(|e| e.to_string())
    })
    .map_err(PyValueError::new_err)
}

fn validate_replay_framework(py: Python<'_>, json: &str, episode: bool) -> PyResult<()> {
    // The pure Rust adapter takes an already verified specification. The public
    // Python seam must also reject missing/drifted installed framework sources.
    PyModule::import(py, "owl.kaggriculture.replay_export")
        .and_then(|module| module.call_method1("validate_native_replay_input", (json, episode)))
        .map(|_| ())
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

mod admission;
mod env;
#[cfg(test)]
mod env_tests;
mod reward;

#[cfg(test)]
mod lifecycle_timing_tests;
