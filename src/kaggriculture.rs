//! PyO3 array boundary for the vendored Kaggriculture kernel.
//!
//! Uses the starter's maturin extension and caller-owned NumPy/Torch storage.
//! The transactional native kernel retains its private Rayon pool. Its legacy
//! C-shaped functions are called within Rust; no Python ctypes or per-step JSON.
use kaggriculture_engine::{ffi, myolie_sampler};
use numpy::{PyReadonlyArrayDyn, PyReadwriteArrayDyn};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::Mutex;

fn take(buffer: ffi::ReBuffer) -> Vec<u8> {
    let bytes = if buffer.len == 0 {
        Vec::new()
    } else {
        // SAFETY: the engine owns this valid allocation until it is freed below.
        unsafe { std::slice::from_raw_parts(buffer.data, buffer.len).to_vec() }
    };
    unsafe { ffi::re_buffer_free(buffer) };
    bytes
}

fn last_error() -> String {
    String::from_utf8_lossy(&take(ffi::re_batch_last_error())).into_owned()
}

fn checked(ok: bool) -> Result<(), String> {
    if ok {
        Ok(())
    } else {
        Err(last_error())
    }
}

#[pyclass(name = "KaggricultureBatch")]
struct PyKaggricultureBatch {
    batch: Mutex<Option<Box<ffi::ReBatch>>>,
}

impl PyKaggricultureBatch {
    fn with_batch<T>(
        &self,
        f: impl FnOnce(*mut ffi::ReBatch) -> Result<T, String>,
    ) -> Result<T, String> {
        let mut guard = self
            .batch
            .lock()
            .map_err(|_| "Kaggriculture batch mutex poisoned")?;
        let batch = guard.as_mut().ok_or("Kaggriculture batch is closed")?;
        f(&mut **batch)
    }
}

#[pymethods]
impl PyKaggricultureBatch {
    #[new]
    #[pyo3(signature = (headers, threads=1))]
    fn new(py: Python<'_>, headers: &str, threads: usize) -> PyResult<Self> {
        if threads < 1 {
            return Err(PyValueError::new_err("threads must be positive"));
        }
        py.detach(|| {
            let ptr = unsafe { ffi::re_batch_create(headers.as_ptr(), headers.len()) };
            if ptr.is_null() {
                return Err(last_error());
            }
            // SAFETY: the constructor transfers a unique heap allocation.
            let mut batch = unsafe { Box::from_raw(ptr) };
            checked(unsafe { ffi::re_batch_set_threads(&mut *batch, threads) })?;
            Ok(Self {
                batch: Mutex::new(Some(batch)),
            })
        })
        .map_err(PyRuntimeError::new_err)
    }

    fn reset(&self, py: Python<'_>, headers: &str) -> PyResult<()> {
        py.detach(|| {
            self.with_batch(|batch| {
                checked(unsafe { ffi::re_batch_reset(batch, headers.as_ptr(), headers.len()) })
            })
        })
        .map_err(PyRuntimeError::new_err)
    }

    fn reset_indices(
        &self,
        py: Python<'_>,
        indices: PyReadonlyArrayDyn<'_, u32>,
        headers: &str,
        version: u32,
    ) -> PyResult<()> {
        let indices = indices.as_slice()?;
        py.detach(|| {
            self.with_batch(|batch| {
                let f = match version {
                    1 => ffi::re_batch_myolie_reset_indices,
                    2 => ffi::re_batch_myolie_reset_indices_v2,
                    _ => return Err("unsupported observation version".into()),
                };
                checked(unsafe {
                    f(
                        batch,
                        indices.as_ptr(),
                        indices.len(),
                        headers.as_ptr(),
                        headers.len(),
                    )
                })
            })
        })
        .map_err(PyRuntimeError::new_err)
    }

    #[allow(clippy::too_many_arguments)]
    fn observe(
        &self,
        py: Python<'_>,
        mut features: PyReadwriteArrayDyn<'_, f32>,
        mut context: PyReadwriteArrayDyn<'_, i64>,
        mut banks: PyReadwriteArrayDyn<'_, f64>,
        mut done: PyReadwriteArrayDyn<'_, u8>,
        version: u32,
    ) -> PyResult<()> {
        let features = features.as_slice_mut()?;
        let context = context.as_slice_mut()?;
        let banks = banks.as_slice_mut()?;
        let done = done.as_slice_mut()?;
        py.detach(|| {
            self.with_batch(|batch| {
                let f = match version {
                    1 => ffi::re_batch_myolie_observe,
                    2 => ffi::re_batch_myolie_observe_v2,
                    _ => return Err("unsupported observation version".into()),
                };
                checked(unsafe {
                    f(
                        batch,
                        features.as_mut_ptr(),
                        features.len(),
                        context.as_mut_ptr(),
                        context.len(),
                        banks.as_mut_ptr(),
                        banks.len(),
                        done.as_mut_ptr(),
                        done.len(),
                    )
                })
            })
        })
        .map_err(PyRuntimeError::new_err)
    }

    #[allow(clippy::too_many_arguments)]
    fn step_frames(
        &self,
        py: Python<'_>,
        frames: PyReadonlyArrayDyn<'_, i16>,
        lengths: PyReadonlyArrayDyn<'_, i32>,
        mut features: PyReadwriteArrayDyn<'_, f32>,
        mut context: PyReadwriteArrayDyn<'_, i64>,
        mut banks: PyReadwriteArrayDyn<'_, f64>,
        mut done: PyReadwriteArrayDyn<'_, u8>,
        version: u32,
    ) -> PyResult<()> {
        let frames = frames.as_slice()?;
        let lengths = lengths.as_slice()?;
        let features = features.as_slice_mut()?;
        let context = context.as_slice_mut()?;
        let banks = banks.as_slice_mut()?;
        let done = done.as_slice_mut()?;
        py.detach(|| {
            self.with_batch(|batch| {
                let f = match version {
                    1 => ffi::re_batch_myolie_step_frames,
                    2 => ffi::re_batch_myolie_step_frames_v2,
                    _ => return Err("unsupported observation version".into()),
                };
                checked(unsafe {
                    f(
                        batch,
                        frames.as_ptr(),
                        frames.len(),
                        lengths.as_ptr(),
                        lengths.len(),
                        features.as_mut_ptr(),
                        features.len(),
                        context.as_mut_ptr(),
                        context.len(),
                        banks.as_mut_ptr(),
                        banks.len(),
                        done.as_mut_ptr(),
                        done.len(),
                    )
                })
            })
        })
        .map_err(PyRuntimeError::new_err)
    }

    fn econ(&self, py: Python<'_>, mut counters: PyReadwriteArrayDyn<'_, u64>) -> PyResult<()> {
        let counters = counters.as_slice_mut()?;
        py.detach(|| {
            self.with_batch(|batch| {
                checked(unsafe {
                    ffi::re_batch_myolie_econ(batch, counters.as_mut_ptr(), counters.len())
                })
            })
        })
        .map_err(PyRuntimeError::new_err)
    }

    fn snapshots(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| {
            self.with_batch(|batch| {
                let bytes = take(unsafe { ffi::re_batch_snapshots(batch) });
                if bytes.is_empty() {
                    return Err(last_error());
                }
                String::from_utf8(bytes).map_err(|e| e.to_string())
            })
        })
        .map_err(PyRuntimeError::new_err)
    }

    fn close(&self) -> PyResult<()> {
        self.batch
            .lock()
            .map_err(|_| PyRuntimeError::new_err("batch mutex poisoned"))?
            .take();
        Ok(())
    }
}

#[pyfunction]
fn kaggriculture_seed_header(py: Python<'_>, request: &str) -> PyResult<String> {
    py.detach(|| {
        let bytes = take(unsafe { ffi::re_native_seed_header(request.as_ptr(), request.len()) });
        if bytes.is_empty() {
            return Err(last_error());
        }
        String::from_utf8(bytes).map_err(|e| e.to_string())
    })
    .map_err(PyValueError::new_err)
}

#[pyfunction]
fn kaggriculture_grammar_plan(
    py: Python<'_>,
    actors: u32,
    orders: u32,
    hire_limit: u32,
) -> PyResult<Bound<'_, PyBytes>> {
    let bytes = py
        .detach(|| myolie_sampler::plan(actors, orders, hire_limit))
        .map_err(PyValueError::new_err)?;
    Ok(PyBytes::new(py, &bytes))
}

#[pyfunction]
fn kaggriculture_decode(
    py: Python<'_>,
    actors: u32,
    orders: u32,
    hire_limit: u32,
    frames: PyReadonlyArrayDyn<'_, i16>,
) -> PyResult<String> {
    let frames = frames.as_slice()?;
    py.detach(|| myolie_sampler::decode(actors, orders, hire_limit, frames).map(|v| v.to_string()))
        .map_err(PyValueError::new_err)
}

pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKaggricultureBatch>()?;
    m.add_function(wrap_pyfunction!(kaggriculture_seed_header, m)?)?;
    m.add_function(wrap_pyfunction!(kaggriculture_grammar_plan, m)?)?;
    m.add_function(wrap_pyfunction!(kaggriculture_decode, m)?)?;
    Ok(())
}
