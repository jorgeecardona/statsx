//! Rust core of statsx, compiled by maturin into the `statsx._accel` extension module.
// pyo3 0.22 #[pyfunction] expands to a PyErr -> PyErr conversion that clippy flags.
#![allow(clippy::useless_conversion)]
use pyo3::prelude::*;

mod bootstrap;
mod conjugate;
mod diagnostics;
mod rng;
mod special;

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn _accel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(conjugate::beta_binomial, m)?)?;
    m.add_function(wrap_pyfunction!(conjugate::gamma_poisson, m)?)?;
    m.add_function(wrap_pyfunction!(conjugate::normal_normal, m)?)?;
    m.add_function(wrap_pyfunction!(conjugate::dirichlet_multinomial, m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap::bootstrap_indices, m)?)?;
    m.add_function(wrap_pyfunction!(diagnostics::convergence, m)?)?;
    Ok(())
}
