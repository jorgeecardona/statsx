//! Rust core of statsx, compiled by maturin into the `statsx._accel` extension module.
use pyo3::prelude::*;

mod conjugate;

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
    Ok(())
}
