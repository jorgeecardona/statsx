//! Rust core of statsx, compiled by maturin into the `statsx._accel` extension module.
// pyo3 0.22 #[pyfunction] expands to a PyErr -> PyErr conversion that clippy flags.
#![allow(clippy::useless_conversion)]
// Index loops over several matching arrays read closer to the formulas in the papers.
#![allow(clippy::needless_range_loop)]
use pyo3::prelude::*;

mod arrow;
mod beta;
mod bootstrap;
mod calibrate;
mod conjugate;
mod diagnostics;
mod eta;
mod gamma;
mod gmm;
mod kmeans;
mod linalg;
// Used by the D-1 posterior method, which waits on the maintainer's choice.
#[allow(dead_code)]
mod negbin;
mod psis;
mod rng;
mod sample;
mod special;
mod structural;

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
    m.add_function(wrap_pyfunction!(diagnostics::effective_sample_size, m)?)?;
    m.add_function(wrap_pyfunction!(eta::nig_fit, m)?)?;
    m.add_function(wrap_pyfunction!(eta::nig_predict, m)?)?;
    m.add_function(wrap_pyfunction!(structural::fit_structural, m)?)?;
    m.add_function(wrap_pyfunction!(beta::beta_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(beta::beta_ppf, m)?)?;
    m.add_function(wrap_pyfunction!(beta::t_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(calibrate::isotonic_fit, m)?)?;
    m.add_function(wrap_pyfunction!(calibrate::platt_fit, m)?)?;
    m.add_function(wrap_pyfunction!(beta::t_ppf, m)?)?;
    m.add_function(wrap_pyfunction!(beta::nct_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(gamma::chi2_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(gamma::chi2_sf, m)?)?;
    m.add_function(wrap_pyfunction!(gmm::gmm_fit, m)?)?;
    m.add_function(wrap_pyfunction!(gmm::gmm_score, m)?)?;
    m.add_function(wrap_pyfunction!(kmeans::kmeans_fit, m)?)?;
    m.add_function(wrap_pyfunction!(kmeans::kmeans_assign, m)?)?;
    m.add_function(wrap_pyfunction!(sample::sample_beta, m)?)?;
    m.add_function(wrap_pyfunction!(sample::sample_gamma, m)?)?;
    m.add_function(wrap_pyfunction!(sample::sample_normal, m)?)?;
    Ok(())
}
