//! Closed-form conjugate posterior updates.
//!
//! Gelman, Carlin, Stern, Dunson, Vehtari & Rubin (2013), "Bayesian Data Analysis", 3rd ed.,
//! CRC Press, DOI 10.1201/b16018, chapters 2 and 3. Decision D-2.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn positive(name: &str, value: f64) -> PyResult<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!("{name} must be finite and > 0, got {value}")))
    }
}

fn non_negative(name: &str, value: f64) -> PyResult<()> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!("{name} must be finite and >= 0, got {value}")))
    }
}

/// Beta(a, b) prior, `successes` out of `trials`: Beta(a + successes, b + trials - successes).
#[pyfunction]
pub fn beta_binomial(a: f64, b: f64, successes: f64, trials: f64) -> PyResult<(f64, f64)> {
    positive("alpha", a)?;
    positive("beta", b)?;
    non_negative("successes", successes)?;
    non_negative("trials", trials)?;
    if successes > trials {
        return Err(PyValueError::new_err(format!(
            "successes {successes} exceed trials {trials}"
        )));
    }
    Ok((a + successes, b + trials - successes))
}

/// Gamma(shape, rate) prior, total count over total exposure: Gamma(shape + count, rate + exposure).
#[pyfunction]
pub fn gamma_poisson(shape: f64, rate: f64, count: f64, exposure: f64) -> PyResult<(f64, f64)> {
    positive("shape", shape)?;
    positive("rate", rate)?;
    non_negative("count", count)?;
    non_negative("exposure", exposure)?;
    Ok((shape + count, rate + exposure))
}

/// Normal(mean0, sd0) prior on the mean, n observations with sample mean `xbar` and known sd
/// `sigma`. Precisions add; the posterior mean is the precision-weighted average.
#[pyfunction]
pub fn normal_normal(mean0: f64, sd0: f64, sigma: f64, n: f64, xbar: f64) -> PyResult<(f64, f64)> {
    positive("sd", sd0)?;
    positive("sigma", sigma)?;
    non_negative("n", n)?;
    if !mean0.is_finite() || !xbar.is_finite() {
        return Err(PyValueError::new_err("prior mean and sample mean must be finite"));
    }
    let prior_precision = 1.0 / (sd0 * sd0);
    let data_precision = n / (sigma * sigma);
    let precision = prior_precision + data_precision;
    let mean = (prior_precision * mean0 + data_precision * xbar) / precision;
    Ok((mean, (1.0 / precision).sqrt()))
}

/// Dirichlet(alpha) prior, multinomial counts: Dirichlet(alpha + counts).
#[pyfunction]
pub fn dirichlet_multinomial(alpha: Vec<f64>, counts: Vec<f64>) -> PyResult<Vec<f64>> {
    if alpha.len() != counts.len() {
        return Err(PyValueError::new_err(format!(
            "alpha has {} categories, counts has {}",
            alpha.len(),
            counts.len()
        )));
    }
    if alpha.len() < 2 {
        return Err(PyValueError::new_err("need at least 2 categories"));
    }
    alpha
        .iter()
        .zip(&counts)
        .map(|(&a, &c)| {
            positive("alpha", a)?;
            non_negative("count", c)?;
            Ok(a + c)
        })
        .collect()
}
