//! Draws from Beta, Gamma and Normal distributions with the xoshiro256++ generator of `rng`.
//!
//! Gamma by Marsaglia & Tsang (2000), "A simple method for generating gamma variables", ACM
//! Transactions on Mathematical Software 26(3):363-372, DOI 10.1145/358407.358414 (in `rng`).
//! Beta as X / (X + Y) for independent gammas X and Y (Gelman et al. 2013, "Bayesian Data
//! Analysis", 3rd ed., appendix A), computed from log gammas so a small shape cannot underflow
//! to 0 / 0. Decision D-4 (generator) and D-24.

use crate::rng::Rng;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// log of a Gamma(shape, 1) draw. For shape < 1, ln G(shape + 1) + ln(U) / shape.
fn ln_gamma_draw(rng: &mut Rng, shape: f64) -> f64 {
    if shape < 1.0 {
        rng.gamma(shape + 1.0).ln() + rng.uniform().ln() / shape
    } else {
        rng.gamma(shape).ln()
    }
}

fn positive(values: &[f64]) -> PyResult<()> {
    if values.iter().all(|v| v.is_finite() && *v > 0.0) {
        Ok(())
    } else {
        Err(PyValueError::new_err("parameters must be finite and > 0"))
    }
}

/// `n` draws from Beta(a, b).
#[pyfunction]
pub fn sample_beta<'py>(py: Python<'py>, a: f64, b: f64, n: usize, seed: u64) -> PyResult<Bound<'py, PyArray1<f64>>> {
    positive(&[a, b])?;
    let out = py.allow_threads(|| {
        let mut rng = Rng::new(seed);
        (0..n)
            .map(|_| {
                let (lx, ly) = (ln_gamma_draw(&mut rng, a), ln_gamma_draw(&mut rng, b));
                1.0 / (1.0 + (ly - lx).exp())
            })
            .collect::<Vec<f64>>()
    });
    Ok(out.into_pyarray_bound(py))
}

/// `n` draws from Gamma(shape, rate).
#[pyfunction]
pub fn sample_gamma<'py>(py: Python<'py>, shape: f64, rate: f64, n: usize, seed: u64) -> PyResult<Bound<'py, PyArray1<f64>>> {
    positive(&[shape, rate])?;
    let out = py.allow_threads(|| {
        let mut rng = Rng::new(seed);
        (0..n).map(|_| rng.gamma(shape) / rate).collect::<Vec<f64>>()
    });
    Ok(out.into_pyarray_bound(py))
}

/// `n` draws from Normal(mean, sd).
#[pyfunction]
pub fn sample_normal<'py>(py: Python<'py>, mean: f64, sd: f64, n: usize, seed: u64) -> PyResult<Bound<'py, PyArray1<f64>>> {
    positive(&[sd])?;
    if !mean.is_finite() {
        return Err(PyValueError::new_err("mean must be finite"));
    }
    let out = py.allow_threads(|| {
        let mut rng = Rng::new(seed);
        (0..n).map(|_| mean + sd * rng.normal()).collect::<Vec<f64>>()
    });
    Ok(out.into_pyarray_bound(py))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn moments(x: &[f64]) -> (f64, f64) {
        let n = x.len() as f64;
        let m = x.iter().sum::<f64>() / n;
        (m, x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n - 1.0))
    }

    #[test]
    fn beta_has_the_right_mean_and_variance_even_for_tiny_shapes() {
        for &(a, b) in &[(2.0, 5.0), (0.5, 0.5), (0.02, 0.03), (30.0, 10.0)] {
            let mut rng = Rng::new(3);
            let x: Vec<f64> = (0..200_000)
                .map(|_| {
                    let (lx, ly) = (ln_gamma_draw(&mut rng, a), ln_gamma_draw(&mut rng, b));
                    1.0 / (1.0 + (ly - lx).exp())
                })
                .collect();
            assert!(x.iter().all(|v| (0.0..=1.0).contains(v)), "{a} {b}");
            let (m, v) = moments(&x);
            let s = a + b;
            assert!((m - a / s).abs() < 0.01, "mean {a} {b}: {m}");
            assert!((v - a * b / (s * s * (s + 1.0))).abs() < 0.01, "var {a} {b}: {v}");
        }
    }
}
