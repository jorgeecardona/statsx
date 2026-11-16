//! Duration model: Bayesian linear regression with a conjugate Normal-Inverse-Gamma prior.
//!
//! Gelman, Carlin, Stern, Dunson, Vehtari & Rubin (2013), "Bayesian Data Analysis", 3rd ed.,
//! chapter 14, DOI 10.1201/b16018. O'Hagan & Forster (2004), "Kendall's Advanced Theory of
//! Statistics, Vol. 2B: Bayesian Inference", 2nd ed., chapter 11 (normal linear model with
//! conjugate prior). Decision D-7.

use crate::linalg::{cho_inverse, cho_solve, cholesky};
use crate::rng::Rng;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Posterior (mean, covariance factor V_n, shape a_n, rate b_n) of y = X beta + e,
/// e ~ N(0, s2), with prior beta | s2 ~ N(0, s2 * coef_var * I), s2 ~ InvGamma(a0, b0).
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn nig_fit<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    coef_var: f64,
    a0: f64,
    b0: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>, f64, f64)> {
    let xv = x.as_array();
    let yv = y.as_slice()?;
    let (n, p) = xv.dim();
    if n != yv.len() {
        return Err(PyValueError::new_err(format!("{n} rows in X, {} values in y", yv.len())));
    }
    if !(coef_var > 0.0 && a0 > 0.0 && b0 > 0.0) {
        return Err(PyValueError::new_err("prior parameters must be > 0"));
    }
    // Precision V_n^-1 = I / coef_var + X^T X ; X^T y.
    let mut prec = vec![0.0; p * p];
    let mut xty = vec![0.0; p];
    for r in 0..n {
        for i in 0..p {
            xty[i] += xv[[r, i]] * yv[r];
            for j in 0..p {
                prec[i * p + j] += xv[[r, i]] * xv[[r, j]];
            }
        }
    }
    for i in 0..p {
        prec[i * p + i] += 1.0 / coef_var;
    }
    let l = cholesky(&prec, p)
        .ok_or_else(|| PyValueError::new_err("posterior precision is not positive definite"))?;
    let mean = cho_solve(&l, p, &xty);
    let cov = cho_inverse(&l, p);
    // b_n = b0 + (y^T y - m_n^T V_n^-1 m_n) / 2, and V_n^-1 m_n = X^T y.
    let yty: f64 = yv.iter().map(|v| v * v).sum();
    let fit: f64 = mean.iter().zip(&xty).map(|(m, t)| m * t).sum();
    let an = a0 + n as f64 / 2.0;
    let bn = b0 + 0.5 * (yty - fit).max(0.0);
    let cov = numpy::ndarray::Array2::from_shape_vec((p, p), cov)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((mean.into_pyarray_bound(py), cov.into_pyarray_bound(py), an, bn))
}

/// Posterior predictive draws of y at each row of `x_new`: shape (rows, n_draws).
/// Each draw takes s2 ~ InvGamma(a_n, b_n), then y ~ N(x^T m_n, s2 (1 + x^T V_n x)), which is
/// the Student-t predictive with 2 a_n degrees of freedom.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn nig_predict<'py>(
    py: Python<'py>,
    mean: PyReadonlyArray1<'py, f64>,
    cov: PyReadonlyArray2<'py, f64>,
    an: f64,
    bn: f64,
    x_new: PyReadonlyArray2<'py, f64>,
    n_draws: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let m = mean.as_slice()?.to_vec();
    let v = cov.as_array().to_owned();
    let xn = x_new.as_array().to_owned();
    let (rows, p) = xn.dim();
    if p != m.len() {
        return Err(PyValueError::new_err(format!("x_new has {p} columns, model has {}", m.len())));
    }
    let out = py.allow_threads(|| {
        let mut rng = Rng::new(seed);
        let mut out = numpy::ndarray::Array2::<f64>::zeros((rows, n_draws));
        for r in 0..rows {
            let loc: f64 = (0..p).map(|i| xn[[r, i]] * m[i]).sum();
            let mut quad = 0.0;
            for i in 0..p {
                for j in 0..p {
                    quad += xn[[r, i]] * v[[i, j]] * xn[[r, j]];
                }
            }
            for d in 0..n_draws {
                let s2 = bn / rng.gamma(an);
                out[[r, d]] = loc + (s2 * (1.0 + quad)).sqrt() * rng.normal();
            }
        }
        out
    });
    Ok(out.into_pyarray_bound(py))
}
