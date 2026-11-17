//! Recalibration of classifier scores: isotonic regression and Platt scaling.
//!
//! Isotonic regression by the pool-adjacent-violators algorithm: Barlow, Bartholomew, Bremner &
//! Brunk (1972), "Statistical Inference under Order Restrictions", Wiley, with the linear-time
//! formulation of Best & Chakravarti (1990), "Active set algorithms for isotonic regression; a
//! unifying framework", Mathematical Programming 47:425-439, DOI 10.1007/BF01580873. Platt
//! scaling by the Newton method with backtracking of Lin, Lin & Weng (2007), "A note on Platt's
//! probabilistic outputs for support vector machines", Machine Learning 68(3):267-276, DOI
//! 10.1007/s10994-007-5018-6, with Platt's (1999) smoothed targets. Decision D-32.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Weighted isotonic (non-decreasing) regression of `y` on its index: the fitted values that
/// minimize the weighted squared error under the order constraint.
pub fn pava(y: &[f64], w: &[f64]) -> Vec<f64> {
    // Each block keeps its weighted mean, total weight and length.
    let mut mean: Vec<f64> = Vec::with_capacity(y.len());
    let mut weight: Vec<f64> = Vec::with_capacity(y.len());
    let mut len: Vec<usize> = Vec::with_capacity(y.len());
    for (&yi, &wi) in y.iter().zip(w) {
        mean.push(yi);
        weight.push(wi);
        len.push(1);
        while mean.len() > 1 && mean[mean.len() - 2] > mean[mean.len() - 1] {
            let (m2, w2, l2) = (mean.pop().unwrap(), weight.pop().unwrap(), len.pop().unwrap());
            let last = mean.len() - 1;
            let total = weight[last] + w2;
            mean[last] = (mean[last] * weight[last] + m2 * w2) / total;
            weight[last] = total;
            len[last] += l2;
        }
    }
    let mut out = Vec::with_capacity(y.len());
    for (m, l) in mean.iter().zip(&len) {
        out.extend(std::iter::repeat_n(*m, *l));
    }
    out
}

fn log1p_exp(x: f64) -> f64 {
    if x > 30.0 {
        x
    } else if x < -30.0 {
        x.exp()
    } else {
        x.exp().ln_1p()
    }
}

/// Fit p = 1 / (1 + exp(-(a f + b))) to scores `f` and 0/1 labels. Returns (a, b).
pub fn platt(f: &[f64], y: &[f64]) -> (f64, f64) {
    let n_pos = y.iter().filter(|v| **v > 0.5).count() as f64;
    let n_neg = y.len() as f64 - n_pos;
    // Targets smoothed by the prior counts (Platt 1999) so a separable sample has a finite fit.
    let hi = (n_pos + 1.0) / (n_pos + 2.0);
    let lo = 1.0 / (n_neg + 2.0);
    let t: Vec<f64> = y.iter().map(|v| if *v > 0.5 { hi } else { lo }).collect();
    // Lin et al. use p = 1 / (1 + exp(A f + B)); a = -A and b = -B.
    let (mut a_p, mut b_p) = (0.0, ((n_neg + 1.0) / (n_pos + 1.0)).ln());
    let objective = |a: f64, b: f64| -> f64 {
        f.iter()
            .zip(&t)
            .map(|(fi, ti)| {
                let z = fi * a + b;
                // -[t log p + (1 - t) log(1 - p)] with p = 1 / (1 + exp(z)).
                (ti - 1.0) * z + log1p_exp(-z)
            })
            .sum()
    };
    let mut fval = objective(a_p, b_p);
    for _ in 0..100 {
        let (mut h11, mut h22, mut h21, mut g1, mut g2) = (1e-12, 1e-12, 0.0, 0.0, 0.0);
        for (fi, ti) in f.iter().zip(&t) {
            let z = fi * a_p + b_p;
            let p = if z >= 0.0 { (-z).exp() / (1.0 + (-z).exp()) } else { 1.0 / (1.0 + z.exp()) };
            let q = 1.0 - p;
            let d2 = p * q;
            h11 += fi * fi * d2;
            h22 += d2;
            h21 += fi * d2;
            let d1 = ti - p;
            g1 += fi * d1;
            g2 += d1;
        }
        if g1.abs() < 1e-5 && g2.abs() < 1e-5 {
            break;
        }
        let det = h11 * h22 - h21 * h21;
        let (da, db) = (-(h22 * g1 - h21 * g2) / det, -(-h21 * g1 + h11 * g2) / det);
        let gd = g1 * da + g2 * db;
        let mut step = 1.0;
        while step >= 1e-10 {
            let (na, nb) = (a_p + step * da, b_p + step * db);
            let nf = objective(na, nb);
            if nf < fval + 1e-4 * step * gd {
                a_p = na;
                b_p = nb;
                fval = nf;
                break;
            }
            step /= 2.0;
        }
        if step < 1e-10 {
            break;
        }
    }
    (-a_p, -b_p)
}

/// Isotonic regression of `y` with weights `w`, both in the order of the sorted scores.
#[pyfunction]
pub fn isotonic_fit<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    w: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (yv, wv) = (y.as_slice()?, w.as_slice()?);
    if yv.len() != wv.len() || yv.is_empty() {
        return Err(PyValueError::new_err("y and w must be non-empty and of the same length"));
    }
    if wv.iter().any(|v| !(*v > 0.0 && v.is_finite())) || yv.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("y must be finite and w finite and > 0"));
    }
    Ok(py.allow_threads(|| pava(yv, wv)).into_pyarray_bound(py))
}

/// Platt scaling: (a, b) of p = 1 / (1 + exp(-(a score + b))).
#[pyfunction]
pub fn platt_fit(scores: PyReadonlyArray1<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<(f64, f64)> {
    let (f, yv) = (scores.as_slice()?, y.as_slice()?);
    if f.len() != yv.len() || f.is_empty() {
        return Err(PyValueError::new_err("scores and y must be non-empty and of the same length"));
    }
    if f.iter().any(|v| !v.is_finite()) || yv.iter().any(|v| *v != 0.0 && *v != 1.0) {
        return Err(PyValueError::new_err("scores must be finite and y must be 0 or 1"));
    }
    Ok(platt(f, yv))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pava_matches_the_textbook_example() {
        let got = pava(&[1.0, 3.0, 2.0, 4.0, 3.0, 5.0], &[1.0; 6]);
        assert_eq!(got, vec![1.0, 2.5, 2.5, 3.5, 3.5, 5.0]);
    }

    #[test]
    fn pava_respects_weights_and_leaves_ordered_data_alone() {
        // 4 with weight 3 pooled with 2 with weight 1: (12 + 2) / 4 = 3.5.
        assert_eq!(pava(&[4.0, 2.0], &[3.0, 1.0]), vec![3.5, 3.5]);
        assert_eq!(pava(&[1.0, 2.0, 2.0, 5.0], &[1.0; 4]), vec![1.0, 2.0, 2.0, 5.0]);
        assert_eq!(pava(&[3.0, 2.0, 1.0], &[1.0; 3]), vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn platt_recovers_the_logistic_coefficients() {
        let mut rng = crate::rng::Rng::new(5);
        let f: Vec<f64> = (0..20000).map(|_| 2.0 * rng.normal()).collect();
        let y: Vec<f64> = f
            .iter()
            .map(|v| if rng.uniform() < 1.0 / (1.0 + (-(0.7 * v - 0.3)).exp()) { 1.0 } else { 0.0 })
            .collect();
        let (a, b) = platt(&f, &y);
        assert!((a - 0.7).abs() < 0.03 && (b + 0.3).abs() < 0.05, "{a} {b}");
    }

    #[test]
    fn platt_has_a_finite_fit_for_separable_scores() {
        let f = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0];
        let y = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let (a, b) = platt(&f, &y);
        assert!(a.is_finite() && a > 0.0 && b.abs() < 1e-6, "{a} {b}");
    }
}
