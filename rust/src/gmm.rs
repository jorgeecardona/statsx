//! Gaussian mixtures by expectation maximization (EM).
//!
//! Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977), "Maximum likelihood from incomplete data
//! via the EM algorithm", Journal of the Royal Statistical Society B 39(1):1-38. The E-step is
//! the posterior probability of each component for each row, by log-sum-exp. The M-step is the
//! weighted mean and covariance. Covariances are full, diagonal or spherical. A small ridge
//! `reg` on the diagonal keeps a covariance positive definite. Decision D-64.
//!
//! A run starts from a k-means partition (Lloyd 1982, k-means++ seeds, see `kmeans.rs`). A run is
//! dropped as degenerate when a component keeps fewer than `min_count` points in effect: that is
//! the unbounded-likelihood solution of a component collapsing onto few points (McLachlan & Peel
//! 2000, section 3.8), not a fit. The best of the other runs, by log-likelihood, is kept.
//!
//! The result does not depend on the number of threads. Rows go in blocks of a fixed size, each
//! block is computed alone, and block results are combined in block order.

use numpy::ndarray::{Array1, Array2};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::kmeans::{lloyd, pool};
use crate::linalg::cholesky;
use crate::rng::Rng;

/// Rows per block. Fixed, so that results do not depend on the thread count.
const BLOCK: usize = 1024;
const LN_2PI: f64 = 1.837_877_066_409_345_3;
/// Lloyd iterations before EM, by run number (cycled). The first run starts from a converged
/// k-means partition. The others start from k-means++ seeds after fewer iterations, so that runs
/// do not all fall into the basin of one k-means partition, which cuts stretched groups across.
const START_ITERATIONS: [usize; 3] = [30, 0, 3];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Cov {
    Full,
    Diag,
    Spherical,
}

impl Cov {
    fn from_code(code: u8) -> Option<Cov> {
        match code {
            0 => Some(Cov::Full),
            1 => Some(Cov::Diag),
            2 => Some(Cov::Spherical),
            _ => None,
        }
    }

    /// Numbers that describe the covariance of one component.
    fn size(self, dim: usize) -> usize {
        match self {
            Cov::Full => dim * dim,
            Cov::Diag => dim,
            Cov::Spherical => 1,
        }
    }
}

#[derive(Clone)]
struct Params {
    k: usize,
    dim: usize,
    cov: Cov,
    weights: Vec<f64>,
    means: Vec<f64>,
    covs: Vec<f64>,
}

/// What the E-step needs from one covariance: the lower Cholesky factor (full), the standard
/// deviations (diagonal), or the standard deviation (spherical), and the log determinant.
struct Factor {
    scale: Vec<f64>,
    log_det: f64,
}

fn factorize(p: &Params) -> Option<Vec<Factor>> {
    let (d, size) = (p.dim, p.cov.size(p.dim));
    let mut out = Vec::with_capacity(p.k);
    for j in 0..p.k {
        let c = &p.covs[j * size..(j + 1) * size];
        let factor = match p.cov {
            Cov::Full => {
                let l = cholesky(c, d)?;
                let log_det = 2.0 * (0..d).map(|i| l[i * d + i].ln()).sum::<f64>();
                Factor { scale: l, log_det }
            }
            Cov::Diag => {
                if c.iter().any(|&v| !v.is_finite() || v <= 0.0) {
                    return None;
                }
                Factor {
                    scale: c.iter().map(|v| v.sqrt()).collect(),
                    log_det: c.iter().map(|v| v.ln()).sum(),
                }
            }
            Cov::Spherical => {
                if !c[0].is_finite() || c[0] <= 0.0 {
                    return None;
                }
                Factor { scale: vec![c[0].sqrt()], log_det: d as f64 * c[0].ln() }
            }
        };
        out.push(factor);
    }
    Some(out)
}

/// log N(x | mean, covariance) for one row. `z` is scratch of length `dim`.
fn log_density(x: &[f64], mean: &[f64], f: &Factor, cov: Cov, z: &mut [f64]) -> f64 {
    let d = x.len();
    let mut q = 0.0;
    match cov {
        Cov::Full => {
            for i in 0..d {
                let mut s = x[i] - mean[i];
                for m in 0..i {
                    s -= f.scale[i * d + m] * z[m];
                }
                z[i] = s / f.scale[i * d + i];
                q += z[i] * z[i];
            }
        }
        Cov::Diag => {
            for i in 0..d {
                let t = (x[i] - mean[i]) / f.scale[i];
                q += t * t;
            }
        }
        Cov::Spherical => {
            for i in 0..d {
                let t = x[i] - mean[i];
                q += t * t;
            }
            q /= f.scale[0] * f.scale[0];
        }
    }
    -0.5 * (d as f64 * LN_2PI + f.log_det + q)
}

/// Responsibilities of every row and the log density of the mixture at every row.
fn e_step(x: &[f64], n: usize, p: &Params, f: &[Factor]) -> (Vec<f64>, Vec<f64>) {
    let (k, d) = (p.k, p.dim);
    let ln_w: Vec<f64> = p.weights.iter().map(|w| w.ln()).collect();
    let blocks: Vec<(Vec<f64>, Vec<f64>)> = (0..n.div_ceil(BLOCK))
        .into_par_iter()
        .map(|b| {
            let start = b * BLOCK;
            let rows = BLOCK.min(n - start);
            let mut resp = vec![0.0; rows * k];
            let mut lse = vec![0.0; rows];
            let mut z = vec![0.0; d];
            for i in 0..rows {
                let row = &x[(start + i) * d..(start + i + 1) * d];
                let r = &mut resp[i * k..(i + 1) * k];
                let mut top = f64::NEG_INFINITY;
                for j in 0..k {
                    r[j] = ln_w[j] + log_density(row, &p.means[j * d..(j + 1) * d], &f[j], p.cov, &mut z);
                    top = top.max(r[j]);
                }
                let total: f64 = r.iter().map(|v| (v - top).exp()).sum();
                let log_total = top + total.ln();
                for v in r.iter_mut() {
                    *v = (*v - log_total).exp();
                }
                lse[i] = log_total;
            }
            (resp, lse)
        })
        .collect();
    let mut resp = Vec::with_capacity(n * k);
    let mut lse = Vec::with_capacity(n);
    for (r, l) in blocks {
        resp.extend(r);
        lse.extend(l);
    }
    (resp, lse)
}

/// Weights, means and covariances from responsibilities, and the effective size of each
/// component. Sums go block by block and are combined in block order.
fn m_step(
    x: &[f64],
    n: usize,
    dim: usize,
    k: usize,
    cov: Cov,
    resp: &[f64],
    reg: f64,
) -> (Params, Vec<f64>) {
    let first: Vec<(Vec<f64>, Vec<f64>)> = (0..n.div_ceil(BLOCK))
        .into_par_iter()
        .map(|b| {
            let (start, end) = (b * BLOCK, ((b + 1) * BLOCK).min(n));
            let mut nk = vec![0.0; k];
            let mut sx = vec![0.0; k * dim];
            for i in start..end {
                for j in 0..k {
                    let r = resp[i * k + j];
                    nk[j] += r;
                    for d in 0..dim {
                        sx[j * dim + d] += r * x[i * dim + d];
                    }
                }
            }
            (nk, sx)
        })
        .collect();
    let mut nk = vec![0.0; k];
    let mut sx = vec![0.0; k * dim];
    for (a, s) in first {
        for j in 0..k {
            nk[j] += a[j];
        }
        for i in 0..k * dim {
            sx[i] += s[i];
        }
    }
    let mut means = vec![0.0; k * dim];
    for j in 0..k {
        if nk[j] > 0.0 {
            for d in 0..dim {
                means[j * dim + d] = sx[j * dim + d] / nk[j];
            }
        }
    }
    let size = cov.size(dim);
    let second: Vec<Vec<f64>> = (0..n.div_ceil(BLOCK))
        .into_par_iter()
        .map(|b| {
            let (start, end) = (b * BLOCK, ((b + 1) * BLOCK).min(n));
            let mut acc = vec![0.0; k * size];
            let mut diff = vec![0.0; dim];
            for i in start..end {
                for j in 0..k {
                    let r = resp[i * k + j];
                    for d in 0..dim {
                        diff[d] = x[i * dim + d] - means[j * dim + d];
                    }
                    let a = &mut acc[j * size..(j + 1) * size];
                    match cov {
                        Cov::Full => {
                            for u in 0..dim {
                                let ru = r * diff[u];
                                for v in 0..=u {
                                    a[u * dim + v] += ru * diff[v];
                                }
                            }
                        }
                        Cov::Diag => {
                            for d in 0..dim {
                                a[d] += r * diff[d] * diff[d];
                            }
                        }
                        Cov::Spherical => {
                            a[0] += r * diff.iter().map(|t| t * t).sum::<f64>();
                        }
                    }
                }
            }
            acc
        })
        .collect();
    let mut covs = vec![0.0; k * size];
    for part in second {
        for i in 0..k * size {
            covs[i] += part[i];
        }
    }
    for j in 0..k {
        let denom = nk[j].max(f64::MIN_POSITIVE);
        let c = &mut covs[j * size..(j + 1) * size];
        match cov {
            Cov::Full => {
                for u in 0..dim {
                    for v in 0..=u {
                        c[u * dim + v] /= denom;
                        c[v * dim + u] = c[u * dim + v];
                    }
                    c[u * dim + u] += reg;
                }
            }
            Cov::Diag => {
                for d in 0..dim {
                    c[d] = c[d] / denom + reg;
                }
            }
            Cov::Spherical => c[0] = c[0] / (denom * dim as f64) + reg,
        }
    }
    let weights = nk.iter().map(|c| c / n as f64).collect();
    (Params { k, dim, cov, weights, means, covs }, nk)
}

struct Run {
    params: Params,
    resp: Vec<f64>,
    log_lik: f64,
    n_iter: usize,
    converged: bool,
    trace: Vec<f64>,
}

struct Settings {
    k: usize,
    cov: Cov,
    max_iter: usize,
    tol: f64,
    reg: f64,
    min_count: f64,
}

/// One EM run from a k-means start. None when a component becomes degenerate.
fn em_run(
    x: &[f64],
    n: usize,
    dim: usize,
    s: &Settings,
    start: usize,
    rng: &mut Rng,
) -> Option<Run> {
    let k = s.k;
    let (labels, _, _, _, _) = lloyd(x, n, dim, k, start, 0.0, rng);
    let mut resp = vec![0.0; n * k];
    for (i, &l) in labels.iter().enumerate() {
        resp[i * k + l as usize] = 1.0;
    }
    let (mut params, nk) = m_step(x, n, dim, k, s.cov, &resp, s.reg);
    if nk.iter().any(|&c| c < s.min_count) {
        return None;
    }
    let mut trace = Vec::new();
    let mut converged = false;
    let mut previous = f64::NEG_INFINITY;
    let mut used = 0;
    for it in 0..s.max_iter {
        used = it + 1;
        let factors = factorize(&params)?;
        let (r, lse) = e_step(x, n, &params, &factors);
        // A sequential sum in row order, so the value does not depend on the thread count.
        let ll: f64 = lse.iter().sum();
        trace.push(ll);
        resp = r;
        if it > 0 && (ll - previous).abs() <= s.tol * n as f64 {
            converged = true;
            break;
        }
        previous = ll;
        let (next, nk) = m_step(x, n, dim, k, s.cov, &resp, s.reg);
        if nk.iter().any(|&c| c < s.min_count) {
            return None;
        }
        params = next;
    }
    if !converged {
        // The parameters moved after the last E-step: score them once more so that the
        // responsibilities, the parameters and the log-likelihood describe the same fit.
        let factors = factorize(&params)?;
        let (r, lse) = e_step(x, n, &params, &factors);
        resp = r;
        trace.push(lse.iter().sum());
    }
    let log_lik = *trace.last()?;
    if !log_lik.is_finite() {
        return None;
    }
    Some(Run { params, resp, log_lik, n_iter: used, converged, trace })
}

type Output<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    f64,
    usize,
    bool,
    Vec<f64>,
    usize,
);

/// Fit a Gaussian mixture by EM `n_init` times and return the run with the highest
/// log-likelihood: weights, means, covariances (flat), responsibilities, log-likelihood,
/// iterations, whether the log-likelihood settled, the log-likelihood at each iteration, and the
/// number of runs dropped as degenerate.
///
/// `cov` is 0 for full, 1 for diagonal, 2 for spherical. `tol` is a bound on the change of the
/// mean log-likelihood per row. `reg` is an absolute ridge; the caller scales it by the variance
/// of the data.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn gmm_fit<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    k: usize,
    cov: u8,
    max_iter: usize,
    tol: f64,
    reg: f64,
    min_count: f64,
    seed: u64,
    n_init: usize,
    threads: usize,
) -> PyResult<Output<'py>> {
    let shape = data.shape();
    let (n, dim) = (shape[0], shape[1]);
    let cov = Cov::from_code(cov).ok_or_else(|| PyValueError::new_err("cov must be 0, 1 or 2"))?;
    if k == 0 || k > n || dim == 0 || n_init == 0 || max_iter == 0 || reg < 0.0 || tol < 0.0 {
        return Err(PyValueError::new_err("need 1 <= k <= n, dim >= 1, n_init >= 1, max_iter >= 1"));
    }
    let x: Vec<f64> = data.as_array().iter().copied().collect();
    let pool = pool(threads)?;
    let settings = Settings { k, cov, max_iter, tol, reg, min_count };
    let (best, dropped) = py.allow_threads(|| {
        pool.install(|| {
            let mut best: Option<Run> = None;
            let mut dropped = 0usize;
            for i in 0..n_init {
                let mut rng = Rng::new(seed.wrapping_add((i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)));
                match em_run(&x, n, dim, &settings, START_ITERATIONS[i % START_ITERATIONS.len()], &mut rng) {
                    None => dropped += 1,
                    Some(run) => {
                        if best.as_ref().is_none_or(|b| run.log_lik > b.log_lik) {
                            best = Some(run);
                        }
                    }
                }
            }
            (best, dropped)
        })
    });
    let run = best.ok_or_else(|| {
        PyValueError::new_err(
            "every run ended with a component that holds too few points: use fewer components, \
             a simpler covariance, or more data",
        )
    })?;
    let p = run.params;
    let means = Array2::from_shape_vec((k, dim), p.means).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let resp = Array2::from_shape_vec((n, k), run.resp).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        Array1::from_vec(p.weights).into_pyarray_bound(py),
        means.into_pyarray_bound(py),
        Array1::from_vec(p.covs).into_pyarray_bound(py),
        resp.into_pyarray_bound(py),
        run.log_lik,
        run.n_iter,
        run.converged,
        run.trace,
        dropped,
    ))
}

/// Responsibilities and the log density of the mixture for each row of `data`, given weights,
/// means and flat covariances.
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn gmm_score<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    means: PyReadonlyArray2<'py, f64>,
    covs: PyReadonlyArray1<'py, f64>,
    cov: u8,
    threads: usize,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
    let (n, dim) = (data.shape()[0], data.shape()[1]);
    let k = weights.len();
    let cov = Cov::from_code(cov).ok_or_else(|| PyValueError::new_err("cov must be 0, 1 or 2"))?;
    if k == 0 || means.shape() != [k, dim] || covs.len() != k * cov.size(dim) {
        return Err(PyValueError::new_err("weights, means and covariances do not match the data"));
    }
    let params = Params {
        k,
        dim,
        cov,
        weights: weights.as_array().iter().copied().collect(),
        means: means.as_array().iter().copied().collect(),
        covs: covs.as_array().iter().copied().collect(),
    };
    if params.weights.iter().any(|&w| !w.is_finite() || w <= 0.0) {
        return Err(PyValueError::new_err("weights must be positive"));
    }
    let factors = factorize(&params)
        .ok_or_else(|| PyValueError::new_err("a covariance is not positive definite"))?;
    let x: Vec<f64> = data.as_array().iter().copied().collect();
    let pool = pool(threads)?;
    let (resp, lse) = py.allow_threads(|| pool.install(|| e_step(&x, n, &params, &factors)));
    let resp = Array2::from_shape_vec((n, k), resp).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((resp.into_pyarray_bound(py), Array1::from_vec(lse).into_pyarray_bound(py)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blobs(per: usize, seed: u64) -> (Vec<f64>, usize) {
        let mut rng = Rng::new(seed);
        let centres = [[0.0, 0.0], [6.0, 0.0], [0.0, 6.0]];
        let sd = [0.5, 1.0, 0.7];
        let mut x = Vec::new();
        for (c, s) in centres.iter().zip(sd) {
            for _ in 0..per {
                x.push(c[0] + s * rng.normal());
                x.push(c[1] + s * rng.normal());
            }
        }
        (x, 3 * per)
    }

    fn settings(k: usize, cov: Cov) -> Settings {
        Settings { k, cov, max_iter: 500, tol: 1e-10, reg: 1e-9, min_count: 3.0 }
    }

    #[test]
    fn the_log_likelihood_never_decreases_and_the_blobs_are_found() {
        let (x, n) = blobs(200, 1);
        for cov in [Cov::Full, Cov::Diag, Cov::Spherical] {
            let mut rng = Rng::new(4);
            let run = em_run(&x, n, 2, &settings(3, cov), 30, &mut rng).expect("a fit");
            for w in run.trace.windows(2) {
                assert!(w[1] >= w[0] - 1e-7 * w[0].abs(), "{cov:?}: {} -> {}", w[0], w[1]);
            }
            for g in 0..3 {
                let rows: Vec<usize> = (g * 200..g * 200 + 200).collect();
                let first = (0..3).max_by(|&a, &b| run.resp[rows[0] * 3 + a].total_cmp(&run.resp[rows[0] * 3 + b])).unwrap();
                let hits = rows.iter().filter(|&&i| run.resp[i * 3 + first] > 0.5).count();
                assert!(hits > 190, "{cov:?}: group {g} has {hits}");
            }
        }
    }

    #[test]
    fn the_result_does_not_depend_on_the_thread_count() {
        let (x, n) = blobs(1500, 2);
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| em_run(&x, n, 2, &settings(3, Cov::Full), 30, &mut Rng::new(5)).unwrap())
        };
        let (a, b) = (run(1), run(4));
        assert_eq!(a.log_lik, b.log_lik);
        assert_eq!(a.params.means, b.params.means);
        assert_eq!(a.params.covs, b.params.covs);
        assert_eq!(a.resp, b.resp);
    }

    #[test]
    fn responsibilities_sum_to_one_and_weights_to_one() {
        let (x, n) = blobs(100, 3);
        let run = em_run(&x, n, 2, &settings(3, Cov::Full), 30, &mut Rng::new(1)).unwrap();
        for i in 0..n {
            let s: f64 = run.resp[i * 3..i * 3 + 3].iter().sum();
            assert!((s - 1.0).abs() < 1e-12);
        }
        assert!((run.params.weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_component_on_a_few_identical_points_is_dropped() {
        let mut x = vec![0.0; 20];
        x.extend([5.0, 5.0, 9.0, 9.0]);
        let s = Settings { k: 3, cov: Cov::Full, max_iter: 100, tol: 1e-9, reg: 1e-9, min_count: 3.0 };
        let mut rng = Rng::new(2);
        let out = em_run(&x, 12, 2, &s, 30, &mut rng);
        assert!(out.is_none());
    }

    #[test]
    fn one_component_is_the_sample_mean_and_covariance() {
        let (x, n) = blobs(50, 4);
        let run = em_run(&x, n, 2, &settings(1, Cov::Full), 30, &mut Rng::new(1)).unwrap();
        let mean0: f64 = (0..n).map(|i| x[i * 2]).sum::<f64>() / n as f64;
        assert!((run.params.means[0] - mean0).abs() < 1e-12);
        let var0: f64 = (0..n).map(|i| (x[i * 2] - mean0).powi(2)).sum::<f64>() / n as f64;
        assert!((run.params.covs[0] - var0 - 1e-9).abs() < 1e-9);
    }
}
