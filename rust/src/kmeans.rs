//! k-means: k-means++ seeding, Lloyd iterations, and exact nearest-centroid assignment.
//!
//! Lloyd, S. P. (1982), "Least squares quantization in PCM", IEEE Transactions on Information
//! Theory 28(2):129-137, DOI 10.1109/TIT.1982.1056489. Seeding: Arthur, D., & Vassilvitskii, S.
//! (2007), "k-means++: the advantages of careful seeding", SODA '07, pp. 1027-1035 (D^2
//! sampling). Decision D-63.
//!
//! The assignment step is one matrix product per block of rows, `X_b C^T`, since
//! `|x - c|^2 = |x|^2 - 2 x.c + |c|^2` and `|x|^2` does not change the nearest centroid.
//!
//! The result does not depend on the number of threads. Work is split into blocks of a fixed
//! number of rows, each block is computed alone, and block results are combined in block order.
//! Floating point addition is not associative, so a reduction tree that follows the thread count
//! would move the last bits of a centroid, and with them a label.

use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::rng::Rng;

/// Rows per block. Fixed, so that results do not depend on the thread count.
const BLOCK: usize = 2048;

pub(crate) type Fit = (Vec<i64>, Vec<f64>, f64, usize, bool);

fn dist2(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for d in 0..a.len() {
        let diff = a[d] - b[d];
        s += diff * diff;
    }
    s
}

/// Nearest centroid of each row of the block `x` (rows x dim), by one matrix product.
///
/// `x` and `centroids` must be shifted to near the origin: the expansion
/// `|x - c|^2 = |x|^2 - 2 x.c + |c|^2` loses the difference between two centroids when the
/// values are large against their distance.
fn assign_block(
    x: &[f64],
    rows: usize,
    dim: usize,
    centroids: &[f64],
    c_sq: &[f64],
    k: usize,
    scratch: &mut [f64],
) -> Vec<usize> {
    // SAFETY: x is rows x dim and centroids is k x dim, both row-major and contiguous; scratch
    // holds at least rows * k values.
    unsafe {
        matrixmultiply::dgemm(
            rows,
            dim,
            k,
            1.0,
            x.as_ptr(),
            dim as isize,
            1,
            centroids.as_ptr(),
            1,
            dim as isize,
            0.0,
            scratch.as_mut_ptr(),
            k as isize,
            1,
        );
    }
    (0..rows)
        .map(|i| {
            let g = &scratch[i * k..i * k + k];
            let mut best = 0usize;
            let mut best_v = f64::INFINITY;
            for j in 0..k {
                let v = c_sq[j] - 2.0 * g[j];
                if v < best_v {
                    best_v = v;
                    best = j;
                }
            }
            best
        })
        .collect()
}

fn assign_all(x: &[f64], n: usize, dim: usize, centroids: &[f64], k: usize) -> (Vec<i64>, Vec<f64>) {
    // Both sides are moved by the mean of the centroids before the product. A move changes no
    // distance, and it keeps the digits that tell two centroids apart when the data are far
    // from the origin.
    let mut shift = vec![0.0; dim];
    for c in centroids.chunks(dim) {
        for d in 0..dim {
            shift[d] += c[d] / k as f64;
        }
    }
    let moved: Vec<f64> = centroids.iter().enumerate().map(|(i, v)| v - shift[i % dim]).collect();
    let c_sq: Vec<f64> = moved.chunks(dim).map(|c| c.iter().map(|v| v * v).sum()).collect();
    let blocks: Vec<Vec<(i64, f64)>> = (0..n.div_ceil(BLOCK))
        .into_par_iter()
        .map(|b| {
            let start = b * BLOCK;
            let rows = BLOCK.min(n - start);
            let raw = &x[start * dim..(start + rows) * dim];
            let near: Vec<f64> = raw.iter().enumerate().map(|(i, v)| v - shift[i % dim]).collect();
            let mut scratch = vec![0.0; rows * k];
            assign_block(&near, rows, dim, &moved, &c_sq, k, &mut scratch)
                .into_iter()
                .enumerate()
                .map(|(i, best)| {
                    // The distance by differences of the values as given: no cancellation in
                    // the reported inertia.
                    let row = &raw[i * dim..(i + 1) * dim];
                    (best as i64, dist2(row, &centroids[best * dim..(best + 1) * dim]))
                })
                .collect()
        })
        .collect();
    let mut labels = Vec::with_capacity(n);
    let mut dist = Vec::with_capacity(n);
    for block in blocks {
        for (l, d) in block {
            labels.push(l);
            dist.push(d);
        }
    }
    (labels, dist)
}

/// k-means++: the first centre uniformly, each next one with probability proportional to the
/// squared distance to the nearest centre chosen so far.
fn seed_centres(x: &[f64], n: usize, dim: usize, k: usize, rng: &mut Rng) -> Vec<f64> {
    let mut centres = vec![0.0; k * dim];
    let first = rng.below(n as u64) as usize;
    centres[..dim].copy_from_slice(&x[first * dim..first * dim + dim]);
    let mut closest: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| dist2(&x[i * dim..i * dim + dim], &centres[..dim]))
        .collect();
    for c in 1..k {
        // A sequential sum in a fixed order: it picks a discrete index, so it must not depend on
        // the thread count.
        let total: f64 = closest.iter().sum();
        let chosen = if total > 0.0 {
            let target = rng.uniform() * total;
            let mut acc = 0.0;
            let mut pick = n - 1;
            for (i, &d) in closest.iter().enumerate() {
                acc += d;
                if acc > target {
                    pick = i;
                    break;
                }
            }
            pick
        } else {
            // Every point sits on a centre already: any distinct draw is as good as another.
            rng.below(n as u64) as usize
        };
        centres[c * dim..c * dim + dim].copy_from_slice(&x[chosen * dim..chosen * dim + dim]);
        let fresh = centres[c * dim..c * dim + dim].to_vec();
        closest
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v = v.min(dist2(&x[i * dim..i * dim + dim], &fresh)));
    }
    centres
}

/// Cluster means, summed block by block and combined in block order.
fn means(x: &[f64], n: usize, dim: usize, labels: &[i64], k: usize) -> (Vec<f64>, Vec<usize>) {
    let parts: Vec<(Vec<f64>, Vec<usize>)> = (0..n.div_ceil(BLOCK))
        .into_par_iter()
        .map(|b| {
            let start = b * BLOCK;
            let end = (start + BLOCK).min(n);
            let mut sums = vec![0.0; k * dim];
            let mut counts = vec![0usize; k];
            for i in start..end {
                let c = labels[i] as usize;
                counts[c] += 1;
                for d in 0..dim {
                    sums[c * dim + d] += x[i * dim + d];
                }
            }
            (sums, counts)
        })
        .collect();
    let mut sums = vec![0.0; k * dim];
    let mut counts = vec![0usize; k];
    for (s, c) in parts {
        for i in 0..k * dim {
            sums[i] += s[i];
        }
        for i in 0..k {
            counts[i] += c[i];
        }
    }
    (sums, counts)
}

/// Scale a centre to length 1. A centre of length 0 (its points cancel out) has no direction and
/// is left as `fallback`.
fn to_unit(centre: &mut [f64], fallback: &[f64]) {
    let norm = centre.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm > 0.0 {
        for v in centre.iter_mut() {
            *v /= norm;
        }
    } else {
        centre.copy_from_slice(fallback);
    }
}

/// With `spherical`, the rows of `x` must have length 1 and each centre is scaled to length 1
/// after it is averaged: spherical k-means (Dhillon & Modha 2001, "Concept decompositions for
/// large sparse text data using clustering", Machine Learning 42:143-175), which maximises the
/// sum of cosines. For unit vectors `|x - c|^2 = 2 - 2 x.c`, so the nearest centre is the one
/// with the largest cosine and the assignment step needs no change. Decision D-71.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lloyd(
    x: &[f64],
    n: usize,
    dim: usize,
    k: usize,
    max_iter: usize,
    tol: f64,
    spherical: bool,
    rng: &mut Rng,
) -> Fit {
    let mut centres = seed_centres(x, n, dim, k, rng);
    let mut labels = vec![-1i64; n];
    let mut dist = vec![0.0; n];
    let mut converged = false;
    let mut used = 0;
    for it in 0..max_iter {
        used = it + 1;
        let (next, d) = assign_all(x, n, dim, &centres, k);
        dist = d;
        let same = next == labels;
        labels = next;
        let (sums, counts) = means(x, n, dim, &labels, k);
        let mut shift = 0.0;
        let mut empty: Vec<usize> = Vec::new();
        for c in 0..k {
            if counts[c] == 0 {
                empty.push(c);
                continue;
            }
            let inv = 1.0 / counts[c] as f64;
            let mut mean: Vec<f64> = (0..dim).map(|d| sums[c * dim + d] * inv).collect();
            if spherical {
                to_unit(&mut mean, &centres[c * dim..(c + 1) * dim]);
            }
            for d in 0..dim {
                let diff = mean[d] - centres[c * dim + d];
                shift += diff * diff;
                centres[c * dim + d] = mean[d];
            }
        }
        if !empty.is_empty() {
            // An empty cluster moves onto the points worst served by their centroid, so the
            // objective cannot rise; ties go to the lower index.
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| dist[b].total_cmp(&dist[a]).then(a.cmp(&b)));
            for (slot, &c) in empty.iter().enumerate() {
                let i = order[slot.min(n - 1)];
                centres[c * dim..c * dim + dim].copy_from_slice(&x[i * dim..i * dim + dim]);
            }
            continue;
        }
        if same || shift <= tol {
            converged = true;
            break;
        }
    }
    // A last assignment so that labels, centres and inertia describe the same fit.
    let (labels_final, dist_final) = assign_all(x, n, dim, &centres, k);
    let inertia: f64 = dist_final.iter().sum();
    let (sums, counts) = means(x, n, dim, &labels_final, k);
    let mut out = centres;
    for c in 0..k {
        if counts[c] > 0 {
            let mut mean: Vec<f64> = (0..dim).map(|d| sums[c * dim + d] / counts[c] as f64).collect();
            if spherical {
                to_unit(&mut mean, &out[c * dim..(c + 1) * dim]);
            }
            out[c * dim..(c + 1) * dim].copy_from_slice(&mean);
        }
    }
    let _ = labels;
    (labels_final, out, inertia, used, converged)
}

pub(crate) fn pool(threads: usize) -> PyResult<rayon::ThreadPool> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .build()
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Fit k-means `n_init` times and return the labels, centres, inertia, iterations and whether
/// Lloyd's iteration stopped before `max_iter`, for the run with the lowest inertia.
///
/// `tol` is an absolute bound on the sum of squared centre moves; the caller scales it by the
/// variance of the data.
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn kmeans_fit<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    k: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
    n_init: usize,
    spherical: bool,
    threads: usize,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray2<f64>>, f64, usize, bool)> {
    let shape = data.shape();
    let (n, dim) = (shape[0], shape[1]);
    if k == 0 || k > n || dim == 0 || n_init == 0 || max_iter == 0 {
        return Err(PyValueError::new_err("need 1 <= k <= n, dim >= 1, n_init >= 1, max_iter >= 1"));
    }
    let x: Vec<f64> = data.as_array().iter().copied().collect();
    let pool = pool(threads)?;
    let best: Fit = py.allow_threads(|| {
        pool.install(|| {
            let mut best: Option<Fit> = None;
            for i in 0..n_init {
                let mut rng = Rng::new(seed.wrapping_add((i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)));
                let fit = lloyd(&x, n, dim, k, max_iter, tol, spherical, &mut rng);
                if best.as_ref().is_none_or(|b| fit.2 < b.2) {
                    best = Some(fit);
                }
            }
            best.expect("n_init >= 1")
        })
    });
    let (labels, centres, inertia, used, converged) = best;
    let centres = Array2::from_shape_vec((k, dim), centres)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        Array1::from_vec(labels).into_pyarray_bound(py),
        centres.into_pyarray_bound(py),
        inertia,
        used,
        converged,
    ))
}

/// The nearest centroid of each row, and the squared distance to it.
#[pyfunction]
pub fn kmeans_assign<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    centroids: PyReadonlyArray2<'py, f64>,
    threads: usize,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)> {
    let (n, dim) = (data.shape()[0], data.shape()[1]);
    let (k, cdim) = (centroids.shape()[0], centroids.shape()[1]);
    if k == 0 || cdim != dim {
        return Err(PyValueError::new_err("centroids need at least one row and the data's columns"));
    }
    let x: Vec<f64> = data.as_array().iter().copied().collect();
    let c: Vec<f64> = centroids.as_array().iter().copied().collect();
    let pool = pool(threads)?;
    let (labels, dist) = py.allow_threads(|| pool.install(|| assign_all(&x, n, dim, &c, k)));
    Ok((
        Array1::from_vec(labels).into_pyarray_bound(py),
        Array1::from_vec(dist).into_pyarray_bound(py),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blobs(per: usize, seed: u64) -> (Vec<f64>, usize) {
        let mut rng = Rng::new(seed);
        let centres = [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]];
        let mut x = Vec::new();
        for c in centres {
            for _ in 0..per {
                x.push(c[0] + 0.3 * rng.normal());
                x.push(c[1] + 0.3 * rng.normal());
            }
        }
        (x, 3 * per)
    }

    #[test]
    fn three_blobs_are_found_and_inertia_matches_the_labels() {
        let (x, n) = blobs(50, 1);
        let mut rng = Rng::new(3);
        let (labels, centres, inertia, _, converged) = lloyd(&x, n, 2, 3, 100, 1e-12, false, &mut rng);
        assert!(converged);
        for g in 0..3 {
            let first = labels[g * 50];
            assert!(labels[g * 50..g * 50 + 50].iter().all(|&l| l == first));
        }
        let direct: f64 = (0..n)
            .map(|i| dist2(&x[i * 2..i * 2 + 2], &centres[labels[i] as usize * 2..labels[i] as usize * 2 + 2]))
            .sum();
        assert!((direct - inertia).abs() < 1e-9 * inertia.max(1.0));
    }

    #[test]
    fn the_result_does_not_depend_on_the_thread_count() {
        let (x, n) = blobs(2500, 2);
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| {
                let mut rng = Rng::new(5);
                lloyd(&x, n, 2, 3, 100, 1e-12, false, &mut rng)
            })
        };
        let (a, b) = (run(1), run(4));
        assert_eq!(a.0, b.0);
        assert_eq!(a.1, b.1);
        assert_eq!(a.2, b.2);
    }

    #[test]
    fn k_equal_to_n_gives_zero_inertia() {
        let x = vec![0.0, 1.0, 5.0, 9.0, 9.5, 20.0];
        let mut rng = Rng::new(1);
        let (_, _, inertia, _, _) = lloyd(&x, 6, 1, 6, 50, 0.0, false, &mut rng);
        assert!(inertia < 1e-12);
    }

    #[test]
    fn spherical_centres_have_length_one_and_a_cancelled_centre_keeps_its_place() {
        // Four unit vectors around two directions.
        let x = [1.0, 0.0, 0.96, 0.28, 0.0, 1.0, 0.28, 0.96];
        let mut rng = Rng::new(1);
        let (_, centres, _, _, _) = lloyd(&x, 4, 2, 2, 100, 0.0, true, &mut rng);
        for c in centres.chunks(2) {
            assert!(((c[0] * c[0] + c[1] * c[1]).sqrt() - 1.0).abs() < 1e-15);
        }
        let mut zero = [0.0, 0.0];
        to_unit(&mut zero, &[0.0, 1.0]);
        assert_eq!(zero, [0.0, 1.0]);
    }

    #[test]
    fn a_point_far_from_the_origin_still_goes_to_the_nearest_centroid() {
        let offset = 1e9;
        let centres = [offset, offset, offset + 2.0, offset];
        let x = [offset + 0.4, offset, offset + 1.6, offset + 0.1, offset + 0.99, offset];
        let (labels, dist) = assign_all(&x, 3, 2, &centres, 2);
        assert_eq!(labels, vec![0, 1, 0]);
        assert!((dist[0] - 0.16).abs() < 1e-6);
    }
}
