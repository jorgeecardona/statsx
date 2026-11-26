//! UMAP: the graph, the spectral start and the layout, from the paper.
//!
//! McInnes, L., Healy, J., & Melville, J. (2018), "UMAP: Uniform Manifold Approximation and
//! Projection for Dimension Reduction", arXiv:1802.03426 (v3, 2020). Section 3.1 gives the graph:
//! for each point `x_i` with neighbours `x_i1 .. x_ik`, `rho_i` is the smallest positive neighbour
//! distance, `sigma_i` solves `sum_j exp(-max(0, d_ij - rho_i) / sigma_i) = log2(k)`, the weight of
//! the directed edge is `exp(-max(0, d_ij - rho_i) / sigma_i)`, and the graph is the symmetric
//! matrix `B = A + A^T - A o A^T` (probabilistic t-conorm). Algorithm 4 gives the start: the
//! eigenvectors of the symmetric normalised Laplacian `I - D^(-1/2) A D^(-1/2)`, skipping the
//! first. Section 4.1 and Algorithm 5 give the layout: sample an edge with probability equal to
//! its weight, move the point along the gradient of `log Phi`, then along the gradient of
//! `log(1 - Phi)` for `n_neg` points drawn at random, with `Phi(r) = 1 / (1 + a r^(2b))` and a
//! learning rate that falls as `1 - e / n_epochs`. Section 3.2 gives the small constant
//! `eps = 0.001` of the repulsive force. Decision D-66.
//!
//! The eigenvectors are found by the locally optimal block preconditioned conjugate gradient
//! method (LOBPCG), Knyazev, A. V. (2001), "Toward the optimal preconditioned eigensolver:
//! locally optimal block preconditioned conjugate gradient method", SIAM J. Sci. Comput.
//! 23(2):517-541, without a preconditioner. The basis is made orthonormal at every iteration,
//! because the plain method loses accuracy when its basis becomes nearly dependent (Duersch,
//! Shao, Yang & Gu (2018), SIAM J. Sci. Comput. 40(5):C655-C676, discuss this). The small dense
//! eigenproblems use Jacobi's method (Jacobi 1846), sweeping all pairs in a fixed order.
//!
//! New points are placed in a finished layout by the same Algorithm 5 with the training points held
//! fixed (`place`), with local connectivity 0 as the uwot vignette and umap-learn do. No paper
//! describes this (Reichmann et al. 2024 say so); decision D-68.
//!
//! The neighbours are exact (blocked matrix products, as in `density`), not the approximate
//! nearest-neighbour descent of Dong, Charikar & Li (2011) that the paper prefers.
//!
//! Nothing here depends on the number of threads. Neighbour blocks have a fixed size. In the
//! layout, every point is updated from the positions at the start of the epoch and from its own
//! random stream, which is seeded by (seed, epoch, point), so no two threads write the same value.

use numpy::ndarray::{Array1, Array2};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::density::{blocks, Points, COLS, ROWS};
use crate::kmeans::pool;
use crate::rng::Rng;

/// Each coordinate of one gradient step is limited to this, as in the reference implementation of
/// the paper. The paper does not state a limit; without one, a pair of points that nearly coincide
/// is thrown far away by the repulsive force, which grows as `1 / (eps + r^2)`.
const CLIP: f64 = 4.0;
/// The `epsilon` of the repulsive force in section 3.2.
const EPS: f64 = 0.001;
/// Rows per chunk in the layout and in sparse products. Fixed.
const CHUNK: usize = 256;
/// Bisection steps for `sigma`. Fixed.
const BISECT: usize = 100;

// ---------------------------------------------------------------------------------------------
// Nearest neighbours

/// The `k` nearest other points of each point: indices and distances, nearest first, ties by index.
fn knn(p: &Points, k: usize) -> (Vec<usize>, Vec<f64>) {
    search(p, 0, p.n, p.n, k)
}

/// The `k` nearest of the rows `0..refs` for each of the rows `q0..q1`, nearest first, ties by
/// index. A row never counts as its own neighbour. When the queries come after the references
/// (`q0 >= refs`) no row is one of its own candidates, so nothing is skipped.
fn search(p: &Points, q0: usize, q1: usize, refs: usize, k: usize) -> (Vec<usize>, Vec<f64>) {
    let parts: Vec<(Vec<usize>, Vec<f64>)> = (0..blocks(q1 - q0))
        .into_par_iter()
        .map(|b| {
            let r0 = q0 + b * ROWS;
            let rows = ROWS.min(q1 - r0);
            let mut best: Vec<Vec<(f64, usize)>> = vec![Vec::with_capacity(k + 1); rows];
            let mut scratch = vec![0.0; rows * COLS];
            let mut c0 = 0;
            while c0 < refs {
                let cols = COLS.min(refs - c0);
                p.gram(r0, rows, c0, cols, &mut scratch);
                for r in 0..rows {
                    let list = &mut best[r];
                    for c in 0..cols {
                        if r0 + r == c0 + c {
                            continue;
                        }
                        let d2 = p.d2(r0 + r, c0 + c, scratch[r * cols + c]);
                        if list.len() < k || d2 < list[list.len() - 1].0 {
                            let at = list.partition_point(|e| e.0 <= d2);
                            list.insert(at, (d2, c0 + c));
                            list.truncate(k);
                        }
                    }
                }
                c0 += cols;
            }
            let mut idx = Vec::with_capacity(rows * k);
            let mut dist = Vec::with_capacity(rows * k);
            for (r, list) in best.iter().enumerate() {
                // Exact distances for the k kept, so the answer does not carry the rounding of
                // the expansion, and so equal distances are ordered by index.
                let mut exact: Vec<(f64, usize)> =
                    list.iter().map(|&(_, j)| (p.exact_d2(r0 + r, j), j)).collect();
                exact.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                for (d2, j) in exact {
                    idx.push(j);
                    dist.push(d2.sqrt());
                }
            }
            (idx, dist)
        })
        .collect();
    let mut idx = Vec::with_capacity((q1 - q0) * k);
    let mut dist = Vec::with_capacity((q1 - q0) * k);
    for (i, d) in parts {
        idx.extend(i);
        dist.extend(d);
    }
    (idx, dist)
}

// ---------------------------------------------------------------------------------------------
// The weighted graph

/// `rho` and `sigma` of one point from its neighbour distances (section 3.1).
fn smooth_knn(dists: &[f64]) -> (f64, f64) {
    let rho = dists.iter().copied().filter(|&d| d > 0.0).fold(f64::INFINITY, f64::min);
    let rho = if rho.is_finite() { rho } else { 0.0 };
    (rho, solve_sigma(dists, rho))
}

/// The `sigma` that solves `sum_j exp(-max(0, d_j - rho) / sigma) = log2(k)` for a given `rho`.
fn solve_sigma(dists: &[f64], rho: f64) -> f64 {
    let k = dists.len();
    let target = (k as f64).log2();
    let total = |sigma: f64| -> f64 { dists.iter().map(|&d| (-(d - rho).max(0.0) / sigma).exp()).sum() };
    // The sum grows with sigma, from the number of neighbours at distance rho (or less) up to k.
    let (mut lo, mut hi) = (0.0, 1.0);
    let mut guard = 0;
    while total(hi) < target && guard < 2000 {
        lo = hi;
        hi *= 2.0;
        guard += 1;
    }
    for _ in 0..BISECT {
        let mid = 0.5 * (lo + hi);
        if total(mid) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

type Csr = (Vec<usize>, Vec<usize>, Vec<f64>);

/// The symmetric graph `B = A + A^T - A o A^T` in compressed rows, from the directed weights.
fn fuzzy_union(n: usize, k: usize, idx: &[usize], w: &[f64]) -> Csr {
    // One record per directed edge, keyed by the unordered pair, so the two directions meet.
    let mut edges: Vec<(usize, usize, f64)> = Vec::with_capacity(n * k);
    for i in 0..n {
        for e in 0..k {
            let (j, v) = (idx[i * k + e], w[i * k + e]);
            if v > 0.0 && j != i {
                edges.push((i.min(j), i.max(j), v));
            }
        }
    }
    edges.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut pairs: Vec<(usize, usize, f64)> = Vec::with_capacity(edges.len());
    for (lo, hi, v) in edges {
        match pairs.last_mut() {
            Some(last) if last.0 == lo && last.1 == hi => last.2 = last.2 + v - last.2 * v,
            _ => pairs.push((lo, hi, v)),
        }
    }
    let mut both: Vec<(usize, usize, f64)> = Vec::with_capacity(2 * pairs.len());
    for &(lo, hi, v) in &pairs {
        both.push((lo, hi, v));
        both.push((hi, lo, v));
    }
    both.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut indptr = vec![0usize; n + 1];
    for &(r, _, _) in &both {
        indptr[r + 1] += 1;
    }
    for i in 0..n {
        indptr[i + 1] += indptr[i];
    }
    let indices = both.iter().map(|e| e.1).collect();
    let data = both.iter().map(|e| e.2).collect();
    (indptr, indices, data)
}

/// Connected components, numbered by the lowest index they hold.
fn components(indptr: &[usize], indices: &[usize]) -> Vec<i64> {
    let n = indptr.len() - 1;
    let mut label = vec![-1i64; n];
    let mut next = 0;
    let mut stack = Vec::new();
    for s in 0..n {
        if label[s] >= 0 {
            continue;
        }
        label[s] = next;
        stack.push(s);
        while let Some(u) = stack.pop() {
            for &v in &indices[indptr[u]..indptr[u + 1]] {
                if label[v] < 0 {
                    label[v] = next;
                    stack.push(v);
                }
            }
        }
        next += 1;
    }
    label
}

// ---------------------------------------------------------------------------------------------
// Spectral start

/// Eigenvalues (any order) and eigenvectors (column `j` of the row-major result) of a symmetric
/// `m x m` matrix, by the cyclic Jacobi method. `a` is destroyed.
fn jacobi(a: &mut [f64], m: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0.0; m * m];
    for i in 0..m {
        v[i * m + i] = 1.0;
    }
    for _ in 0..100 {
        let mut off = 0.0;
        let mut diag = 0.0;
        for i in 0..m {
            diag += a[i * m + i] * a[i * m + i];
            for j in 0..i {
                off += a[i * m + j] * a[i * m + j];
            }
        }
        if off <= 1e-30 * (diag + off) || off == 0.0 {
            break;
        }
        for p in 0..m {
            for q in p + 1..m {
                let apq = a[p * m + q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (a[q * m + q] - a[p * m + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for r in 0..m {
                    let (arp, arq) = (a[r * m + p], a[r * m + q]);
                    a[r * m + p] = c * arp - s * arq;
                    a[r * m + q] = s * arp + c * arq;
                }
                for r in 0..m {
                    let (apr, aqr) = (a[p * m + r], a[q * m + r]);
                    a[p * m + r] = c * apr - s * aqr;
                    a[q * m + r] = s * apr + c * aqr;
                }
                for r in 0..m {
                    let (vrp, vrq) = (v[r * m + p], v[r * m + q]);
                    v[r * m + p] = c * vrp - s * vrq;
                    v[r * m + q] = s * vrp + c * vrq;
                }
            }
        }
    }
    ((0..m).map(|i| a[i * m + i]).collect(), v)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `out = D^(-1/2) A D^(-1/2) x`. Each output value is a sum in a fixed order.
fn normalised_product(g: &Csr, inv_sqrt: &[f64], x: &[f64]) -> Vec<f64> {
    let (indptr, indices, data) = g;
    let mut out = vec![0.0; x.len()];
    out.par_chunks_mut(CHUNK).enumerate().for_each(|(c, chunk)| {
        for (r, o) in chunk.iter_mut().enumerate() {
            let i = c * CHUNK + r;
            let mut s = 0.0;
            for e in indptr[i]..indptr[i + 1] {
                let j = indices[e];
                s += data[e] * inv_sqrt[j] * x[j];
            }
            *o = s * inv_sqrt[i];
        }
    });
    out
}

/// Make the columns orthonormal to `fixed` and to each other, in order, dropping a column that is
/// (numerically) in the span of the earlier ones.
fn orthonormalise(cols: Vec<Vec<f64>>, fixed: &[f64]) -> Vec<Vec<f64>> {
    let mut out: Vec<Vec<f64>> = Vec::with_capacity(cols.len());
    for mut v in cols {
        let before = dot(&v, &v).sqrt();
        if before == 0.0 {
            continue;
        }
        for _ in 0..2 {
            let c = dot(fixed, &v);
            v.iter_mut().zip(fixed).for_each(|(x, u)| *x -= c * u);
            for u in &out {
                let c = dot(u, &v);
                v.iter_mut().zip(u).for_each(|(x, y)| *x -= c * y);
            }
        }
        let after = dot(&v, &v).sqrt();
        if after <= 1e-10 * before {
            continue;
        }
        v.iter_mut().for_each(|x| *x /= after);
        out.push(v);
    }
    out
}

struct Spectral {
    vectors: Vec<Vec<f64>>,
    values: Vec<f64>,
    converged: bool,
    iterations: usize,
}

/// The `b` largest eigenvectors of `S = D^(-1/2) A D^(-1/2)` apart from the trivial one
/// `D^(1/2) 1`, which is the second to the `(b + 1)`-th smallest of the Laplacian `I - S`. The
/// graph must be connected, so that the trivial eigenvalue is simple.
fn lobpcg(g: &Csr, b: usize, max_iter: usize, tol: f64, seed: u64) -> Option<Spectral> {
    let n = g.0.len() - 1;
    let degree: Vec<f64> = (0..n).map(|i| g.2[g.0[i]..g.0[i + 1]].iter().sum()).collect();
    let inv_sqrt: Vec<f64> = degree.iter().map(|d| 1.0 / d.sqrt()).collect();
    let total: f64 = degree.iter().sum();
    let trivial: Vec<f64> = degree.iter().map(|d| (d / total).sqrt()).collect();
    let mut rng = Rng::new(seed);
    let start: Vec<Vec<f64>> = (0..b).map(|_| (0..n).map(|_| rng.normal()).collect()).collect();
    let mut x = orthonormalise(start, &trivial);
    if x.len() < b {
        return None;
    }
    let mut p: Vec<Vec<f64>> = Vec::new();
    let mut converged = false;
    let mut iterations = 0;
    let mut values = vec![0.0; b];
    for it in 1..=max_iter {
        iterations = it;
        let ax: Vec<Vec<f64>> = x.iter().map(|c| normalised_product(g, &inv_sqrt, c)).collect();
        let mut residuals = Vec::with_capacity(b);
        let mut worst = 0.0f64;
        for j in 0..b {
            values[j] = dot(&x[j], &ax[j]);
            let r: Vec<f64> = ax[j].iter().zip(&x[j]).map(|(a, c)| a - values[j] * c).collect();
            worst = worst.max(dot(&r, &r).sqrt());
            residuals.push(r);
        }
        if worst <= tol {
            converged = true;
            break;
        }
        let mut cols = x.clone();
        cols.extend(residuals);
        cols.extend(p.iter().cloned());
        let z = orthonormalise(cols, &trivial);
        if z.len() < b {
            return None;
        }
        let m = z.len();
        let sz: Vec<Vec<f64>> = z.iter().map(|c| normalised_product(g, &inv_sqrt, c)).collect();
        let mut gram = vec![0.0; m * m];
        for i in 0..m {
            for j in 0..=i {
                let v = 0.5 * (dot(&z[i], &sz[j]) + dot(&z[j], &sz[i]));
                gram[i * m + j] = v;
                gram[j * m + i] = v;
            }
        }
        let (vals, vecs) = jacobi(&mut gram, m);
        let mut order: Vec<usize> = (0..m).collect();
        order.sort_by(|&a, &c| vals[c].total_cmp(&vals[a]));
        let mut next_x = Vec::with_capacity(b);
        let mut next_p = Vec::with_capacity(b);
        for &col in order.iter().take(b) {
            let mut xv = vec![0.0; n];
            let mut pv = vec![0.0; n];
            for (i, zi) in z.iter().enumerate() {
                let c = vecs[i * m + col];
                xv.iter_mut().zip(zi).for_each(|(o, v)| *o += c * v);
                if i >= b {
                    pv.iter_mut().zip(zi).for_each(|(o, v)| *o += c * v);
                }
            }
            next_x.push(xv);
            next_p.push(pv);
        }
        x = orthonormalise(next_x, &trivial);
        if x.len() < b {
            return None;
        }
        p = next_p;
    }
    // The values belong to the vectors that were tested, so recompute them for the ones returned.
    for j in 0..b {
        values[j] = dot(&x[j], &normalised_product(g, &inv_sqrt, &x[j]));
    }
    Some(Spectral { vectors: x, values, converged, iterations })
}

// ---------------------------------------------------------------------------------------------
// Layout

fn clip(v: f64) -> f64 {
    v.clamp(-CLIP, CLIP)
}

/// The gradient of `log Phi` with respect to `y_i` is this times `y_i - y_j`, where `r2` is the
/// squared distance: `-2 a b r^(2(b-1)) / (1 + a r^(2b))`.
fn attraction(r2: f64, a: f64, b: f64) -> f64 {
    -2.0 * a * b * r2.powf(b - 1.0) / (1.0 + a * r2.powf(b))
}

/// The gradient of `log(1 - Phi)` with respect to `y_i` is this times `y_i - y_j`:
/// `2 b / ((eps + r^2)(1 + a r^(2b)))`, the form of section 3.2.
fn repulsion(r2: f64, a: f64, b: f64) -> f64 {
    2.0 * b / ((EPS + r2) * (1.0 + a * r2.powf(b)))
}

/// Squared distance of two rows of `y`.
fn dist2(y: &[f64], dim: usize, i: usize, j: usize) -> f64 {
    let mut s = 0.0;
    for c in 0..dim {
        let t = y[i * dim + c] - y[j * dim + c];
        s += t * t;
    }
    s
}

/// Moves one point by the edges that start at it and by its negative samples (Algorithm 5).
///
/// `heads` are the other ends of its edges and `weights` the probabilities of sampling them. The
/// other ends and the negative samples are rows of `others`, read as they are. `own` is the row
/// of `others` that is this point itself, which a negative sample skips.
#[allow(clippy::too_many_arguments)]
fn move_point(
    pos: &mut [f64],
    rng: &mut Rng,
    others: &[f64],
    dim: usize,
    heads: &[usize],
    weights: &[f64],
    own: Option<usize>,
    ab: (f64, f64),
    alpha: f64,
    n_neg: usize,
) {
    let n_others = others.len() / dim;
    let (a, b) = ab;
    for (e, &j) in heads.iter().enumerate() {
        if rng.uniform() >= weights[e] {
            continue;
        }
        // Attraction: the gradient of log Phi, -2ab r^(2(b-1)) / (1 + a r^(2b)) (y_i - y_j).
        let mut r2 = 0.0;
        for c in 0..dim {
            let t = pos[c] - others[j * dim + c];
            r2 += t * t;
        }
        if r2 > 0.0 {
            let coef = attraction(r2, a, b);
            for c in 0..dim {
                pos[c] += alpha * clip(coef * (pos[c] - others[j * dim + c]));
            }
        }
        // Repulsion: the gradient of log(1 - Phi), 2b / ((eps + r^2)(1 + a r^(2b))) (y_i - y_k).
        for _ in 0..n_neg {
            let kk = rng.below(n_others as u64) as usize;
            if Some(kk) == own {
                continue;
            }
            let mut r2 = 0.0;
            for c in 0..dim {
                let t = pos[c] - others[kk * dim + c];
                r2 += t * t;
            }
            if r2 > 0.0 {
                let coef = repulsion(r2, a, b);
                for c in 0..dim {
                    pos[c] += alpha * clip(coef * (pos[c] - others[kk * dim + c]));
                }
            } else {
                // Coinciding points have no direction to move in; push all the same way.
                for c in 0..dim {
                    pos[c] += alpha * CLIP;
                }
            }
        }
    }
}

/// The seed of the random stream of one point in one epoch.
fn point_stream(stream: u64, epoch_index: u64, i: usize) -> u64 {
    stream
        ^ epoch_index.wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (i as u64).wrapping_mul(0xc2b2_ae3d_27d4_eb4f)
}

/// One epoch of Algorithm 5. Every point is moved by the edges that start at it and by its own
/// negative samples, the other end being read at its position at the start of the epoch.
#[allow(clippy::too_many_arguments)]
fn epoch(
    cur: &[f64],
    next: &mut [f64],
    dim: usize,
    g: &Csr,
    ab: (f64, f64),
    alpha: f64,
    n_neg: usize,
    stream: u64,
    epoch_index: u64,
) {
    let (indptr, indices, weights) = g;
    next.par_chunks_mut(CHUNK * dim).enumerate().for_each(|(c, chunk)| {
        for (r, out) in chunk.chunks_mut(dim).enumerate() {
            let i = c * CHUNK + r;
            let mut rng = Rng::new(point_stream(stream, epoch_index, i));
            let mut pos: Vec<f64> = cur[i * dim..(i + 1) * dim].to_vec();
            let (lo, hi) = (indptr[i], indptr[i + 1]);
            move_point(
                &mut pos,
                &mut rng,
                cur,
                dim,
                &indices[lo..hi],
                &weights[lo..hi],
                Some(i),
                ab,
                alpha,
                n_neg,
            );
            out.copy_from_slice(&pos);
        }
    });
}

#[allow(clippy::too_many_arguments)]
fn layout(
    start: &[f64],
    dim: usize,
    g: &Csr,
    ab: (f64, f64),
    n_epochs: usize,
    n_neg: usize,
    learning_rate: f64,
    seed: u64,
) -> Vec<f64> {
    let mut cur = start.to_vec();
    let mut next = vec![0.0; cur.len()];
    for e in 0..n_epochs {
        // Section 4.1, Algorithm 5: alpha falls linearly from its first value, which the paper
        // sets to 1, to 1 / n_epochs of it.
        let alpha = learning_rate * (1.0 - e as f64 / n_epochs as f64);
        epoch(&cur, &mut next, dim, g, ab, alpha, n_neg, seed, e as u64);
        std::mem::swap(&mut cur, &mut next);
    }
    cur
}

/// New points placed in a fixed layout.
///
/// Row `i` of `idx` and `w` holds the `k` training neighbours of new point `i` and the weights of
/// those edges (section 3.1). The start is the weighted mean of the neighbours' positions. Then
/// each epoch runs Algorithm 5 for the new point alone: its edges attract it, and negative samples
/// drawn from the training points repel it. The training positions never move and the new points
/// do not see each other, so each point has its own stream, seeded by (seed, epoch, point), and
/// the result does not depend on the number of threads. Decision D-68.
#[allow(clippy::too_many_arguments)]
fn place(
    train: &[f64],
    dim: usize,
    idx: &[usize],
    w: &[f64],
    k: usize,
    ab: (f64, f64),
    n_epochs: usize,
    n_neg: usize,
    seed: u64,
) -> Vec<f64> {
    let m = idx.len() / k;
    let mut cur = vec![0.0; m * dim];
    for i in 0..m {
        let total: f64 = w[i * k..(i + 1) * k].iter().sum();
        for e in 0..k {
            let j = idx[i * k + e];
            for c in 0..dim {
                cur[i * dim + c] += w[i * k + e] * train[j * dim + c] / total;
            }
        }
    }
    for e in 0..n_epochs {
        let alpha = 1.0 - e as f64 / n_epochs as f64;
        cur.par_chunks_mut(CHUNK * dim).enumerate().for_each(|(c, chunk)| {
            for (r, pos) in chunk.chunks_mut(dim).enumerate() {
                let i = c * CHUNK + r;
                let mut rng = Rng::new(point_stream(seed, e as u64, i));
                move_point(
                    pos,
                    &mut rng,
                    train,
                    dim,
                    &idx[i * k..(i + 1) * k],
                    &w[i * k..(i + 1) * k],
                    None,
                    ab,
                    alpha,
                    n_neg,
                );
            }
        });
    }
    cur
}

/// The fuzzy set cross entropy of Definition 10 over all pairs of points, with `mu` the weight of
/// the graph (zero where there is no edge) and `nu = Phi` of the distance.
fn cross_entropy(y: &[f64], dim: usize, g: &Csr, ab: (f64, f64)) -> f64 {
    let n = y.len() / dim;
    let (a, b) = ab;
    // -log(1 - Phi) = log(1 + t) - log(t), t = a r^(2b): the term of a pair with mu = 0.
    let absent = |r2: f64| -> f64 {
        let t = a * r2.powf(b);
        (1.0 + t).ln() - t.ln()
    };
    let parts: Vec<f64> = (0..n.div_ceil(CHUNK))
        .into_par_iter()
        .map(|c| {
            let mut s = 0.0;
            for i in c * CHUNK..((c + 1) * CHUNK).min(n) {
                for j in i + 1..n {
                    s += absent(dist2(y, dim, i, j));
                }
                for e in g.0[i]..g.0[i + 1] {
                    let j = g.1[e];
                    if j <= i {
                        continue;
                    }
                    let mu = g.2[e];
                    let t = a * dist2(y, dim, i, j).powf(b);
                    let log_nu = -(1.0 + t).ln();
                    let log_not = t.ln() - (1.0 + t).ln();
                    let mut term = mu * (mu.ln() - log_nu);
                    if mu < 1.0 {
                        term += (1.0 - mu) * ((1.0 - mu).ln() - log_not);
                    }
                    s += term - absent(dist2(y, dim, i, j));
                }
            }
            s
        })
        .collect();
    parts.iter().sum()
}

// ---------------------------------------------------------------------------------------------
// Python

fn to_usize(v: PyReadonlyArray1<'_, i64>) -> PyResult<Vec<usize>> {
    v.as_array()
        .iter()
        .map(|&x| usize::try_from(x).map_err(|_| PyValueError::new_err("negative index")))
        .collect()
}

fn as_pattern(
    indptr: PyReadonlyArray1<'_, i64>,
    indices: PyReadonlyArray1<'_, i64>,
) -> PyResult<(Vec<usize>, Vec<usize>)> {
    let (indptr, indices) = (to_usize(indptr)?, to_usize(indices)?);
    let n = indptr.len().saturating_sub(1);
    if indptr.is_empty()
        || indptr[n] != indices.len()
        || indices.iter().any(|&j| j >= n)
        || indptr.windows(2).any(|w| w[0] > w[1])
    {
        return Err(PyValueError::new_err("the graph is not a valid compressed row matrix"));
    }
    Ok((indptr, indices))
}

fn as_csr(
    indptr: PyReadonlyArray1<'_, i64>,
    indices: PyReadonlyArray1<'_, i64>,
    data: PyReadonlyArray1<'_, f64>,
) -> PyResult<Csr> {
    let (indptr, indices) = as_pattern(indptr, indices)?;
    let data: Vec<f64> = data.as_array().iter().copied().collect();
    if indices.len() != data.len() {
        return Err(PyValueError::new_err("indices and data differ in length"));
    }
    Ok((indptr, indices, data))
}

fn to_i64(v: Vec<usize>) -> Array1<i64> {
    Array1::from_vec(v.into_iter().map(|x| x as i64).collect())
}

type KnnOut<'py> = (Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<f64>>);

/// The `k` nearest other points of each point, exact: indices and distances, nearest first.
#[pyfunction]
pub fn umap_knn<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    k: usize,
    threads: usize,
) -> PyResult<KnnOut<'py>> {
    let (n, dim) = (data.shape()[0], data.shape()[1]);
    if k == 0 || k >= n || dim == 0 {
        return Err(PyValueError::new_err("need 1 <= k <= n - 1"));
    }
    let flat: Vec<f64> = data.as_array().iter().copied().collect();
    let pool = pool(threads)?;
    let (idx, dist) = py.allow_threads(|| pool.install(|| knn(&Points::new(&flat, n, dim), k)));
    let idx = Array2::from_shape_vec((n, k), idx.into_iter().map(|x| x as i64).collect())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dist = Array2::from_shape_vec((n, k), dist).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((idx.into_pyarray_bound(py), dist.into_pyarray_bound(py)))
}

type FuzzyOut<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
);

/// `rho`, `sigma` and the symmetric graph (compressed rows) from the neighbour table.
#[pyfunction]
pub fn umap_fuzzy<'py>(
    py: Python<'py>,
    idx: PyReadonlyArray2<'py, i64>,
    dist: PyReadonlyArray2<'py, f64>,
) -> PyResult<FuzzyOut<'py>> {
    let (n, k) = (idx.shape()[0], idx.shape()[1]);
    if dist.shape() != [n, k] || k < 2 {
        return Err(PyValueError::new_err("need matching (n, k) tables with k >= 2"));
    }
    let table: Vec<usize> = idx
        .as_array()
        .iter()
        .map(|&x| usize::try_from(x).ok().filter(|&j| j < n))
        .collect::<Option<_>>()
        .ok_or_else(|| PyValueError::new_err("neighbour index out of range"))?;
    let d: Vec<f64> = dist.as_array().iter().copied().collect();
    let mut rho = vec![0.0; n];
    let mut sigma = vec![0.0; n];
    let mut weight = vec![0.0; n * k];
    for i in 0..n {
        let row = &d[i * k..(i + 1) * k];
        let (r, s) = smooth_knn(row);
        rho[i] = r;
        sigma[i] = s;
        for (e, &dj) in row.iter().enumerate() {
            weight[i * k + e] = (-(dj - r).max(0.0) / s).exp();
        }
    }
    let (indptr, indices, data) = fuzzy_union(n, k, &table, &weight);
    Ok((
        Array1::from_vec(rho).into_pyarray_bound(py),
        Array1::from_vec(sigma).into_pyarray_bound(py),
        to_i64(indptr).into_pyarray_bound(py),
        to_i64(indices).into_pyarray_bound(py),
        Array1::from_vec(data).into_pyarray_bound(py),
    ))
}

/// Connected components of a graph in compressed rows, numbered by the lowest index they hold.
#[pyfunction]
pub fn umap_components<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let (indptr, indices) = as_pattern(indptr, indices)?;
    Ok(Array1::from_vec(components(&indptr, &indices)).into_pyarray_bound(py))
}

type SpectralOut<'py> =
    (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>, bool, usize);

/// The `dim` eigenvectors of the normalised Laplacian after the trivial one, of a connected
/// graph: vectors (n, dim), the eigenvalues of `D^(-1/2) A D^(-1/2)` for them (largest first),
/// whether the residual fell below `tol`, and the iterations used. `None` when the graph is too
/// small to hold `dim` such vectors.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn umap_spectral<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    dim: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
    threads: usize,
) -> PyResult<Option<SpectralOut<'py>>> {
    let g = as_csr(indptr, indices, data)?;
    let n = g.0.len() - 1;
    if dim == 0 || g.0.windows(2).any(|w| w[0] == w[1]) {
        return Err(PyValueError::new_err("need dim >= 1 and no isolated vertex"));
    }
    let pool = pool(threads)?;
    let found = py.allow_threads(|| pool.install(|| lobpcg(&g, dim, max_iter, tol, seed)));
    Ok(found.map(|s| {
        let vecs = Array2::from_shape_fn((n, dim), |(i, j)| s.vectors[j][i]);
        (
            vecs.into_pyarray_bound(py),
            Array1::from_vec(s.values).into_pyarray_bound(py),
            s.converged,
            s.iterations,
        )
    }))
}

/// Algorithm 5: the layout, from the start positions `y0` and the graph.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn umap_layout<'py>(
    py: Python<'py>,
    y0: PyReadonlyArray2<'py, f64>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    a: f64,
    b: f64,
    n_epochs: usize,
    n_neg: usize,
    learning_rate: f64,
    seed: u64,
    threads: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (n, dim) = (y0.shape()[0], y0.shape()[1]);
    let g = as_csr(indptr, indices, data)?;
    let positive = a > 0.0 && b > 0.0 && learning_rate > 0.0;
    if g.0.len() != n + 1 || dim == 0 || n_epochs == 0 || !positive {
        return Err(PyValueError::new_err(
            "need a graph on the n points, n_epochs >= 1, a, b, learning_rate > 0",
        ));
    }
    let start: Vec<f64> = y0.as_array().iter().copied().collect();
    let pool = pool(threads)?;
    let ab = (a, b);
    let out = py.allow_threads(|| {
        pool.install(|| layout(&start, dim, &g, ab, n_epochs, n_neg, learning_rate, seed))
    });
    let out = Array2::from_shape_vec((n, dim), out).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out.into_pyarray_bound(py))
}

/// The `k` nearest of the rows of `reference` for each row of `query`, exact: indices into
/// `reference` and distances, nearest first.
#[pyfunction]
pub fn umap_knn_between<'py>(
    py: Python<'py>,
    reference: PyReadonlyArray2<'py, f64>,
    query: PyReadonlyArray2<'py, f64>,
    k: usize,
    threads: usize,
) -> PyResult<KnnOut<'py>> {
    let (n, dim) = (reference.shape()[0], reference.shape()[1]);
    let m = query.shape()[0];
    if query.shape()[1] != dim || m == 0 || k == 0 || k > n || dim == 0 {
        return Err(PyValueError::new_err("need 1 <= k <= n and queries with the same columns"));
    }
    let mut flat: Vec<f64> = reference.as_array().iter().copied().collect();
    flat.extend(query.as_array().iter().copied());
    let pool = pool(threads)?;
    let (idx, dist) =
        py.allow_threads(|| pool.install(|| search(&Points::new(&flat, n + m, dim), n, n + m, n, k)));
    let idx = Array2::from_shape_vec((m, k), idx.into_iter().map(|x| x as i64).collect())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dist = Array2::from_shape_vec((m, k), dist).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((idx.into_pyarray_bound(py), dist.into_pyarray_bound(py)))
}

type PlaceOut<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>);

/// Places new points in a fixed layout from their training neighbours: the positions, then the
/// `sigma` of each new point. The weights have `rho = 0` (local connectivity 0): a new point is
/// not forced to be fully connected to its nearest training point, as in uwot and umap-learn.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn umap_place<'py>(
    py: Python<'py>,
    embedding: PyReadonlyArray2<'py, f64>,
    idx: PyReadonlyArray2<'py, i64>,
    dist: PyReadonlyArray2<'py, f64>,
    a: f64,
    b: f64,
    n_epochs: usize,
    n_neg: usize,
    seed: u64,
    threads: usize,
) -> PyResult<PlaceOut<'py>> {
    let (n, dim) = (embedding.shape()[0], embedding.shape()[1]);
    let (m, k) = (idx.shape()[0], idx.shape()[1]);
    if dist.shape() != [m, k] || m == 0 || k < 2 || dim == 0 || !(a > 0.0 && b > 0.0) {
        return Err(PyValueError::new_err("need matching (m, k) tables with k >= 2 and a, b > 0"));
    }
    let table: Vec<usize> = idx
        .as_array()
        .iter()
        .map(|&x| usize::try_from(x).ok().filter(|&j| j < n))
        .collect::<Option<_>>()
        .ok_or_else(|| PyValueError::new_err("neighbour index out of range"))?;
    let d: Vec<f64> = dist.as_array().iter().copied().collect();
    if d.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err(PyValueError::new_err("distances must be finite and not negative"));
    }
    let train: Vec<f64> = embedding.as_array().iter().copied().collect();
    let mut sigma = vec![0.0; m];
    let mut weight = vec![0.0; m * k];
    for i in 0..m {
        let row = &d[i * k..(i + 1) * k];
        let s = solve_sigma(row, 0.0);
        sigma[i] = s;
        for (e, &dj) in row.iter().enumerate() {
            weight[i * k + e] = (-dj / s).exp();
        }
    }
    let pool = pool(threads)?;
    let out = py.allow_threads(|| {
        pool.install(|| place(&train, dim, &table, &weight, k, (a, b), n_epochs, n_neg, seed))
    });
    let out = Array2::from_shape_vec((m, dim), out).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        out.into_pyarray_bound(py),
        Array1::from_vec(sigma).into_pyarray_bound(py),
    ))
}

/// The cross entropy of Definition 10 between the graph and the embedding, over all pairs.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn umap_cross_entropy<'py>(
    py: Python<'py>,
    y: PyReadonlyArray2<'py, f64>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    a: f64,
    b: f64,
    threads: usize,
) -> PyResult<f64> {
    let (n, dim) = (y.shape()[0], y.shape()[1]);
    let g = as_csr(indptr, indices, data)?;
    if g.0.len() != n + 1 || dim == 0 || !(a > 0.0 && b > 0.0) {
        return Err(PyValueError::new_err("need a graph on the n points and a, b > 0"));
    }
    let flat: Vec<f64> = y.as_array().iter().copied().collect();
    let pool = pool(threads)?;
    Ok(py.allow_threads(|| pool.install(|| cross_entropy(&flat, dim, &g, (a, b)))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobi_finds_the_eigenpairs_of_a_symmetric_matrix() {
        let m = 5;
        let mut rng = Rng::new(3);
        let mut a = vec![0.0; m * m];
        for i in 0..m {
            for j in 0..=i {
                let v = rng.normal();
                a[i * m + j] = v;
                a[j * m + i] = v;
            }
        }
        let original = a.clone();
        let (vals, vecs) = jacobi(&mut a, m);
        for col in 0..m {
            for i in 0..m {
                let av: f64 = (0..m).map(|j| original[i * m + j] * vecs[j * m + col]).sum();
                assert!((av - vals[col] * vecs[i * m + col]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn sigma_solves_the_equation_of_the_paper() {
        let d = [0.5, 0.7, 0.8, 1.1, 1.5, 2.0, 2.2, 3.0];
        let (rho, sigma) = smooth_knn(&d);
        assert_eq!(rho, 0.5);
        let total: f64 = d.iter().map(|&x| (-(x - rho).max(0.0) / sigma).exp()).sum();
        assert!((total - 3.0).abs() < 1e-12, "{total}");
    }

    #[test]
    fn the_union_of_two_directed_edges_is_the_probabilistic_sum() {
        // 0 -> 1 with weight 0.5, 1 -> 0 with weight 0.4, 1 -> 2 with weight 0.3.
        let idx = [1, 2, 0, 2, 0, 1];
        let w = [0.5, 0.0, 0.4, 0.3, 0.0, 0.0];
        let (indptr, indices, data) = fuzzy_union(3, 2, &idx, &w);
        assert_eq!(indptr, vec![0, 1, 3, 4]);
        assert_eq!(indices, vec![1, 0, 2, 1]);
        assert!((data[0] - (0.5 + 0.4 - 0.2)).abs() < 1e-15);
        assert!((data[1] - data[0]).abs() < 1e-15);
        assert!((data[2] - 0.3).abs() < 1e-15 && (data[3] - 0.3).abs() < 1e-15);
    }

    #[test]
    fn the_eigenvalues_of_a_ring_are_found() {
        // On a ring of n vertices S = A / 2 has the eigenvalues cos(2 pi j / n); after the
        // trivial one, the largest is cos(2 pi / n), twice.
        let n = 60;
        let mut indptr = vec![0usize];
        let mut indices = Vec::new();
        for i in 0..n {
            indices.push((i + n - 1) % n);
            indices.push((i + 1) % n);
            indptr.push(indices.len());
        }
        let g: Csr = (indptr, indices, vec![1.0; 2 * n]);
        let s = lobpcg(&g, 2, 2000, 1e-10, 5).expect("spectral");
        let want = (2.0 * std::f64::consts::PI / n as f64).cos();
        assert!(s.converged);
        assert!((s.values[0] - want).abs() < 1e-9 && (s.values[1] - want).abs() < 1e-9);
    }

    #[test]
    fn the_forces_are_the_gradients_of_log_phi_and_log_one_minus_phi() {
        // Central differences in the first coordinate of y_i, against y_j at the origin.
        let (a, b) = (1.58, 0.9);
        let phi = |r2: f64| 1.0 / (1.0 + a * r2.powf(b));
        for &(x, y) in &[(0.6, 0.3), (1.0, 0.0), (2.0, -1.5)] {
            let h = 1e-6;
            let r2 = |dx: f64| (x + dx) * (x + dx) + y * y;
            let up = (phi(r2(h)).ln() - phi(r2(-h)).ln()) / (2.0 * h);
            assert!((attraction(r2(0.0), a, b) * x - up).abs() < 1e-7 * (1.0 + up.abs()));
            let down = ((1.0 - phi(r2(h))).ln() - (1.0 - phi(r2(-h))).ln()) / (2.0 * h);
            // The force has eps in the denominator, so it differs from the exact gradient by a
            // relative eps / r^2.
            let want = repulsion(r2(0.0), a, b) * x;
            assert!((want - down).abs() < 2.0 * EPS / r2(0.0) * down.abs(), "{want} {down}");
        }
    }

    #[test]
    fn the_cross_entropy_is_zero_when_the_graph_is_the_embedding() {
        // Two points with mu = nu: the entropy of equal fuzzy sets is zero.
        let (a, b) = (1.5, 0.9);
        let y = [0.0, 0.0, 1.0, 0.0];
        let nu = 1.0 / (1.0 + a);
        let g: Csr = (vec![0, 1, 2], vec![1, 0], vec![nu, nu]);
        assert!(cross_entropy(&y, 2, &g, (a, b)).abs() < 1e-12);
    }

    #[test]
    fn a_search_between_two_sets_equals_the_brute_force_answer() {
        let (n, m, dim, k) = (300, 70, 4, 6);
        let mut rng = Rng::new(11);
        let flat: Vec<f64> = (0..(n + m) * dim).map(|_| rng.normal()).collect();
        let p = Points::new(&flat, n + m, dim);
        let (idx, dist) = search(&p, n, n + m, n, k);
        for q in 0..m {
            let mut all: Vec<(f64, usize)> = (0..n).map(|j| (p.exact_d2(n + q, j), j)).collect();
            all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            for e in 0..k {
                assert_eq!(idx[q * k + e], all[e].1);
                assert!((dist[q * k + e] - all[e].0.sqrt()).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn a_new_point_with_one_heavy_edge_is_attracted_to_its_neighbour() {
        // One new point, two training points far apart. The edge to the first has weight 1 and
        // the second has weight 0, so with no negative samples the point ends at the first.
        let train = [0.0, 0.0, 10.0, 0.0];
        let out = place(&train, 2, &[0, 1], &[1.0, 0.0], 2, (1.5769, 0.8951), 200, 0, 1);
        assert!(out[0].abs() < 1e-6 && out[1].abs() < 1e-6, "{out:?}");
    }
}
