//! Density clustering: DBSCAN and HDBSCAN*, with exact distances and no neighbour index.
//!
//! DBSCAN: Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996), "A density-based algorithm for
//! discovering clusters in large spatial databases with noise", KDD-96, pp. 226-231. The
//! neighbourhood `N_eps(p) = {q : dist(p, q) <= eps}` holds `p`. A point is a core point when
//! `|N_eps(p)| >= MinPts`. Clusters are the density-connected sets of core points, plus the border
//! points reachable from them. A border point that two clusters reach goes to the one discovered
//! first (section 4.1), which here is the one whose lowest-index core point comes first.
//!
//! HDBSCAN*: Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013), "Density-based clustering
//! based on hierarchical density estimates", PAKDD 2013, LNCS 7819, pp. 160-172; Campello, R. J.
//! G. B., Moulavi, D., Zimek, A., & Sander, J. (2015), ACM TKDD 10(1), article 5. As summarised in
//! McInnes, L., & Healy, J. (2017), "Accelerated hierarchical density based clustering", ICDM
//! Workshops, pp. 33-42, section 2.2: the core distance of a point is its distance to its k-th
//! nearest neighbour (the point itself counts as the first); the mutual reachability distance is
//! `max(core_i, core_j, d_ij)`; the single linkage hierarchy of that distance is read from its
//! minimum spanning tree; it is condensed with a minimum cluster size `m` in terms of
//! `lambda = 1 / epsilon`; the stability of a cluster is the sum over its points of
//! `lambda_max - lambda_min`; and the flat clustering is the set of non-overlapping clusters of
//! largest total stability. Decision D-65.
//!
//! The minimum spanning tree is exact (Prim's algorithm, O(n^2) distances, O(n d) memory), not a
//! neighbour-graph approximation. Nothing here depends on the number of threads: row blocks and
//! column tiles have fixed sizes, cluster membership is the same for any order of unions, and
//! minima are exact.

use std::sync::atomic::{AtomicUsize, Ordering};

use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::kmeans::pool;

/// Rows per block and columns per tile of the distance matrix. Fixed.
pub(crate) const ROWS: usize = 128;
pub(crate) const COLS: usize = 2048;
/// Vertices per chunk in Prim's algorithm. Fixed.
const CHUNK: usize = 2048;
/// Distances below this count as this, so that `lambda = 1 / distance` stays finite when two
/// points coincide.
const MIN_DISTANCE: f64 = 1e-100;
/// A squared distance this close to the threshold, relative to the norms, is recomputed exactly.
const BAND: f64 = 1e-9;

pub(crate) struct Points {
    pub(crate) x: Vec<f64>,
    pub(crate) sq: Vec<f64>,
    pub(crate) n: usize,
    pub(crate) dim: usize,
}

impl Points {
    /// Centre the data first: it does not change a distance, and it makes the expansion
    /// `|a|^2 + |b|^2 - 2 a.b` lose fewer digits when the points are far from the origin.
    pub(crate) fn new(data: &[f64], n: usize, dim: usize) -> Points {
        let mut mean = vec![0.0; dim];
        for i in 0..n {
            for d in 0..dim {
                mean[d] += data[i * dim + d];
            }
        }
        for m in mean.iter_mut() {
            *m /= n as f64;
        }
        let mut x = data.to_vec();
        for i in 0..n {
            for d in 0..dim {
                x[i * dim + d] -= mean[d];
            }
        }
        let sq = (0..n).map(|i| x[i * dim..(i + 1) * dim].iter().map(|v| v * v).sum()).collect();
        Points { x, sq, n, dim }
    }

    pub(crate) fn row(&self, i: usize) -> &[f64] {
        &self.x[i * self.dim..(i + 1) * self.dim]
    }

    pub(crate) fn exact_d2(&self, i: usize, j: usize) -> f64 {
        let (a, b) = (self.row(i), self.row(j));
        let mut s = 0.0;
        for d in 0..self.dim {
            let t = a[d] - b[d];
            s += t * t;
        }
        s
    }

    /// Inner products of `rows` rows from `r0` with `cols` rows from `c0`, into `out`.
    pub(crate) fn gram(&self, r0: usize, rows: usize, c0: usize, cols: usize, out: &mut [f64]) {
        // SAFETY: both operands are contiguous row-major blocks of `x` inside its bounds, and
        // `out` holds rows * cols values.
        unsafe {
            matrixmultiply::dgemm(
                rows,
                self.dim,
                cols,
                1.0,
                self.x.as_ptr().add(r0 * self.dim),
                self.dim as isize,
                1,
                self.x.as_ptr().add(c0 * self.dim),
                1,
                self.dim as isize,
                0.0,
                out.as_mut_ptr(),
                cols as isize,
                1,
            );
        }
    }

    /// Squared distance of `i` and `j` given their inner product `g`.
    pub(crate) fn d2(&self, i: usize, j: usize, g: f64) -> f64 {
        (self.sq[i] + self.sq[j] - 2.0 * g).max(0.0)
    }

    /// Whether `dist(i, j) <= eps`, exact near the threshold.
    fn within(&self, i: usize, j: usize, g: f64, eps2: f64) -> bool {
        let d2 = self.d2(i, j, g);
        if (d2 - eps2).abs() <= BAND * (self.sq[i] + self.sq[j] + eps2) {
            self.exact_d2(i, j) <= eps2
        } else {
            d2 <= eps2
        }
    }
}

pub(crate) fn blocks(n: usize) -> usize {
    n.div_ceil(ROWS)
}

// ---------------------------------------------------------------------------------------------
// k-th nearest neighbour distance

/// Distance from each point to its `k`-th nearest point, the point itself being the first.
fn kth_distance(p: &Points, k: usize) -> Vec<f64> {
    let parts: Vec<Vec<f64>> = (0..blocks(p.n))
        .into_par_iter()
        .map(|b| {
            let r0 = b * ROWS;
            let rows = ROWS.min(p.n - r0);
            let mut best: Vec<Vec<(f64, usize)>> = vec![Vec::with_capacity(k + 1); rows];
            let mut scratch = vec![0.0; rows * COLS];
            let mut c0 = 0;
            while c0 < p.n {
                let cols = COLS.min(p.n - c0);
                p.gram(r0, rows, c0, cols, &mut scratch);
                for r in 0..rows {
                    let list = &mut best[r];
                    for c in 0..cols {
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
            // Exact distances for the k kept, so the answer does not carry the rounding of the
            // expansion.
            (0..rows)
                .map(|r| {
                    let i = r0 + r;
                    let mut exact: Vec<f64> = best[r].iter().map(|&(_, j)| p.exact_d2(i, j)).collect();
                    exact.sort_by(|a, b| a.total_cmp(b));
                    exact[k - 1].sqrt()
                })
                .collect()
        })
        .collect();
    parts.into_iter().flatten().collect()
}

// ---------------------------------------------------------------------------------------------
// DBSCAN

fn find(parent: &[AtomicUsize], mut i: usize) -> usize {
    loop {
        let up = parent[i].load(Ordering::Relaxed);
        if up == i {
            return i;
        }
        let grand = parent[up].load(Ordering::Relaxed);
        // Path halving. A lost race only leaves a longer path.
        let _ = parent[i].compare_exchange(up, grand, Ordering::Relaxed, Ordering::Relaxed);
        i = grand;
    }
}

/// Join the sets of `a` and `b`, keeping the lower index as the root. The sets that result do
/// not depend on the order of the calls.
fn union(parent: &[AtomicUsize], a: usize, b: usize) {
    loop {
        let (ra, rb) = (find(parent, a), find(parent, b));
        if ra == rb {
            return;
        }
        let (lo, hi) = (ra.min(rb), ra.max(rb));
        if parent[hi].compare_exchange(hi, lo, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
            return;
        }
    }
}

/// Labels (`-1` for noise) and the core mask.
fn dbscan(p: &Points, eps: f64, min_samples: usize) -> (Vec<i64>, Vec<bool>) {
    let n = p.n;
    let eps2 = eps * eps;
    // Pass 1: neighbourhood sizes, which give the core points.
    let counts: Vec<Vec<usize>> = (0..blocks(n))
        .into_par_iter()
        .map(|b| {
            let r0 = b * ROWS;
            let rows = ROWS.min(n - r0);
            let mut count = vec![0usize; rows];
            let mut scratch = vec![0.0; rows * COLS];
            let mut c0 = 0;
            while c0 < n {
                let cols = COLS.min(n - c0);
                p.gram(r0, rows, c0, cols, &mut scratch);
                for r in 0..rows {
                    for c in 0..cols {
                        let (i, j) = (r0 + r, c0 + c);
                        if i == j || p.within(i, j, scratch[r * cols + c], eps2) {
                            count[r] += 1;
                        }
                    }
                }
                c0 += cols;
            }
            count
        })
        .collect();
    let core: Vec<bool> = counts.into_iter().flatten().map(|c| c >= min_samples).collect();

    // Pass 2: join core points that lie within eps of each other. Each pair is visited once.
    let parent: Vec<AtomicUsize> = (0..n).map(AtomicUsize::new).collect();
    (0..blocks(n)).into_par_iter().for_each(|b| {
        let r0 = b * ROWS;
        let rows = ROWS.min(n - r0);
        if !core[r0..r0 + rows].iter().any(|&c| c) {
            return;
        }
        let mut scratch = vec![0.0; rows * COLS];
        let mut c0 = (r0 / COLS) * COLS;
        while c0 < n {
            let cols = COLS.min(n - c0);
            p.gram(r0, rows, c0, cols, &mut scratch);
            for r in 0..rows {
                let i = r0 + r;
                if !core[i] {
                    continue;
                }
                for c in 0..cols {
                    let j = c0 + c;
                    if j > i && core[j] && p.within(i, j, scratch[r * cols + c], eps2) {
                        union(&parent, i, j);
                    }
                }
            }
            c0 += cols;
        }
    });

    // Clusters are numbered by their lowest core index.
    let mut rank = vec![usize::MAX; n];
    let mut next = 0;
    for i in 0..n {
        if core[i] && find(&parent, i) == i {
            rank[i] = next;
            next += 1;
        }
    }

    // Pass 3: a border point joins the cluster that is discovered first among those that reach
    // it, which is the one with the lowest root.
    let borders: Vec<Vec<i64>> = (0..blocks(n))
        .into_par_iter()
        .map(|b| {
            let r0 = b * ROWS;
            let rows = ROWS.min(n - r0);
            let mut out = vec![-1i64; rows];
            let mut any = false;
            for r in 0..rows {
                if core[r0 + r] {
                    out[r] = rank[find(&parent, r0 + r)] as i64;
                } else {
                    any = true;
                }
            }
            if !any {
                return out;
            }
            let mut root = vec![usize::MAX; rows];
            let mut scratch = vec![0.0; rows * COLS];
            let mut c0 = 0;
            while c0 < n {
                let cols = COLS.min(n - c0);
                p.gram(r0, rows, c0, cols, &mut scratch);
                for r in 0..rows {
                    if core[r0 + r] {
                        continue;
                    }
                    for c in 0..cols {
                        let j = c0 + c;
                        if core[j] && p.within(r0 + r, j, scratch[r * cols + c], eps2) {
                            root[r] = root[r].min(find(&parent, j));
                        }
                    }
                }
                c0 += cols;
            }
            for r in 0..rows {
                if !core[r0 + r] && root[r] != usize::MAX {
                    out[r] = rank[root[r]] as i64;
                }
            }
            out
        })
        .collect();
    (borders.into_iter().flatten().collect(), core)
}

// ---------------------------------------------------------------------------------------------
// HDBSCAN*

/// Minimum spanning tree of the mutual reachability distance, by Prim's algorithm. Edges come
/// in the order they join the tree, as (from, to, weight).
fn prim(p: &Points, core: &[f64]) -> Vec<(usize, usize, f64)> {
    let n = p.n;
    let mut best = vec![f64::INFINITY; n];
    let mut from = vec![0usize; n];
    let mut in_tree = vec![false; n];
    let mut edges = Vec::with_capacity(n - 1);
    let mut u = 0usize;
    in_tree[0] = true;
    for _ in 1..n {
        let cu = core[u];
        let urow = p.row(u);
        let tree = &in_tree;
        let step = |base: usize, b: &mut [f64], f: &mut [usize]| -> (f64, usize) {
            let mut top = (f64::INFINITY, usize::MAX);
            for k in 0..b.len() {
                let v = base + k;
                if tree[v] {
                    continue;
                }
                let lower = cu.max(core[v]);
                if lower < b[k] {
                    let vrow = p.row(v);
                    let mut s = 0.0;
                    for d in 0..p.dim {
                        let t = urow[d] - vrow[d];
                        s += t * t;
                    }
                    let w = lower.max(s.sqrt());
                    if w < b[k] {
                        b[k] = w;
                        f[k] = u;
                    }
                }
                if b[k] < top.0 {
                    top = (b[k], v);
                }
            }
            top
        };
        let tops: Vec<(f64, usize)> = if n <= CHUNK {
            vec![step(0, &mut best, &mut from)]
        } else {
            best.par_chunks_mut(CHUNK)
                .zip(from.par_chunks_mut(CHUNK))
                .enumerate()
                .map(|(c, (b, f))| step(c * CHUNK, b, f))
                .collect()
        };
        let mut pick = (f64::INFINITY, usize::MAX);
        for t in tops {
            if t.0 < pick.0 {
                pick = t;
            }
        }
        let v = pick.1;
        edges.push((from[v], v, pick.0));
        in_tree[v] = true;
        u = v;
    }
    edges
}

/// The condensed tree, with a stability for each of its clusters.
struct Condensed {
    /// One row per fall-out or split: parent cluster, child (a point below `n`, a cluster at
    /// `n + index`), lambda, and the child's size.
    parent: Vec<usize>,
    child: Vec<usize>,
    lambda: Vec<f64>,
    size: Vec<usize>,
    stability: Vec<f64>,
}

/// The cosine distance `1 - cos` of two unit vectors at Euclidean distance `e` is `e^2 / 2`.
/// Distances up to 1e-14 are rounding of the normalised coordinates: the same direction.
fn cosine_of_chord(e: f64) -> f64 {
    if e <= 1e-14 {
        0.0
    } else {
        e * e / 2.0
    }
}

fn lambda_of(distance: f64) -> f64 {
    1.0 / distance.max(MIN_DISTANCE)
}

/// Single linkage from the tree edges, then the condensed tree for a minimum cluster size.
fn condense(n: usize, edges: &[(usize, usize, f64)], m: usize) -> Condensed {
    let mut order: Vec<usize> = (0..edges.len()).collect();
    order.sort_by(|&a, &b| {
        edges[a].2.total_cmp(&edges[b].2).then(edges[a].0.cmp(&edges[b].0)).then(edges[a].1.cmp(&edges[b].1))
    });
    // Dendrogram nodes: points are 0..n, merges are n..2n-1.
    let mut set: Vec<usize> = (0..2 * n - 1).collect();
    let mut size = vec![1usize; 2 * n - 1];
    let mut left = vec![0usize; n - 1];
    let mut right = vec![0usize; n - 1];
    let mut height = vec![0.0; n - 1];
    fn root(set: &mut [usize], mut i: usize) -> usize {
        while set[i] != i {
            set[i] = set[set[i]];
            i = set[i];
        }
        i
    }
    for (step, &e) in order.iter().enumerate() {
        let (a, b) = (root(&mut set, edges[e].0), root(&mut set, edges[e].1));
        let id = n + step;
        left[step] = a;
        right[step] = b;
        height[step] = edges[e].2;
        size[id] = size[a] + size[b];
        set[a] = id;
        set[b] = id;
    }
    let top = 2 * n - 2;
    let leaves = |start: usize, out: &mut Vec<usize>| {
        let mut stack = vec![start];
        while let Some(v) = stack.pop() {
            if v < n {
                out.push(v);
            } else {
                stack.push(left[v - n]);
                stack.push(right[v - n]);
            }
        }
    };

    let mut tree = Condensed {
        parent: Vec::new(),
        child: Vec::new(),
        lambda: Vec::new(),
        size: Vec::new(),
        stability: Vec::new(),
    };
    let mut birth = vec![0.0f64];
    let mut stack = vec![(top, 0usize)];
    while let Some((v, cluster)) = stack.pop() {
        let lam = lambda_of(height[v - n]);
        // Edges of equal weight join at one level, in an order that carries no meaning. Take
        // every merge of this height at once: its parts are the subtrees below it.
        let mut parts: Vec<usize> = Vec::new();
        let mut open = vec![v];
        while let Some(u) = open.pop() {
            for child in [left[u - n], right[u - n]] {
                if child >= n && height[child - n] == height[v - n] {
                    open.push(child);
                } else {
                    parts.push(child);
                }
            }
        }
        let fall = |node: usize, tree: &mut Condensed| {
            let mut pts = Vec::new();
            leaves(node, &mut pts);
            pts.sort_unstable();
            for q in pts {
                tree.parent.push(cluster);
                tree.child.push(q);
                tree.lambda.push(lam);
                tree.size.push(1);
            }
        };
        let mut big: Vec<(usize, usize)> = Vec::new(); // (lowest point, node)
        for &part in &parts {
            if size[part] >= m {
                let mut pts = Vec::new();
                leaves(part, &mut pts);
                big.push((pts.into_iter().min().unwrap_or(0), part));
            } else {
                fall(part, &mut tree);
            }
        }
        big.sort_unstable();
        match big.len() {
            0 => {}
            1 => stack.push((big[0].1, cluster)),
            _ => {
                for &(_, node) in &big {
                    let id = birth.len();
                    birth.push(lam);
                    tree.parent.push(cluster);
                    tree.child.push(n + id);
                    tree.lambda.push(lam);
                    tree.size.push(size[node]);
                    stack.push((node, id));
                }
            }
        }
    }
    let mut stability = vec![0.0; birth.len()];
    for k in 0..tree.parent.len() {
        let c = tree.parent[k];
        stability[c] += (tree.lambda[k] - birth[c]) * tree.size[k] as f64;
    }
    tree.stability = stability;
    tree
}

/// Select non-overlapping clusters of the largest total stability. Returns, for each cluster of
/// the condensed tree, its label or -1, and the label of each point.
fn select(n: usize, tree: &Condensed, allow_single: bool) -> (Vec<i64>, Vec<i64>) {
    let k = tree.stability.len();
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); k];
    for r in 0..tree.parent.len() {
        if tree.child[r] >= n {
            children[tree.parent[r]].push(tree.child[r] - n);
        }
    }
    // Clusters are numbered parent first, so the reverse order visits children first.
    let mut best = vec![0.0; k];
    let mut keep = vec![false; k];
    for c in (0..k).rev() {
        let below: f64 = children[c].iter().map(|&d| best[d]).sum();
        let selectable = c != 0 || allow_single;
        if selectable && tree.stability[c] >= below {
            best[c] = tree.stability[c];
            keep[c] = true;
        } else {
            best[c] = below;
        }
    }
    // From the top, the first kept cluster on each path is chosen.
    let mut chosen = vec![false; k];
    let mut stack = vec![0usize];
    while let Some(c) = stack.pop() {
        if keep[c] {
            chosen[c] = true;
        } else {
            stack.extend(children[c].iter().copied());
        }
    }
    let mut label = vec![-1i64; k];
    let mut next = 0;
    for c in 0..k {
        if chosen[c] {
            label[c] = next;
            next += 1;
        }
    }
    // Each cluster takes the label of its nearest chosen ancestor, itself included.
    let mut owner = vec![-1i64; k];
    let mut stack = vec![(0usize, -1i64)];
    while let Some((c, inherited)) = stack.pop() {
        let mine = if label[c] >= 0 { label[c] } else { inherited };
        owner[c] = mine;
        for &d in &children[c] {
            stack.push((d, mine));
        }
    }
    let mut point = vec![-1i64; n];
    for r in 0..tree.parent.len() {
        if tree.child[r] < n {
            point[tree.child[r]] = owner[tree.parent[r]];
        }
    }
    (label, point)
}

type DbscanOut<'py> = (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<bool>>);

/// DBSCAN: the label of each point (-1 for noise) and whether it is a core point.
#[pyfunction]
pub fn dbscan_fit<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    eps: f64,
    min_samples: usize,
    threads: usize,
) -> PyResult<DbscanOut<'py>> {
    let (n, dim) = (data.shape()[0], data.shape()[1]);
    if n == 0 || dim == 0 || !(eps > 0.0 && eps.is_finite()) || min_samples == 0 {
        return Err(PyValueError::new_err("need eps > 0 and min_samples >= 1"));
    }
    let flat: Vec<f64> = data.as_array().iter().copied().collect();
    let pool = pool(threads)?;
    let (labels, core) = py.allow_threads(|| {
        pool.install(|| {
            let p = Points::new(&flat, n, dim);
            dbscan(&p, eps, min_samples)
        })
    });
    Ok((Array1::from_vec(labels).into_pyarray_bound(py), Array1::from_vec(core).into_pyarray_bound(py)))
}

/// The distance from each point to its `k`-th nearest point, the point itself counting as the
/// first (so `k = 1` gives zeros).
#[pyfunction]
pub fn kth_distances<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    k: usize,
    threads: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (n, dim) = (data.shape()[0], data.shape()[1]);
    if k == 0 || k > n || dim == 0 {
        return Err(PyValueError::new_err("need 1 <= k <= n"));
    }
    let flat: Vec<f64> = data.as_array().iter().copied().collect();
    let pool = pool(threads)?;
    let out = py.allow_threads(|| pool.install(|| kth_distance(&Points::new(&flat, n, dim), k)));
    Ok(Array1::from_vec(out).into_pyarray_bound(py))
}

type HdbscanOut<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
);

/// HDBSCAN*: point labels, core distances, the tree edges (from, to, weight), the condensed tree
/// (parent, child, lambda, size), the stability and label of each condensed cluster.
#[pyfunction]
pub fn hdbscan_fit<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    min_samples: usize,
    min_cluster_size: usize,
    allow_single_cluster: bool,
    cosine: bool,
    threads: usize,
) -> PyResult<HdbscanOut<'py>> {
    let (n, dim) = (data.shape()[0], data.shape()[1]);
    if n < 2 || dim == 0 || min_samples == 0 || min_samples > n || min_cluster_size < 2 {
        return Err(PyValueError::new_err(
            "need n >= 2, 1 <= min_samples <= n and min_cluster_size >= 2",
        ));
    }
    let flat: Vec<f64> = data.as_array().iter().copied().collect();
    let pool = pool(threads)?;
    let (core, edges, tree, label, point) = py.allow_threads(|| {
        pool.install(|| {
            let p = Points::new(&flat, n, dim);
            let mut core = kth_distance(&p, min_samples);
            let mut edges = prim(&p, &core);
            if cosine {
                // The rows are unit vectors. The map from the Euclidean to the cosine distance
                // rises, so the neighbours, the mutual reachability order and the tree are the
                // same; only the weights change, and with them the densities lambda = 1 / d.
                for v in core.iter_mut() {
                    *v = cosine_of_chord(*v);
                }
                for e in edges.iter_mut() {
                    e.2 = cosine_of_chord(e.2);
                }
            }
            let tree = condense(n, &edges, min_cluster_size);
            let (label, point) = select(n, &tree, allow_single_cluster);
            (core, edges, tree, label, point)
        })
    });
    let mst: Vec<f64> = edges.iter().flat_map(|&(a, b, w)| [a as f64, b as f64, w]).collect();
    let mst = Array2::from_shape_vec((n - 1, 3), mst).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let to_i64 = |v: &[usize]| Array1::from_vec(v.iter().map(|&x| x as i64).collect::<Vec<i64>>());
    Ok((
        Array1::from_vec(point).into_pyarray_bound(py),
        Array1::from_vec(core).into_pyarray_bound(py),
        mst.into_pyarray_bound(py),
        to_i64(&tree.parent).into_pyarray_bound(py),
        to_i64(&tree.child).into_pyarray_bound(py),
        Array1::from_vec(tree.lambda).into_pyarray_bound(py),
        to_i64(&tree.size).into_pyarray_bound(py),
        Array1::from_vec(tree.stability).into_pyarray_bound(py),
        Array1::from_vec(label).into_pyarray_bound(py),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn blobs(per: usize, seed: u64) -> (Vec<f64>, usize) {
        let mut rng = Rng::new(seed);
        let mut x = Vec::new();
        for c in [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]] {
            for _ in 0..per {
                x.push(c[0] + 0.4 * rng.normal());
                x.push(c[1] + 0.4 * rng.normal());
            }
        }
        (x, 3 * per)
    }

    #[test]
    fn dbscan_finds_three_blobs_and_marks_a_far_point_as_noise() {
        let (mut x, mut n) = blobs(60, 1);
        x.extend([50.0, 50.0]);
        n += 1;
        let p = Points::new(&x, n, 2);
        let (labels, core) = dbscan(&p, 1.0, 5);
        for g in 0..3 {
            let first = labels[g * 60];
            assert!(first >= 0);
            assert!(labels[g * 60..g * 60 + 60].iter().filter(|&&l| l == first).count() > 55);
        }
        assert_eq!(labels[n - 1], -1);
        assert!(!core[n - 1]);
    }

    #[test]
    fn the_result_does_not_depend_on_the_thread_count() {
        let (x, n) = blobs(900, 2);
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| {
                let p = Points::new(&x, n, 2);
                let core = kth_distance(&p, 5);
                let edges = prim(&p, &core);
                (dbscan(&p, 0.8, 5), core, edges)
            })
        };
        let (a, b) = (run(1), run(4));
        assert_eq!(a.0, b.0);
        assert_eq!(a.1, b.1);
        assert_eq!(a.2, b.2);
    }

    #[test]
    fn prim_returns_n_minus_one_edges_that_connect_every_point() {
        let (x, n) = blobs(40, 3);
        let p = Points::new(&x, n, 2);
        let core = kth_distance(&p, 4);
        let edges = prim(&p, &core);
        assert_eq!(edges.len(), n - 1);
        let mut set: Vec<usize> = (0..n).collect();
        fn root(s: &mut [usize], mut i: usize) -> usize {
            while s[i] != i {
                i = s[i];
            }
            i
        }
        for &(a, b, _) in &edges {
            let (ra, rb) = (root(&mut set, a), root(&mut set, b));
            assert_ne!(ra, rb);
            set[ra] = rb;
        }
    }

    #[test]
    fn hdbscan_finds_three_blobs() {
        let (x, n) = blobs(80, 4);
        let p = Points::new(&x, n, 2);
        let core = kth_distance(&p, 5);
        let edges = prim(&p, &core);
        let tree = condense(n, &edges, 10);
        let (_, point) = select(n, &tree, false);
        let mut labels: Vec<i64> = point.clone();
        labels.sort_unstable();
        labels.dedup();
        assert!(labels.iter().filter(|&&l| l >= 0).count() == 3, "{labels:?}");
    }
}
