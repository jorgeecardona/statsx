//! Bootstrap resample indices for four resample units: i.i.d., stratified, clustered, block.
//!
//! Efron (1979), DOI 10.1214/aos/1176344552 (i.i.d.). Davison & Hinkley (1997), "Bootstrap
//! Methods and their Application", chapter 3, DOI 10.1017/CBO9780511802843 (stratified and
//! clustered). Kunsch (1989), DOI 10.1214/aos/1176347265 (moving block). Decision D-3.

use crate::rng::Rng;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// One replicate's indices into the original sample, appended to `out`.
fn replicate(unit: &Unit, n: usize, rng: &mut Rng, out: &mut Vec<u64>) {
    match unit {
        Unit::Iid => (0..n).for_each(|_| out.push(rng.below(n as u64))),
        Unit::Stratified(groups) => {
            for members in groups {
                for _ in 0..members.len() {
                    out.push(members[rng.below(members.len() as u64) as usize]);
                }
            }
        }
        Unit::Clustered(groups) => {
            for _ in 0..groups.len() {
                out.extend_from_slice(&groups[rng.below(groups.len() as u64) as usize]);
            }
        }
        Unit::Block(length) => {
            let starts = (n - length + 1) as u64;
            let mut taken = 0;
            while taken < n {
                let start = rng.below(starts) as usize;
                let take = (*length).min(n - taken);
                out.extend((start..start + take).map(|i| i as u64));
                taken += take;
            }
        }
    }
}

enum Unit {
    Iid,
    Stratified(Vec<Vec<u64>>),
    Clustered(Vec<Vec<u64>>),
    Block(usize),
}

/// Group observation indices by label, in order of first appearance.
fn group_by(labels: &[i64]) -> Vec<Vec<u64>> {
    let mut order: Vec<i64> = Vec::new();
    let mut groups: Vec<Vec<u64>> = Vec::new();
    for (i, label) in labels.iter().enumerate() {
        match order.iter().position(|l| l == label) {
            Some(g) => groups[g].push(i as u64),
            None => {
                order.push(*label);
                groups.push(vec![i as u64]);
            }
        }
    }
    groups
}

/// Indices of `n_resamples` replicates, flat, with `offsets` (length n_resamples + 1).
///
/// `kind` is "iid", "stratified", "clustered" or "block". `labels` holds one integer label per
/// observation for stratified and clustered; `block_length` is used for block.
#[pyfunction]
#[pyo3(signature = (n, kind, n_resamples, seed, labels=None, block_length=0))]
#[allow(clippy::type_complexity)]
pub fn bootstrap_indices<'py>(
    py: Python<'py>,
    n: usize,
    kind: &str,
    n_resamples: usize,
    seed: u64,
    labels: Option<PyReadonlyArray1<'py, i64>>,
    block_length: usize,
) -> PyResult<(Bound<'py, PyArray1<u64>>, Bound<'py, PyArray1<u64>>)> {
    if n == 0 {
        return Err(PyValueError::new_err("cannot bootstrap an empty sample"));
    }
    if n_resamples == 0 {
        return Err(PyValueError::new_err("n_resamples must be at least 1"));
    }
    let grouped = |what: &str| -> PyResult<Vec<Vec<u64>>> {
        let labels = labels
            .as_ref()
            .ok_or_else(|| PyValueError::new_err(format!("{what} needs labels")))?;
        let labels = labels.as_slice()?;
        if labels.len() != n {
            return Err(PyValueError::new_err(format!(
                "{} labels for {n} observations",
                labels.len()
            )));
        }
        Ok(group_by(labels))
    };
    let unit = match kind {
        "iid" => Unit::Iid,
        "stratified" => Unit::Stratified(grouped("stratified")?),
        "clustered" => {
            let groups = grouped("clustered")?;
            if groups.len() < 2 {
                return Err(PyValueError::new_err("clustered needs at least 2 clusters"));
            }
            Unit::Clustered(groups)
        }
        "block" => {
            if block_length == 0 || block_length > n {
                return Err(PyValueError::new_err(format!(
                    "block_length must be in [1, {n}], got {block_length}"
                )));
            }
            Unit::Block(block_length)
        }
        other => return Err(PyValueError::new_err(format!("unknown resample unit {other:?}"))),
    };
    let (indices, offsets) = py.allow_threads(|| {
        let mut rng = Rng::new(seed);
        let mut indices = Vec::with_capacity(n * n_resamples);
        let mut offsets = Vec::with_capacity(n_resamples + 1);
        offsets.push(0u64);
        for _ in 0..n_resamples {
            replicate(&unit, n, &mut rng, &mut indices);
            offsets.push(indices.len() as u64);
        }
        (indices, offsets)
    });
    Ok((indices.into_pyarray_bound(py), offsets.into_pyarray_bound(py)))
}
