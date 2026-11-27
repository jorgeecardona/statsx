//! The nearest centre of each row: a matrix product rules centres out with a proven error bound,
//! and `dist2` decides between the rest. Decision D-81.
//!
//! The choice is the smallest `dist2` (D-76) over all centres, ties to the lower index. The
//! product only removes centres that cannot be that one, so a product that rounds differently
//! (fused multiply-add, another summation order, another processor) cannot change the choice.
//! The product is computed in float32 or float64 (`Element`).
//!
//! Rows and centres are moved by one vector `m` before the product. A move changes no distance,
//! and it keeps the digits that tell two centres apart when the data are far from the origin.
//! With `x' = fl(x - m)`, `c' = fl(c - m)`, `d` columns and the unit roundoff `u` of the product:
//!
//! 1. `|x' . c' - fl(x' . c')| <= gamma_d |x'| |c'|`, `gamma_d = d u / (1 - d u)`, for any order
//!    of the sum (Higham, N. J. (2002), Accuracy and Stability of Numerical Algorithms, 2nd ed.,
//!    SIAM, section 3.1, with the Cauchy-Schwarz inequality).
//! 2. `|x' - (x - m)| <= u |x'|` (plus the float64 rounding of the move), and the same for `c'`,
//!    so the distance of the moved rows is within `u (|x'| + |c'|)` of the distance asked for.
//!
//! Each term also gets the rounding of the float64 steps and the absolute error of underflow.
//! One bound serves a block of rows, from its longest row and the longest centre. When a product
//! could overflow, every distance of the block is computed.

use crate::simd::dist2;

/// Unit roundoff of float64.
const U64: f64 = f64::EPSILON / 2.0;
/// Every bound is widened by this factor, which covers the rounding of the bound itself.
const SLACK: f64 = 1.001;
/// A distance this small may be lost to float64 underflow in `dist2`.
const FLOOR: f64 = 1e-150;

/// The number type of the matrix product.
pub(crate) trait Element: Copy {
    /// Unit roundoff.
    const UNIT: f64;
    /// The smallest value above zero: the largest error of one operation that underflows.
    const TINY: f64;
    /// The largest finite value.
    const LARGEST: f64;
    /// `value` rounded to this type.
    fn rounded(value: f64) -> Self;
    /// The value in float64, which is exact.
    fn widened(self) -> f64;
    /// `a b^T`, row-major, for `a` of `n` x `dim` and `b` of `k` x `dim`.
    fn product(a: &[Self], b: &[Self], n: usize, k: usize, dim: usize) -> Vec<Self>;
}

macro_rules! element {
    ($t:ty, $gemm:path, $tiny:expr) => {
        impl Element for $t {
            const UNIT: f64 = <$t>::EPSILON as f64 / 2.0;
            const TINY: f64 = $tiny;
            const LARGEST: f64 = <$t>::MAX as f64;

            fn rounded(value: f64) -> Self {
                value as $t
            }

            fn widened(self) -> f64 {
                self as f64
            }

            fn product(a: &[Self], b: &[Self], n: usize, k: usize, dim: usize) -> Vec<Self> {
                assert!(a.len() == n * dim && b.len() == k * dim);
                let mut out = vec![0.0; n * k];
                // SAFETY: a is n x dim and b is k x dim, both row-major and contiguous (checked
                // above), and out holds n x k values.
                unsafe {
                    $gemm(
                        n,
                        dim,
                        k,
                        1.0,
                        a.as_ptr(),
                        dim as isize,
                        1,
                        b.as_ptr(),
                        1,
                        dim as isize,
                        0.0,
                        out.as_mut_ptr(),
                        k as isize,
                        1,
                    );
                }
                out
            }
        }
    };
}

element!(f32, matrixmultiply::sgemm, 1.4e-45);
element!(f64, matrixmultiply::dgemm, 5e-324);

/// `n u / (1 - n u)`, or infinity when `n u` reaches 1 and there is no bound.
fn gamma(n: usize, u: f64) -> f64 {
    let nu = n as f64 * u;
    if nu < 1.0 {
        nu / (1.0 - nu)
    } else {
        f64::INFINITY
    }
}

/// Rows moved by a common vector, in the type of the product, with their squared lengths.
pub(crate) struct Moved<E> {
    values: Vec<E>,
    norm_sq: Vec<f64>,
    dim: usize,
}

impl<E: Element> Moved<E> {
    /// The rows of `x` (row-major, `shift.len()` columns) minus `shift`.
    pub(crate) fn new(x: &[f64], shift: &[f64]) -> Self {
        let dim = shift.len();
        let mut values = vec![E::rounded(0.0); x.len()];
        let norm_sq = values
            .chunks_exact_mut(dim)
            .zip(x.chunks_exact(dim))
            .map(|(out, row)| {
                let mut sum = 0.0;
                for ((o, v), s) in out.iter_mut().zip(row).zip(shift) {
                    *o = E::rounded(v - s);
                    sum += o.widened() * o.widened();
                }
                sum
            })
            .collect();
        Moved { values, norm_sq, dim }
    }

    fn rows(&self) -> usize {
        self.norm_sq.len()
    }

    /// The largest squared length, infinity if a value is not a number.
    fn longest(&self) -> f64 {
        self.norm_sq.iter().fold(0.0, |m, &v| if v <= m { m } else if v > m { v } else { f64::INFINITY })
    }
}

/// How far above the smallest `|c'|^2 - 2 x' . c'` of a row the value of its nearest centre can
/// be, for all rows with `|x'|^2 <= xx` and centres with `|c'|^2 <= cc`, in `dim` columns. None
/// when there is no bound: a product could overflow, or a value is not finite.
fn margin<E: Element>(dim: usize, xx: f64, cc: f64) -> Option<f64> {
    let (x, c) = (xx.sqrt(), cc.sqrt());
    // `dist2` of the nearest centre is at most this factor above that of any other.
    let ratio = 1.0 + 4.0 * gamma(dim + 4, U64) + 1e-12;
    let root = (dim as f64).sqrt();
    // the error of `|x'|^2 + value` against the squared distance of the moved rows
    let error = 2.0 * gamma(dim + 2, E::UNIT) * x * c
        + gamma(dim + 4, U64) * (xx + cc + 2.0 * x * c)
        + 4.0 * dim as f64 * E::TINY;
    // the error of that distance against the one of the rows as given
    let moved = (E::UNIT + 4.0 * U64) * (x + c) + root * E::TINY;
    // With r the distance of the moved rows, the nearest centre b and any centre m have
    // r_b <= ratio r_m + extra, and r_m <= |x'| + |c'|.
    let extra = (ratio + 1.0) * moved + FLOOR;
    let length = SLACK * (x + c);
    let apart = (ratio * ratio - 1.0) * length * length + 2.0 * ratio * extra * length + extra * extra;
    let margin = SLACK * (2.0 * error + apart);
    (x * c < E::LARGEST / 4.0 && margin.is_finite()).then_some(margin)
}

/// `|c'|^2 - 2 x' . c'` of one row for every centre, written to `values`. Returns the position
/// of the smallest (the first of equal ones), that value and the second smallest.
#[inline]
fn smallest<E: Element>(values: &mut [f64], norm_sq: &[f64], product: &[E]) -> (usize, f64, f64) {
    let (mut at, mut least, mut second) = (0, f64::INFINITY, f64::INFINITY);
    for (j, ((out, &cc), &g)) in values.iter_mut().zip(norm_sq).zip(product).enumerate() {
        let v = cc - 2.0 * g.widened();
        *out = v;
        if v < least {
            (at, second, least) = (j, least, v);
        } else if v < second {
            second = v;
        }
    }
    (at, least, second)
}

/// The nearest of the centres `candidates` to `row` by `dist2`, the first of those at the same
/// distance.
fn choose(row: &[f64], c: &[f64], candidates: impl Iterator<Item = usize>) -> (usize, f64) {
    let dim = row.len();
    let mut best: Option<(usize, f64)> = None;
    for j in candidates {
        let d = dist2(row, &c[j * dim..(j + 1) * dim]);
        if best.is_none_or(|(_, least)| d < least) {
            best = Some((j, d));
        }
    }
    best.expect("a centre is always a candidate")
}

/// For each row of `x`, the index of the nearest row of `c` and the squared distance to it, from
/// every distance.
pub(crate) fn exact(x: &[f64], c: &[f64], dim: usize) -> Vec<(usize, f64)> {
    let k = c.len() / dim;
    x.chunks_exact(dim).map(|row| choose(row, c, 0..k)).collect()
}

/// The same result as `exact`, with the distances to the centres that the float32 product cannot
/// rule out.
///
/// `x` and `c` are the values as given, row-major with `shift.len()` columns; `centres` is `c`
/// moved by the same `shift`.
pub(crate) fn screened<E: Element>(
    x: &[f64],
    c: &[f64],
    centres: &Moved<E>,
    shift: &[f64],
) -> Vec<(usize, f64)> {
    let dim = shift.len();
    assert!(centres.dim == dim && c.len() == centres.rows() * dim && !c.is_empty());
    let rows = Moved::<E>::new(x, shift);
    let Some(margin) = margin::<E>(dim, rows.longest(), centres.longest()) else {
        return exact(x, c, dim);
    };
    let k = centres.rows();
    let products = E::product(&rows.values, &centres.values, rows.rows(), k, dim);
    let mut values = vec![0.0; k];
    x.chunks_exact(dim)
        .zip(products.chunks_exact(k))
        .map(|(row, product)| {
            let (at, least, second) = smallest(&mut values, &centres.norm_sq, product);
            let limit = least + margin;
            if second > limit {
                (at, dist2(row, &c[at * dim..(at + 1) * dim]))
            } else {
                choose(row, c, (0..k).filter(|&j| values[j] <= limit))
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn mean(c: &[f64], dim: usize) -> Vec<f64> {
        let k = (c.len() / dim) as f64;
        (0..dim).map(|d| c.chunks(dim).map(|r| r[d]).sum::<f64>() / k).collect()
    }

    /// Rows and centres of one kind of hard case.
    fn case(kind: &str, dim: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let mut rng = Rng::new(seed);
        let (n, k) = (200, 12);
        let (scale, offset) = match kind {
            "far" => (1.0, 1e9),
            "huge" => (1e30, 1e40),
            "tiny" => (1e-42, 0.0),
            "wide" => (1e20, 0.0),
            _ => (1.0, 0.0),
        };
        let mut draw = |count: usize| -> Vec<f64> {
            (0..count * dim).map(|_| offset + scale * rng.normal()).collect()
        };
        let (mut x, mut c) = (draw(n), draw(k));
        if kind == "ties" {
            // Centres that are copies or differ in the last digit of one value, and rows that
            // sit on a centre or half way between two.
            for j in 1..k {
                let (head, tail) = c.split_at_mut(j * dim);
                tail[..dim].copy_from_slice(&head[..dim]);
                if j % 2 == 0 {
                    let v = &mut tail[j % dim];
                    *v = f64::from_bits(v.to_bits() + j as u64);
                }
            }
            for (i, row) in x.chunks_mut(dim).enumerate().take(100) {
                row.copy_from_slice(&c[(i % k) * dim..(i % k + 1) * dim]);
                if i % 2 == 0 {
                    row[0] += 0.5;
                }
            }
        }
        (x, c)
    }

    const KINDS: [&str; 6] = ["plain", "far", "huge", "tiny", "wide", "ties"];

    /// How many centres the product leaves for each row, and whether the nearest is among them.
    fn left<E: Element>(x: &[f64], c: &[f64], dim: usize) -> Vec<(usize, bool)> {
        let shift = mean(c, dim);
        let (rows, centres) = (Moved::<E>::new(x, &shift), Moved::<E>::new(c, &shift));
        let k = centres.rows();
        let truth = exact(x, c, dim);
        let Some(margin) = margin::<E>(dim, rows.longest(), centres.longest()) else {
            return truth.iter().map(|_| (k, true)).collect();
        };
        let products = E::product(&rows.values, &centres.values, rows.rows(), k, dim);
        let mut values = vec![0.0; k];
        truth
            .iter()
            .zip(products.chunks_exact(k))
            .map(|(&(nearest, _), product)| {
                let limit = smallest(&mut values, &centres.norm_sq, product).1 + margin;
                (values.iter().filter(|&&v| v <= limit).count(), values[nearest] <= limit)
            })
            .collect()
    }

    fn found<E: Element>(x: &[f64], c: &[f64], dim: usize) -> Vec<(usize, f64)> {
        let shift = mean(c, dim);
        screened(x, c, &Moved::<E>::new(c, &shift), &shift)
    }

    #[test]
    fn the_bound_never_rules_out_the_nearest_centre() {
        for kind in KINDS {
            for dim in [1usize, 2, 9, 64, 1031] {
                for seed in 0..4 {
                    let (x, c) = case(kind, dim, seed);
                    for rows in [left::<f32>(&x, &c, dim), left::<f64>(&x, &c, dim)] {
                        let ruled_out = rows.iter().filter(|r| !r.1).count();
                        assert_eq!(ruled_out, 0, "{kind} {dim} {seed}");
                    }
                }
            }
        }
    }

    #[test]
    fn the_screen_never_changes_the_nearest_centre() {
        for kind in KINDS {
            for dim in [1usize, 2, 9, 64, 1031] {
                for seed in 0..4 {
                    let (x, c) = case(kind, dim, seed);
                    let truth = exact(&x, &c, dim);
                    assert_eq!(found::<f32>(&x, &c, dim), truth, "f32 {kind} {dim} {seed}");
                    assert_eq!(found::<f64>(&x, &c, dim), truth, "f64 {kind} {dim} {seed}");
                }
            }
        }
    }

    #[test]
    fn on_spread_data_the_screen_leaves_about_one_centre_per_row() {
        let (x, c) = case("plain", 1024, 3);
        for rows in [left::<f32>(&x, &c, 1024), left::<f64>(&x, &c, 1024)] {
            let kept: usize = rows.iter().map(|r| r.0).sum();
            assert!(kept < rows.len() * 2, "{kept} centres for {} rows", rows.len());
        }
    }
}
