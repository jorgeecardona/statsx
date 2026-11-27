//! The squared Euclidean distance, summed in eight fixed lanes so that it can use vector
//! instructions.
//!
//! A plain running sum cannot be vectorized: floating point addition is not associative, so the
//! compiler may not reorder it. Here the order is written out: value `i` goes to lane `i mod 8`,
//! the lanes are added in a fixed tree, and the values past the last full block of eight are
//! added one by one. Narrow and wide registers, and the scalar code, therefore give the same
//! bits. No fused multiply-add is used. With fewer than eight values the result is the plain
//! running sum. Decision D-76.
//!
//! The first row may be float32: each value is widened to float64, which is exact, so the result
//! has the bits of the same sum over a float64 copy. Decision D-82.

/// Values per block, one per lane.
const LANES: usize = 8;

/// A value of a row of data as it is stored.
pub(crate) trait Value: Copy + Send + Sync {
    /// The value in float64, which is exact.
    fn wide(self) -> f64;
}

impl Value for f32 {
    #[inline(always)]
    fn wide(self) -> f64 {
        f64::from(self)
    }
}

impl Value for f64 {
    #[inline(always)]
    fn wide(self) -> f64 {
        self
    }
}

#[inline(always)]
fn in_lanes<V: Value>(a: &[V], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let blocks_a = a[..n].chunks_exact(LANES);
    let blocks_b = b[..n].chunks_exact(LANES);
    let (rest_a, rest_b) = (blocks_a.remainder(), blocks_b.remainder());
    let mut lane = [0.0f64; LANES];
    for (x, y) in blocks_a.zip(blocks_b) {
        for l in 0..LANES {
            let t = x[l].wide() - y[l];
            lane[l] += t * t;
        }
    }
    let mut rest = 0.0;
    for (x, y) in rest_a.iter().zip(rest_b) {
        let t = x.wide() - y;
        rest += t * t;
    }
    ((lane[0] + lane[4]) + (lane[1] + lane[5])) + ((lane[2] + lane[6]) + (lane[3] + lane[7])) + rest
}

/// The same code compiled for the 256 bit registers of AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn in_lanes_avx2<V: Value>(a: &[V], b: &[f64]) -> f64 {
    in_lanes(a, b)
}

/// The sum of squared differences of two rows of the same length.
#[inline]
pub(crate) fn dist2<V: Value>(a: &[V], b: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if a.len() >= LANES && std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: the processor has AVX2, checked on the line above.
            return unsafe { in_lanes_avx2(a, b) };
        }
    }
    in_lanes(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn the_vector_and_the_scalar_code_give_the_same_bits() {
        let mut rng = Rng::new(5);
        for n in [0usize, 1, 7, 8, 9, 16, 63, 1024, 1031] {
            let a: Vec<f64> = (0..n).map(|_| rng.normal() * 1e3).collect();
            let b: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
            assert_eq!(dist2(&a, &b).to_bits(), in_lanes(&a, &b).to_bits(), "n = {n}");
            let plain: f64 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
            assert!((dist2(&a, &b) - plain).abs() <= 1e-12 * plain);
            if n > 0 && n < LANES {
                assert_eq!(dist2(&a, &b).to_bits(), plain.to_bits());
            }
            let single: Vec<f32> = a.iter().map(|&v| v as f32).collect();
            let copy: Vec<f64> = single.iter().map(|&v| f64::from(v)).collect();
            assert_eq!(dist2(&single, &b).to_bits(), dist2(&copy, &b).to_bits(), "n = {n}");
        }
    }
}
