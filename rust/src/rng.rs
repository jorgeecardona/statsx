//! Pseudorandom numbers: xoshiro256++ seeded by SplitMix64, and unbiased integers in a range.
//!
//! Blackman & Vigna (2021), "Scrambled Linear Pseudorandom Number Generators", ACM TOMS 47(4),
//! DOI 10.1145/3460772. Lemire (2019), "Fast Random Integer Generation in an Interval",
//! ACM TOMS 45(1), DOI 10.1145/3230636. Decision D-4. Normal draws by inversion (AS 241, D-6);
//! gamma draws by Marsaglia & Tsang (2000), "A simple method for generating gamma variables",
//! ACM TOMS 26(3):363-372, DOI 10.1145/358407.358414. Decision D-8.

use crate::special::norm_ppf;

pub struct Rng {
    s: [u64; 4],
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut state = seed;
        let s = [
            splitmix64(&mut state),
            splitmix64(&mut state),
            splitmix64(&mut state),
            splitmix64(&mut state),
        ];
        Rng { s }
    }

    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform integer in [0, n), without modulo bias. `n` must be > 0.
    pub fn below(&mut self, n: u64) -> u64 {
        debug_assert!(n > 0);
        let mut m = u128::from(self.next_u64()) * u128::from(n);
        let mut low = m as u64;
        if low < n {
            let threshold = n.wrapping_neg() % n;
            while low < threshold {
                m = u128::from(self.next_u64()) * u128::from(n);
                low = m as u64;
            }
        }
        (m >> 64) as u64
    }

    /// Uniform in the open interval (0, 1), 53 random bits.
    pub fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// Standard normal by inversion of the CDF.
    pub fn normal(&mut self) -> f64 {
        norm_ppf(self.uniform())
    }

    /// Gamma(shape, 1). For shape < 1, draws Gamma(shape + 1) and multiplies by U^(1/shape)
    /// (Marsaglia & Tsang 2000, section 6). `shape` must be > 0.
    pub fn gamma(&mut self, shape: f64) -> f64 {
        if shape < 1.0 {
            return self.gamma(shape + 1.0) * self.uniform().powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal();
            let v = 1.0 + c * x;
            if v <= 0.0 {
                continue;
            }
            let v = v * v * v;
            let u = self.uniform();
            if u < 1.0 - 0.0331 * x.powi(4) || u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_stream() {
        let (mut a, mut b) = (Rng::new(7), Rng::new(7));
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn below_is_uniform_by_chi_square() {
        let mut rng = Rng::new(1);
        let k = 10usize;
        let draws = 100_000;
        let mut counts = vec![0u32; k];
        for _ in 0..draws {
            counts[rng.below(k as u64) as usize] += 1;
        }
        let expected = draws as f64 / k as f64;
        let chi2: f64 = counts.iter().map(|&c| (c as f64 - expected).powi(2) / expected).sum();
        // 9 degrees of freedom: the 0.999 quantile is 27.88.
        assert!(chi2 < 27.88, "chi2 = {chi2}");
    }

    #[test]
    fn gamma_moments() {
        let mut rng = Rng::new(2);
        for shape in [0.5, 1.0, 3.7, 50.0] {
            let n = 200_000;
            let draws: Vec<f64> = (0..n).map(|_| rng.gamma(shape)).collect();
            let mean = draws.iter().sum::<f64>() / n as f64;
            let var = draws.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
            // mean = var = shape; 6 standard errors of the mean
            assert!((mean - shape).abs() < 6.0 * (shape / n as f64).sqrt(), "{shape}: {mean}");
            assert!((var / shape - 1.0).abs() < 0.03, "{shape}: var {var}");
        }
    }
}
