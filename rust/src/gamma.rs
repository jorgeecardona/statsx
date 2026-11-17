//! Regularized incomplete gamma function and the chi-square distribution built on it.
//!
//! `P(a, x)` uses its power series for `x < a + 1`; `Q(a, x) = 1 - P(a, x)` uses the continued
//! fraction by the modified Lentz method for `x >= a + 1`. Each tail is computed directly, so
//! small upper tails keep their relative accuracy. See DLMF sections 8.7 and 8.9, and Gautschi
//! (1979), ACM Transactions on Mathematical Software 5(4):466-481.

use crate::special::lgamma;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// log of x^a e^-x / Gamma(a).
fn ln_front(a: f64, x: f64) -> f64 {
    a * x.ln() - x - lgamma(a)
}

/// Series for P(a, x), fast for x < a + 1.
fn series(a: f64, x: f64) -> f64 {
    let mut term = 1.0 / a;
    let mut sum = term;
    for n in 1..=100_000 {
        term *= x / (a + n as f64);
        sum += term;
        if term.abs() < sum.abs() * 1e-17 {
            break;
        }
    }
    sum * ln_front(a, x).exp()
}

/// Continued fraction for Q(a, x), fast for x >= a + 1.
fn cont_frac(a: f64, x: f64) -> f64 {
    const TINY: f64 = 1e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / TINY;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..=100_000 {
        let i = i as f64;
        let an = -i * (i - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < TINY {
            d = TINY;
        }
        c = b + an / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < 1e-16 {
            break;
        }
    }
    h * ln_front(a, x).exp()
}

/// P(a, x), the lower regularized incomplete gamma, for a > 0.
pub fn gammainc_p(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else if x < a + 1.0 {
        series(a, x)
    } else {
        1.0 - cont_frac(a, x)
    }
}

/// Q(a, x), the upper regularized incomplete gamma, for a > 0.
pub fn gammainc_q(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        1.0
    } else if x < a + 1.0 {
        1.0 - series(a, x)
    } else {
        cont_frac(a, x)
    }
}

fn check_df(df: f64) -> PyResult<()> {
    if df > 0.0 && df.is_finite() {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!("df must be finite and > 0, got {df}")))
    }
}

/// Chi-square cdf at each ``x``.
#[pyfunction]
pub fn chi2_cdf(x: PyReadonlyArray1<'_, f64>, df: f64) -> PyResult<Vec<f64>> {
    check_df(df)?;
    Ok(x.as_array().iter().map(|&v| gammainc_p(df / 2.0, v / 2.0)).collect())
}

/// Chi-square survival function (the p-value of a statistic) at each ``x``.
#[pyfunction]
pub fn chi2_sf(x: PyReadonlyArray1<'_, f64>, df: f64) -> PyResult<Vec<f64>> {
    check_df(df)?;
    Ok(x.as_array().iter().map(|&v| gammainc_q(df / 2.0, v / 2.0)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(got: f64, want: f64, rel: f64) {
        assert!((got - want).abs() <= rel * want.abs().max(1e-300), "got {got}, want {want}");
    }

    #[test]
    fn lower_and_upper_tails_match_mpmath() {
        // mpmath at 40 digits: (a, x, P, Q).
        let cases = [
            (0.5, 0.3, 0.561_421_973_919_000_136, 0.438_578_026_080_999_864),
            (1.0, 2.0, 0.864_664_716_763_387_308, 0.135_335_283_236_612_692),
            (2.5, 1.0, 0.150_854_963_915_390_364, 0.849_145_036_084_609_636),
            (10.0, 12.0, 0.757_607_838_329_487_651, 0.242_392_161_670_512_349),
            (50.0, 40.0, 0.070_335_066_659_394_954, 0.929_664_933_340_605_046),
            (100.0, 150.0, 0.999_994_075_459_664_516, 5.924_540_335_483_916e-6),
            (0.5, 25.0, 0.999_999_999_998_462_540, 1.537_459_794_428_035e-12),
            (3.0, 60.0, 1.0, 1.629_586_652_937_822e-23),
            (500.0, 520.0, 0.815_308_850_901_256_396, 0.184_691_149_098_743_604),
        ];
        for (a, x, p, q) in cases {
            close(gammainc_p(a, x), p, 1e-12);
            close(gammainc_q(a, x), q, 1e-12);
        }
    }

    #[test]
    fn outside_the_support_the_tails_are_zero_and_one() {
        assert_eq!(gammainc_p(2.0, -1.0), 0.0);
        assert_eq!(gammainc_q(2.0, 0.0), 1.0);
    }
}
