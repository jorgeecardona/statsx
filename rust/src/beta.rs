//! Regularized incomplete beta function I_x(a, b) and its inverse.
//!
//! I_x(a, b) is the Beta(a, b) CDF. It is evaluated with the continued fraction of Didonato &
//! Morris (1992) and Press et al. (2007), "Numerical Recipes", 3rd ed., section 6.4, using the
//! modified Lentz method, and the symmetry I_x(a, b) = 1 - I_{1-x}(b, a) so that the fraction is
//! used where it converges fast. The inverse solves I_x(a, b) = p by Newton steps inside a
//! shrinking bracket, with bisection when a step leaves the bracket. The Student t CDF and
//! quantile follow from it, as P(|T| > t) = I_{df/(df+t^2)}(df/2, 1/2) (Abramowitz & Stegun
//! 1964, formula 26.7.1). Decisions D-23 and D-28.

use crate::special::lgamma;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// log of x^a (1-x)^b / B(a, b).
fn ln_front(a: f64, b: f64, x: f64) -> f64 {
    lgamma(a + b) - lgamma(a) - lgamma(b) + a * x.ln() + b * (-x).ln_1p()
}

/// Continued fraction of I_x(a, b), valid and fast for x < (a + 1) / (a + b + 2).
fn cont_frac(a: f64, b: f64, x: f64) -> f64 {
    const TINY: f64 = 1e-300;
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=100_000 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
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
    h
}

/// I_x(a, b) for a, b > 0 and any x (0 below 0, 1 above 1).
pub fn betainc(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let front = ln_front(a, b, x).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * cont_frac(a, b, x) / a
    } else {
        // x^a (1-x)^b / B(a, b) is the same under (a, b, x) -> (b, a, 1 - x).
        1.0 - front * cont_frac(b, a, 1.0 - x) / b
    }
}

/// The x with I_x(a, b) = p, for a, b > 0 and 0 <= p <= 1.
pub fn beta_quantile(a: f64, b: f64, p: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }
    // Solve in the tail that keeps the continued fraction accurate.
    if p > 0.5 {
        return 1.0 - beta_quantile(b, a, 1.0 - p);
    }
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    let mut x = (a / (a + b)).clamp(1e-12, 1.0 - 1e-12);
    for _ in 0..300 {
        let f = betainc(a, b, x) - p;
        if f > 0.0 {
            hi = x;
        } else {
            lo = x;
        }
        let pdf = (ln_front(a, b, x) - x.ln() - (-x).ln_1p()).exp();
        let mut next = x - f / pdf;
        if !(next > lo && next < hi) {
            next = 0.5 * (lo + hi);
        }
        if (next - x).abs() <= 1e-15 * x.abs().max(1e-300) {
            return next;
        }
        x = next;
    }
    x
}

/// Student t CDF with `df` degrees of freedom (df > 0, not necessarily an integer):
/// P(|T| > t) = I_{df / (df + t^2)}(df / 2, 1 / 2).
pub fn t_cdf_value(t: f64, df: f64) -> f64 {
    if t == 0.0 {
        return 0.5;
    }
    let tail = 0.5 * betainc(0.5 * df, 0.5, df / (df + t * t));
    if t > 0.0 {
        1.0 - tail
    } else {
        tail
    }
}

/// Student t quantile for a probability in (0, 1).
pub fn t_quantile(p: f64, df: f64) -> f64 {
    if p == 0.5 {
        return 0.0;
    }
    // I_x(df / 2, 1 / 2) = 2 min(p, 1 - p) with x = df / (df + t^2).
    let x = beta_quantile(0.5 * df, 0.5, 2.0 * p.min(1.0 - p));
    let t = (df * (1.0 - x) / x).sqrt();
    if p > 0.5 {
        t
    } else {
        -t
    }
}

/// Standard normal CDF through the complementary error function.
fn norm_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / std::f64::consts::SQRT_2)
}

/// Complementary error function: the Maclaurin series of the error function for x < 2, and the
/// continued fraction of the upper incomplete gamma function (Numerical Recipes, section 6.2)
/// above it.
fn erfc(x: f64) -> f64 {
    if x < 0.0 {
        return 2.0 - erfc(-x);
    }
    if x < 2.0 {
        // erf(x) = 2 / sqrt(pi) * sum_n (-1)^n x^(2n+1) / (n! (2n+1)).
        let (mut term, mut sum) = (x, x);
        for n in 1..120 {
            let n = n as f64;
            term *= -x * x / n;
            sum += term / (2.0 * n + 1.0);
        }
        return 1.0 - 2.0 / std::f64::consts::PI.sqrt() * sum;
    }
    // erfc(x) = exp(-x^2) / (x sqrt(pi)) * 1 / (1 + 1/(2 x^2) / (1 + 2/(2 x^2) / (1 + ...))): the
    // continued fraction K = x + (1/2) / (x + 1 / (x + (3/2) / (x + 2 / (x + ...)))).
    let mut f = 0.0;
    for k in (1..=200).rev() {
        f = (k as f64 / 2.0) / (x + f);
    }
    (-x * x).exp() / ((x + f) * std::f64::consts::PI.sqrt())
}

/// Noncentral t CDF P(T <= t) with `df` degrees of freedom and noncentrality `delta`.
///
/// Algorithm AS 243: Lenth, R. V. (1989), "Cumulative distribution function of the non-central t
/// distribution", Applied Statistics 38(1):185-189, DOI 10.2307/2347693, a Poisson-weighted sum of
/// incomplete beta functions; for df above 4e5 the normal approximation of Abramowitz & Stegun
/// (1964), formula 26.7.10.
pub fn nct_cdf_value(t: f64, df: f64, delta: f64) -> f64 {
    if t < 0.0 {
        return 1.0 - nct_cdf_value(-t, df, -delta);
    }
    if df > 4e5 {
        let s = 1.0 / (4.0 * df);
        return norm_cdf((t * (1.0 - s) - delta) / (1.0 + t * t * 2.0 * s).sqrt());
    }
    let x = t * t / (t * t + df);
    if x <= 0.0 {
        return norm_cdf(-delta);
    }
    let lambda = delta * delta;
    let mut p = 0.5 * (-0.5 * lambda).exp();
    let mut q = (2.0 / std::f64::consts::PI).sqrt() * p * delta;
    let mut s = 0.5 - p;
    let (mut a, b) = (0.5, 0.5 * df);
    let rxb = (1.0 - x).powf(b);
    let albeta = 0.5 * std::f64::consts::PI.ln() + lgamma(b) - lgamma(0.5 + b);
    let mut xodd = betainc(a, b, x);
    let mut godd = 2.0 * rxb * (a * x.ln() - albeta).exp();
    let tnc = b * x;
    let mut xeven = if tnc < f64::EPSILON { tnc } else { 1.0 - rxb };
    let mut geven = tnc * rxb;
    let mut total = p * xodd + q * xeven;
    for it in 1..=2000 {
        a += 1.0;
        xodd -= godd;
        xeven -= geven;
        godd *= x * (a + b - 1.0) / a;
        geven *= x * (a + b - 0.5) / (a + 0.5);
        p *= lambda / (2.0 * it as f64);
        q *= lambda / (2.0 * it as f64 + 1.0);
        total += p * xodd + q * xeven;
        s -= p;
        if s < -1e-10 {
            break;
        }
        if s >= 0.0 && (2.0 * s * (xodd - godd)).abs() < 1e-14 {
            break;
        }
    }
    (total + norm_cdf(-delta)).min(1.0)
}

fn check(a: f64, b: f64) -> PyResult<()> {
    if a > 0.0 && b > 0.0 && a.is_finite() && b.is_finite() {
        Ok(())
    } else {
        Err(PyValueError::new_err("a and b must be finite and > 0"))
    }
}

/// Beta(a, b) CDF at each x.
#[pyfunction]
pub fn beta_cdf(x: PyReadonlyArray1<'_, f64>, a: f64, b: f64) -> PyResult<Vec<f64>> {
    check(a, b)?;
    Ok(x.as_slice()?.iter().map(|&v| betainc(a, b, v)).collect())
}

/// Beta(a, b) quantile at each probability in [0, 1].
#[pyfunction]
pub fn beta_ppf(p: PyReadonlyArray1<'_, f64>, a: f64, b: f64) -> PyResult<Vec<f64>> {
    check(a, b)?;
    let probs = p.as_slice()?;
    if probs.iter().any(|v| !(0.0..=1.0).contains(v)) {
        return Err(PyValueError::new_err("probabilities must be in [0, 1]"));
    }
    Ok(probs.iter().map(|&v| beta_quantile(a, b, v)).collect())
}

/// Student t CDF at each x.
#[pyfunction]
pub fn t_cdf(x: PyReadonlyArray1<'_, f64>, df: f64) -> PyResult<Vec<f64>> {
    if !(df > 0.0 && df.is_finite()) {
        return Err(PyValueError::new_err("df must be finite and > 0"));
    }
    Ok(x.as_slice()?.iter().map(|&v| t_cdf_value(v, df)).collect())
}

/// Student t quantile at each probability in (0, 1).
#[pyfunction]
pub fn t_ppf(p: PyReadonlyArray1<'_, f64>, df: f64) -> PyResult<Vec<f64>> {
    if !(df > 0.0 && df.is_finite()) {
        return Err(PyValueError::new_err("df must be finite and > 0"));
    }
    let probs = p.as_slice()?;
    if probs.iter().any(|v| !(*v > 0.0 && *v < 1.0)) {
        return Err(PyValueError::new_err("probabilities must be in (0, 1)"));
    }
    Ok(probs.iter().map(|&v| t_quantile(v, df)).collect())
}

/// Noncentral t CDF at each x.
#[pyfunction]
pub fn nct_cdf(x: PyReadonlyArray1<'_, f64>, df: f64, delta: f64) -> PyResult<Vec<f64>> {
    if !(df > 0.0 && df.is_finite() && delta.is_finite()) {
        return Err(PyValueError::new_err("df must be finite and > 0, delta finite"));
    }
    Ok(x.as_slice()?.iter().map(|&v| nct_cdf_value(v, df, delta)).collect())
}

#[cfg(test)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    fn close(got: f64, want: f64, rel: f64) {
        assert!((got - want).abs() <= rel * want.abs().max(1e-300), "got {got}, want {want}");
    }

    #[test]
    fn cdf_matches_mpmath() {
        // mpmath at 40 digits.
        close(betainc(0.5, 0.5, 0.3), 0.369_010_119_565_545_375, 1e-13);
        close(betainc(2.0, 5.0, 0.4), 0.766_72, 1e-13);
        close(betainc(10.0, 3.0, 0.85), 0.735_818_086_223_450_85, 1e-13);
        close(betainc(0.5, 10.5, 0.05), 0.694_937_996_942_589_7, 1e-13);
        close(betainc(100.5, 50.5, 0.66), 0.435_139_575_443_790_4, 1e-12);
        close(betainc(1.0, 1.0, 0.25), 0.25, 1e-14);
        close(betainc(3.5, 0.5, 0.99), 0.797_971_695_234_850_9, 1e-12);
        close(betainc(1000.5, 2000.5, 0.33), 0.348_217_037_292_959_3, 1e-11);
    }

    #[test]
    fn quantile_matches_mpmath() {
        close(beta_quantile(0.5, 0.5, 0.025), 0.001_541_333_133_436_012, 1e-12);
        close(beta_quantile(2.0, 5.0, 0.5), 0.264_449_983_295_659_96, 1e-12);
        close(beta_quantile(10.0, 3.0, 0.975), 0.945_139_355_472_007_25, 1e-12);
        close(beta_quantile(0.5, 10.5, 0.025), 4.789_043_315_758_19e-5, 1e-11);
        close(beta_quantile(100.5, 50.5, 0.9), 0.714_216_167_131_967_5, 1e-12);
        close(beta_quantile(3.5, 0.5, 0.001), 0.193_129_056_605_695_28, 1e-12);
        close(beta_quantile(1000.5, 2000.5, 0.975), 0.350_356_377_407_910_64, 1e-11);
        close(beta_quantile(11.0, 1.0, 0.025), 0.715_085_847_081_845_6, 1e-12);
        close(beta_quantile(1.0, 11.0, 0.975), 0.284_914_152_918_154_4, 1e-12);
    }

    #[test]
    fn quantile_inverts_the_cdf_and_handles_the_ends() {
        for &(a, b) in &[(0.5, 0.5), (3.0, 7.0), (40.0, 2.5), (0.2, 30.0)] {
            for &p in &[1e-6, 0.01, 0.3, 0.5, 0.8, 0.999] {
                close(betainc(a, b, beta_quantile(a, b, p)), p, 1e-9);
            }
        }
        assert_eq!(beta_quantile(2.0, 3.0, 0.0), 0.0);
        assert_eq!(beta_quantile(2.0, 3.0, 1.0), 1.0);
        assert_eq!(betainc(2.0, 3.0, -1.0), 0.0);
        assert_eq!(betainc(2.0, 3.0, 2.0), 1.0);
    }

    #[test]
    fn student_t_matches_mpmath() {
        close(t_cdf_value(2.0, 5.0), 0.949_030_260_585_070_82, 1e-13);
        close(t_cdf_value(-1.3, 12.0), 0.109_008_585_541_757_04, 1e-12);
        close(t_cdf_value(0.4, 1.0), 0.621_118_941_590_843_4, 1e-13);
        close(t_cdf_value(10.0, 3.0), 0.998_935_800_470_792_9, 1e-13);
        close(t_cdf_value(3.1, 100.5), 0.998_744_578_630_531_8, 1e-12);
        close(t_cdf_value(-0.05, 7.5), 0.480_714_069_985_727_26, 1e-12);
        close(t_cdf_value(1.7, 2.0), 0.884_383_286_899_218_4, 1e-13);
        close(t_quantile(0.975, 5.0), 2.570_581_835_636_314_8, 1e-12);
        close(t_quantile(0.95, 12.0), 1.782_287_555_649_319_5, 1e-12);
        close(t_quantile(0.6, 1.0), 0.324_919_696_232_906_25, 1e-12);
        close(t_quantile(0.999, 3.0), 10.214_531_852_407_383, 1e-11);
        close(t_quantile(0.025, 100.5), -1.983_850_655_335_799_8, 1e-12);
        close(t_quantile(0.9, 7.5), 1.405_211_846_415_953, 1e-12);
        assert_eq!(t_quantile(0.5, 4.0), 0.0);
    }

    #[test]
    fn noncentral_t_matches_mpmath_integration() {
        // mpmath: the integral of Phi(t sqrt(u / df) - delta) over the chi-square density of u.
        close(nct_cdf_value(2.0, 10.0, 1.0), 0.807_611_562_530_375_26, 1e-12);
        close(nct_cdf_value(1.5, 4.0, 2.5), 0.165_841_563_826_235_06, 1e-12);
        close(nct_cdf_value(-0.5, 7.0, 1.0), 0.070_771_848_236_596_07, 1e-11);
        close(nct_cdf_value(3.0, 30.0, 3.2), 0.416_519_184_859_404_17, 1e-12);
        close(nct_cdf_value(0.0, 12.0, 0.8), 0.211_855_398_583_396_67, 1e-12);
        close(nct_cdf_value(5.0, 9.0, 4.0), 0.708_793_324_760_329_1, 1e-12);
        close(nct_cdf_value(2.2, 5.5, -1.0), 0.996_090_499_019_067_4, 1e-12);
        close(nct_cdf_value(1.0, 100.0, 0.5), 0.690_145_928_737_984_3, 1e-12);
        close(nct_cdf_value(8.0, 20.0, 6.0), 0.882_675_292_194_773_9, 1e-12);
    }

    #[test]
    fn noncentral_t_with_no_shift_is_the_central_t_and_the_normal_cdf_is_accurate() {
        for &(t, df) in &[(0.7, 5.0), (-2.1, 12.0), (3.3, 2.5)] {
            close(nct_cdf_value(t, df, 0.0), t_cdf_value(t, df), 1e-13);
        }
        close(norm_cdf(1.959_963_984_540_054), 0.975, 1e-14);
        close(norm_cdf(-3.0), 0.001_349_898_031_630_094_6, 1e-13);
        close(norm_cdf(0.3), 0.617_911_422_188_952_9, 1e-14);
    }
}
