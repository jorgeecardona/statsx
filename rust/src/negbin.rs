//! Negative binomial log-likelihood on the log-mean scale, with derivatives.
//!
//! Mean mu = exp(eta), size alpha: Var = mu + mu^2 / alpha (the NB2 form of Cameron & Trivedi
//! (2013), "Regression Analysis of Count Data", 2nd ed., Cambridge University Press,
//! DOI 10.1017/CBO9781139013567, chapter 3). Decision D-11.

use crate::special::{digamma, lgamma};

/// log p(y | eta, alpha). `y` is a count (as f64), `alpha` > 0.
pub fn logpmf(y: f64, eta: f64, alpha: f64) -> f64 {
    let mu = eta.exp();
    // log(alpha / (alpha + mu)) and log(mu / (alpha + mu)) without overflow in mu.
    let log_a_frac = -(1.0 + mu / alpha).ln();
    let log_mu_frac = eta - (alpha + mu).ln();
    let y_term = if y > 0.0 { y * log_mu_frac } else { 0.0 };
    lgamma(y + alpha) - lgamma(alpha) - lgamma(y + 1.0) + alpha * log_a_frac + y_term
}

/// First and second derivatives of `logpmf` in eta.
pub fn d_eta(y: f64, eta: f64, alpha: f64) -> (f64, f64) {
    let mu = eta.exp();
    let w = mu / (alpha + mu);
    (y - (y + alpha) * w, -(y + alpha) * w * (1.0 - w))
}

/// Derivative of `logpmf` in alpha.
pub fn d_alpha(y: f64, eta: f64, alpha: f64) -> f64 {
    let mu = eta.exp();
    digamma(y + alpha) - digamma(alpha) - (1.0 + mu / alpha).ln() + (mu - y) / (alpha + mu)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_mpmath() {
        // mpmath at 40 digits.
        let got = logpmf(7.0, 4.2_f64.ln(), 2.5);
        assert!((got + 2.854_213_804_801_162_5).abs() < 1e-13, "{got}");
        let got = logpmf(0.0, 0.3_f64.ln(), 1000.0);
        assert!((got + 0.299_955_008_997_975_5).abs() < 1e-13, "{got}");
    }

    #[test]
    fn derivatives_match_finite_differences() {
        let h = 1e-5;
        for &(y, eta, alpha) in &[(0.0, -1.0, 0.7), (3.0, 1.2, 5.0), (40.0, 3.0, 2.0), (1.0, 6.0, 50.0)] {
            let (g, hess) = d_eta(y, eta, alpha);
            let fd_g = (logpmf(y, eta + h, alpha) - logpmf(y, eta - h, alpha)) / (2.0 * h);
            let fd_h = (d_eta(y, eta + h, alpha).0 - d_eta(y, eta - h, alpha).0) / (2.0 * h);
            let fd_a = (logpmf(y, eta, alpha + h) - logpmf(y, eta, alpha - h)) / (2.0 * h);
            assert!((g - fd_g).abs() < 1e-6 * g.abs().max(1.0), "grad {y} {eta} {alpha}");
            assert!((hess - fd_h).abs() < 1e-6 * hess.abs().max(1.0), "hess {y} {eta} {alpha}");
            let da = d_alpha(y, eta, alpha);
            assert!((da - fd_a).abs() < 1e-6 * da.abs().max(1.0), "alpha {y} {eta} {alpha}");
        }
    }

    #[test]
    fn concave_in_eta() {
        for eta in [-5.0, 0.0, 5.0] {
            assert!(d_eta(3.0, eta, 1.5).1 < 0.0);
        }
    }

    #[test]
    fn large_eta_does_not_overflow() {
        assert!(logpmf(1e6, 14.0, 3.0).is_finite());
    }
}
