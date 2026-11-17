//! Pareto smoothed importance sampling: smoothed weights and the k-hat diagnostic.
//!
//! Vehtari, Simpson, Gelman, Yao & Gabry (2024), "Pareto smoothed importance sampling",
//! JMLR 25(72):1-58. The generalized Pareto fit is the empirical Bayes estimate of Zhang &
//! Stephens (2009), "A new and efficient estimation method for the generalized Pareto
//! distribution", Technometrics 51(3):316-325, DOI 10.1198/tech.2009.08017, with the weakly
//! informative prior on k of the 2024 paper (10 pseudo-observations at 0.5). Decision D-1.

/// Smoothed, normalized weights, k-hat, and the importance sampling effective sample size.
pub struct Psis {
    pub weights: Vec<f64>,
    pub khat: f64,
    pub ess: f64,
}

/// Generalized Pareto (shape k, scale sigma) fit to positive sorted excesses `x`.
fn gpd_fit(x: &[f64]) -> (f64, f64) {
    let n = x.len();
    let nf = n as f64;
    let m = 30 + (nf.sqrt() as usize);
    let quartile = x[((nf / 4.0 + 0.5) as usize).max(1) - 1];
    let x_max = x[n - 1];
    let b: Vec<f64> = (1..=m)
        .map(|j| (1.0 - (m as f64 / (j as f64 - 0.5)).sqrt()) / (3.0 * quartile) + 1.0 / x_max)
        .collect();
    let k_of = |bj: f64| x.iter().map(|v| (-bj * v).ln_1p()).sum::<f64>() / nf;
    let len_scale: Vec<f64> = b
        .iter()
        .map(|&bj| {
            let k = k_of(bj);
            nf * ((-bj / k).ln() - k - 1.0)
        })
        .collect();
    let mut w: Vec<f64> = len_scale
        .iter()
        .map(|&li| 1.0 / len_scale.iter().map(|&lj| (lj - li).exp()).sum::<f64>())
        .collect();
    for wi in w.iter_mut() {
        if wi.is_nan() || *wi < 10.0 * f64::EPSILON {
            *wi = 0.0;
        }
    }
    let total: f64 = w.iter().sum();
    let b_post: f64 = b.iter().zip(&w).map(|(bj, wj)| bj * wj / total).sum();
    let k = k_of(b_post);
    let sigma = -k / b_post;
    ((nf * k + 5.0) / (nf + 10.0), sigma)
}

/// Quantile of the generalized Pareto distribution.
fn gpd_quantile(p: f64, k: f64, sigma: f64) -> f64 {
    if k.abs() < 1e-12 {
        -sigma * (-p).ln_1p()
    } else {
        sigma * (-k * (-p).ln_1p()).exp_m1() / k
    }
}

/// Smooth the upper tail of the raw log weights and return normalized weights and k-hat.
/// With 4 or fewer tail draws no smoothing is possible and k-hat is infinite.
pub fn psis(log_w: &[f64]) -> Psis {
    let s = log_w.len();
    let max = log_w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut x: Vec<f64> = log_w.iter().map(|v| v - max).collect();
    let tail_len = (0.2 * s as f64).min(3.0 * (s as f64).sqrt()).ceil() as usize;
    let mut order: Vec<usize> = (0..s).collect();
    order.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut khat = f64::INFINITY;
    if tail_len > 4 && tail_len < s {
        let cutoff_log = x[order[s - tail_len - 1]].max(f64::MIN_POSITIVE.ln());
        let cutoff = cutoff_log.exp();
        let tail: Vec<usize> = order[s - tail_len..].iter().cloned().filter(|&i| x[i] > cutoff_log).collect();
        if tail.len() > 4 {
            let excess: Vec<f64> = tail.iter().map(|&i| x[i].exp() - cutoff).collect();
            // `order` is ascending, so `excess` is ascending as well.
            let (k, sigma) = gpd_fit(&excess);
            khat = k;
            let m = tail.len() as f64;
            for (rank, &i) in tail.iter().enumerate() {
                let p = (rank as f64 + 0.5) / m;
                let smoothed = (gpd_quantile(p, k, sigma) + cutoff).ln();
                // A weight never exceeds the largest raw weight (log 0 after centering).
                x[i] = smoothed.min(0.0);
            }
        }
    }
    let mut w: Vec<f64> = x.iter().map(|v| v.exp()).collect();
    let total: f64 = w.iter().sum();
    for wi in w.iter_mut() {
        *wi /= total;
    }
    let ess = 1.0 / w.iter().map(|v| v * v).sum::<f64>();
    Psis { weights: w, khat, ess }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn recovers_pareto_tail_index() {
        // Weights w = U^(-xi) have a Pareto tail with shape xi. The mean k-hat over 30
        // samples is near xi; one sample has a standard error of about 0.1.
        for &xi in &[0.2, 0.5, 0.9] {
            let mut rng = Rng::new(11);
            let mean: f64 = (0..30)
                .map(|_| {
                    let log_w: Vec<f64> = (0..4000).map(|_| -xi * rng.uniform().ln()).collect();
                    psis(&log_w).khat
                })
                .sum::<f64>()
                / 30.0;
            assert!((mean - xi).abs() < 0.06, "xi {xi} mean khat {mean}");
        }
    }

    #[test]
    fn light_tails_give_small_khat_and_high_ess() {
        let mut rng = Rng::new(5);
        let log_w: Vec<f64> = (0..2000).map(|_| 0.1 * rng.normal()).collect();
        let r = psis(&log_w);
        assert!(r.khat < 0.3, "{}", r.khat);
        assert!(r.ess > 1800.0, "{}", r.ess);
        assert!((r.weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn tiny_samples_are_not_smoothed() {
        assert!(psis(&[0.0, -1.0, -2.0]).khat.is_infinite());
    }
}
