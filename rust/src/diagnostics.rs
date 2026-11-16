//! MCMC convergence diagnostics: rank-normalized split R-hat, bulk ESS and tail ESS.
//!
//! Vehtari, Gelman, Simpson, Carpenter & Burkner (2021), "Rank-normalization, folding, and
//! localization: An improved R-hat for assessing convergence of MCMC", Bayesian Analysis
//! 16(2):667-718, DOI 10.1214/20-BA1221. ESS uses Geyer's initial monotone sequence
//! (Geyer 1992, "Practical Markov chain Monte Carlo", Statistical Science 7(4):473-483,
//! DOI 10.1214/ss/1177011137). Decision D-5.

use crate::special::norm_ppf;
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type Chains = Vec<Vec<f64>>;

/// Split every chain into its first and last halves (the middle draw is dropped for odd length).
fn split(chains: &Chains) -> Chains {
    let half = chains[0].len() / 2;
    chains
        .iter()
        .flat_map(|c| [c[..half].to_vec(), c[c.len() - half..].to_vec()])
        .collect()
}

/// Replace every draw by the normal score of its pooled rank, ties averaged:
/// z = Phi^-1((r - 3/8) / (S + 1/4)).
fn rank_normalize(chains: &Chains) -> Chains {
    let n = chains[0].len();
    let flat: Vec<f64> = chains.iter().flatten().copied().collect();
    let total = flat.len();
    let mut order: Vec<usize> = (0..total).collect();
    order.sort_by(|&a, &b| flat[a].total_cmp(&flat[b]));
    let mut z = vec![0.0; total];
    let mut i = 0;
    while i < total {
        let mut j = i;
        while j + 1 < total && flat[order[j + 1]] == flat[order[i]] {
            j += 1;
        }
        let rank = (i + j) as f64 / 2.0 + 1.0;
        let score = norm_ppf((rank - 0.375) / (total as f64 + 0.25));
        for &k in &order[i..=j] {
            z[k] = score;
        }
        i = j + 1;
    }
    z.chunks(n).map(<[f64]>::to_vec).collect()
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

/// Sample variance with denominator len - 1.
fn var(x: &[f64]) -> f64 {
    let m = mean(x);
    x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (x.len() as f64 - 1.0)
}

/// Classic potential scale reduction on already split chains; NaN when within-chain variance is 0.
fn rhat_basic(chains: &Chains) -> f64 {
    let n = chains[0].len() as f64;
    let means: Vec<f64> = chains.iter().map(|c| mean(c)).collect();
    let w = mean(&chains.iter().map(|c| var(c)).collect::<Vec<_>>());
    if w <= 0.0 {
        return f64::NAN;
    }
    let var_plus = (n - 1.0) / n * w + var(&means);
    (var_plus / w).sqrt()
}

/// Effective sample size of split chains (Vehtari et al. 2021, eq. 10-11; Geyer 1992).
fn ess_basic(chains: &Chains) -> f64 {
    let m = chains.len();
    let n = chains[0].len();
    let nf = n as f64;
    let means: Vec<f64> = chains.iter().map(|c| mean(c)).collect();
    // Mean over chains of the biased autocovariance at `lag`.
    let acov = |lag: usize| -> f64 {
        chains
            .iter()
            .zip(&means)
            .map(|(c, mu)| (0..n - lag).map(|t| (c[t] - mu) * (c[t + lag] - mu)).sum::<f64>() / nf)
            .sum::<f64>()
            / m as f64
    };
    let mean_var = acov(0) * nf / (nf - 1.0);
    if mean_var <= 0.0 {
        return f64::NAN;
    }
    let var_plus = mean_var * (nf - 1.0) / nf + if m > 1 { var(&means) } else { 0.0 };
    let rho = |lag: usize| 1.0 - (mean_var - acov(lag)) / var_plus;

    let mut rho_t = vec![0.0; n];
    rho_t[0] = 1.0;
    let mut even = 1.0;
    let mut odd = rho(1);
    rho_t[1] = odd;
    let mut t = 1;
    while t + 5 < n && even + odd > 0.0 {
        even = rho(t + 1);
        odd = rho(t + 2);
        if even + odd >= 0.0 {
            rho_t[t + 1] = even;
            rho_t[t + 2] = odd;
        }
        t += 2;
    }
    let max_t = t;
    if even > 0.0 && max_t + 1 < n {
        rho_t[max_t + 1] = even;
    }
    // Initial monotone sequence: pair sums may not increase.
    let mut t = 1;
    while t + 2 <= max_t {
        let prev = rho_t[t - 1] + rho_t[t];
        if rho_t[t + 1] + rho_t[t + 2] > prev {
            rho_t[t + 1] = prev / 2.0;
            rho_t[t + 2] = prev / 2.0;
        }
        t += 2;
    }
    let total = (m * n) as f64;
    let tail = if max_t + 1 < n { rho_t[max_t + 1] } else { 0.0 };
    let tau = (-1.0 + 2.0 * rho_t[..=max_t.min(n - 1)].iter().sum::<f64>() + tail)
        .max(1.0 / total.log10());
    total / tau
}

/// Linear-interpolation quantile (type 7) of all draws.
fn quantile(chains: &Chains, q: f64) -> f64 {
    let mut flat: Vec<f64> = chains.iter().flatten().copied().collect();
    flat.sort_by(f64::total_cmp);
    let h = (flat.len() - 1) as f64 * q;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(flat.len() - 1);
    flat[lo] + (h - lo as f64) * (flat[hi] - flat[lo])
}

pub fn rhat(chains: &Chains) -> f64 {
    let s = split(chains);
    let bulk = rhat_basic(&rank_normalize(&s));
    let med = quantile(&s, 0.5);
    let folded: Chains = s.iter().map(|c| c.iter().map(|v| (v - med).abs()).collect()).collect();
    let tail = rhat_basic(&rank_normalize(&folded));
    if bulk.is_nan() || tail.is_nan() {
        f64::NAN
    } else {
        bulk.max(tail)
    }
}

pub fn ess_bulk(chains: &Chains) -> f64 {
    ess_basic(&rank_normalize(&split(chains)))
}

pub fn ess_tail(chains: &Chains) -> f64 {
    let s = split(chains);
    let at = |q: f64| {
        let cut = quantile(&s, q);
        let ind: Chains = s
            .iter()
            .map(|c| c.iter().map(|&v| f64::from(u8::from(v <= cut))).collect())
            .collect();
        ess_basic(&ind)
    };
    at(0.05).min(at(0.95))
}

fn to_chains(draws: &PyReadonlyArray2<'_, f64>) -> PyResult<Chains> {
    let view = draws.as_array();
    let (m, n) = view.dim();
    if m < 1 || n < 8 {
        return Err(PyValueError::new_err(format!(
            "need draws of shape (chains, draws) with at least 8 draws, got ({m}, {n})"
        )));
    }
    Ok(view.outer_iter().map(|row| row.to_vec()).collect())
}

/// (R-hat, bulk ESS, tail ESS) of one parameter; `draws` has shape (chains, draws).
#[pyfunction]
pub fn convergence(py: Python<'_>, draws: PyReadonlyArray2<'_, f64>) -> PyResult<(f64, f64, f64)> {
    let chains = to_chains(&draws)?;
    Ok(py.allow_threads(|| (rhat(&chains), ess_bulk(&chains), ess_tail(&chains))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn normal(rng: &mut Rng) -> f64 {
        let u = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        norm_ppf(u.max(1e-300))
    }

    fn ar1(rng: &mut Rng, phi: f64, n: usize, shift: f64) -> Vec<f64> {
        let mut x = 0.0;
        (0..n)
            .map(|_| {
                x = phi * x + (1.0 - phi * phi).sqrt() * normal(rng);
                x + shift
            })
            .collect()
    }

    #[test]
    fn iid_chains_have_rhat_near_one_and_ess_near_total() {
        let mut rng = Rng::new(3);
        let chains: Chains = (0..4).map(|_| ar1(&mut rng, 0.0, 1000, 0.0)).collect();
        let r = rhat(&chains);
        assert!(r < 1.01, "rhat {r}");
        let ess = ess_bulk(&chains);
        assert!((3200.0..4800.0).contains(&ess), "ess {ess}");
    }

    #[test]
    fn ar1_ess_matches_theory() {
        // ESS / N = (1 - phi) / (1 + phi) = 1/3 for phi = 0.5.
        let mut rng = Rng::new(5);
        let chains: Chains = (0..4).map(|_| ar1(&mut rng, 0.5, 5000, 0.0)).collect();
        let ratio = ess_bulk(&chains) / 20000.0;
        assert!((ratio - 1.0 / 3.0).abs() < 0.05, "ratio {ratio}");
    }

    #[test]
    fn shifted_chain_gives_large_rhat() {
        let mut rng = Rng::new(7);
        let mut chains: Chains = (0..3).map(|_| ar1(&mut rng, 0.0, 500, 0.0)).collect();
        chains.push(ar1(&mut rng, 0.0, 500, 3.0));
        assert!(rhat(&chains) > 1.1);
    }
}
