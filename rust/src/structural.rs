//! Posterior of a negative binomial structural time series by the nested Laplace approximation.
//!
//! Model: log mu_t = (F beta)_t + d_t, with a Gaussian prior on the fixed effects beta, a
//! Gaussian random walk d_t = d_{t-1} + sigma z_t (d_0 = 0), and y_t ~ NB2(mu_t, alpha) with
//! alpha = 1 / (phi^2 + 1e-3). The hyperparameters are u = log sigma and v = log phi, each with a
//! half-normal prior. Decision D-1; likelihood decision D-11.
//!
//! Method: Rue, Martino & Chopin (2009), "Approximate Bayesian inference for latent Gaussian
//! models by using integrated nested Laplace approximations", J. R. Stat. Soc. B 71(2):319-392,
//! DOI 10.1111/j.1467-9868.2008.00700.x (grid over the hyperparameters, Gaussian approximation of
//! the latent field at each point; Tierney & Kadane (1986), JASA 81(393):82-86). Draws come from
//! the mixture of those Gaussians and are corrected by importance weights, smoothed by PSIS
//! (see `psis`). The latent field is x = (beta, d); its precision has an arrowhead pattern, see
//! `arrow`.

use crate::arrow::Arrow;
use crate::negbin::{d_eta, logpmf, poisson_d_eta, poisson_logpmf};
use crate::psis::psis;
use crate::rng::Rng;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const ETA_LIMIT: f64 = 30.0;
const PHI_FLOOR: f64 = 1e-3;

/// Model inputs. `f` is the T x nf fixed-effects design, row-major.
pub struct Spec {
    pub y: Vec<f64>,
    pub f: Vec<f64>,
    pub nf: usize,
    pub prior_mean: Vec<f64>,
    pub prior_prec: Vec<f64>,
    /// Scale s of the half-normal prior on sigma.
    pub sigma_scale: f64,
    /// Scale s of the half-normal prior on phi.
    pub phi_scale: f64,
    /// Poisson counts: no dispersion, so the grid has one column and phi has no prior.
    pub poisson: bool,
}

/// Mode of the latent field at fixed hyperparameters, with the factor of its Hessian.
pub struct Mode {
    pub x: Vec<f64>,
    /// Log of the (unnormalized) joint density of y and x at the mode.
    pub f: f64,
    pub arrow: Arrow,
    pub converged: bool,
}

/// Evaluated hyperparameter grid: row i is u = us[i], column j is v = vs[j].
pub struct Grid {
    pub us: Vec<f64>,
    pub vs: Vec<f64>,
    pub log_marg: Vec<f64>,
    pub modes: Vec<Vec<f64>>,
    pub unconverged: usize,
    pub not_pd: usize,
}

/// Numbers the fit check reads. The thresholds are applied in Python.
pub struct Diagnostics {
    pub khat: f64,
    pub ess: f64,
    /// Largest posterior mass on any outer row or column of the grid.
    pub edge_mass: f64,
    pub unconverged: usize,
    pub not_pd: usize,
    pub draw_failures: usize,
    pub grid_points: usize,
}

pub struct Fit {
    /// Equal-weight draws of the latent field, n_draws x (nf + T), row-major.
    pub draws: Vec<f64>,
    /// Matching hyperparameter draws (u, v), n_draws x 2.
    pub hyper: Vec<f64>,
    pub diagnostics: Diagnostics,
}

/// Grid shape: a coarse pass over fixed ranges, then a refined pass centred on the posterior.
pub struct GridConfig {
    pub u_range: (f64, f64),
    pub v_range: (f64, f64),
    pub coarse: (usize, usize),
    pub fine: (usize, usize),
    /// The refined grid spans mean +- `span` standard deviations of the coarse posterior.
    pub span: f64,
    /// Floor on that standard deviation, so a very sharp coarse posterior keeps a usable grid.
    pub min_sd: f64,
}

impl Default for GridConfig {
    fn default() -> Self {
        GridConfig {
            u_range: (-6.0, 0.0),
            v_range: (-4.0, 1.5),
            coarse: (18, 16),
            fine: (21, 21),
            span: 4.0,
            min_sd: 0.15,
        }
    }
}

fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..n).map(|i| lo + (hi - lo) * i as f64 / (n - 1) as f64).collect()
}

fn alpha_of(v: f64) -> f64 {
    1.0 / (v.exp().powi(2) + PHI_FLOOR)
}

/// log of a half-normal(scale) density at x > 0.
fn half_normal_ln(x: f64, scale: f64) -> f64 {
    (2.0 / (scale * (2.0 * std::f64::consts::PI).sqrt())).ln() - x * x / (2.0 * scale * scale)
}

impl Spec {
    pub fn t(&self) -> usize {
        self.y.len()
    }

    pub fn p(&self) -> usize {
        self.nf + self.y.len()
    }

    /// Log prior of (u, v) with the Jacobian of the log transform.
    pub fn log_prior_hyper(&self, u: f64, v: f64) -> f64 {
        let drift = half_normal_ln(u.exp(), self.sigma_scale) + u;
        if self.poisson {
            drift
        } else {
            drift + half_normal_ln(v.exp(), self.phi_scale) + v
        }
    }

    /// 0.5 log det Q for the latent prior precision Q.
    fn half_logdet_q(&self, sigma: f64) -> f64 {
        0.5 * self.prior_prec.iter().map(|p| p.ln()).sum::<f64>() - self.t() as f64 * sigma.ln()
    }

    fn eta(&self, x: &[f64]) -> Vec<f64> {
        let nf = self.nf;
        (0..self.t())
            .map(|t| {
                let fixed: f64 = (0..nf).map(|k| self.f[t * nf + k] * x[k]).sum();
                (fixed + x[nf + t]).clamp(-ETA_LIMIT, ETA_LIMIT)
            })
            .collect()
    }

    fn loglik(&self, eta: &[f64], alpha: f64) -> f64 {
        // A NaN count is a month that was not observed: it adds nothing to the likelihood.
        let seen = self.y.iter().zip(eta).filter(|(y, _)| !y.is_nan());
        if self.poisson {
            return seen.map(|(&y, &e)| poisson_logpmf(y, e)).sum();
        }
        seen.map(|(&y, &e)| logpmf(y, e, alpha)).sum()
    }

    /// r^T Q r for r = x - prior mean (the drift prior mean is 0).
    fn quad_prior(&self, sigma: f64, x: &[f64]) -> f64 {
        let nf = self.nf;
        let fixed: f64 = (0..nf)
            .map(|k| self.prior_prec[k] * (x[k] - self.prior_mean[k]).powi(2))
            .sum();
        let mut prev = 0.0;
        let mut inc2 = 0.0;
        for t in 0..self.t() {
            let d = x[nf + t];
            inc2 += (d - prev).powi(2);
            prev = d;
        }
        fixed + inc2 / (sigma * sigma)
    }

    /// Log joint density of (y, x) up to the terms that do not depend on x.
    fn objective(&self, sigma: f64, alpha: f64, x: &[f64]) -> f64 {
        self.loglik(&self.eta(x), alpha) - 0.5 * self.quad_prior(sigma, x)
    }

    /// Gradient of the objective and the Hessian factor of its negative.
    fn gradient_and_factor(&self, sigma: f64, alpha: f64, x: &[f64]) -> (Vec<f64>, Option<Arrow>) {
        let (nf, t_len) = (self.nf, self.t());
        let eta = self.eta(x);
        let s2 = sigma * sigma;
        let mut g = vec![0.0; nf + t_len];
        let mut w = vec![0.0; t_len];
        for t in 0..t_len {
            let (score, h) = if self.y[t].is_nan() {
                (0.0, 0.0)
            } else if self.poisson {
                poisson_d_eta(self.y[t], eta[t])
            } else {
                d_eta(self.y[t], eta[t], alpha)
            };
            w[t] = -h;
            g[nf + t] = score;
            for k in 0..nf {
                g[k] += self.f[t * nf + k] * score;
            }
        }
        for k in 0..nf {
            g[k] -= self.prior_prec[k] * (x[k] - self.prior_mean[k]);
        }
        // (D^T D d)_t / sigma^2 with d_0 = 0 and a diagonal of 1 in the last row.
        for t in 0..t_len {
            let here = x[nf + t] - if t > 0 { x[nf + t - 1] } else { 0.0 };
            let next = if t + 1 < t_len { x[nf + t + 1] - x[nf + t] } else { 0.0 };
            g[nf + t] -= (here - next) / s2;
        }
        let mut b = vec![0.0; nf * nf];
        let mut c = vec![0.0; nf * t_len];
        for k in 0..nf {
            b[k * nf + k] = self.prior_prec[k];
            for t in 0..t_len {
                c[k * t_len + t] = self.f[t * nf + k] * w[t];
            }
            for j in 0..nf {
                for t in 0..t_len {
                    b[k * nf + j] += self.f[t * nf + k] * w[t] * self.f[t * nf + j];
                }
            }
        }
        let e_diag: Vec<f64> = (0..t_len)
            .map(|t| w[t] + if t + 1 < t_len { 2.0 } else { 1.0 } / s2)
            .collect();
        let e_off = vec![-1.0 / s2; t_len.saturating_sub(1)];
        (g, Arrow::factor(nf, &b, &c, &e_diag, &e_off))
    }

    /// Newton iteration with step halving from `x0`. None when a Hessian is not positive definite.
    pub fn find_mode(&self, sigma: f64, alpha: f64, x0: &[f64]) -> Option<Mode> {
        let mut x = x0.to_vec();
        let mut f = self.objective(sigma, alpha, &x);
        let mut converged = false;
        for _ in 0..60 {
            let (g, arrow) = self.gradient_and_factor(sigma, alpha, &x);
            let step = arrow?.solve(&g);
            // Newton decrement: half the squared length of the step in the Hessian norm.
            let decrement: f64 = 0.5 * g.iter().zip(&step).map(|(a, b)| a * b).sum::<f64>();
            if decrement < 1e-9 {
                converged = true;
                break;
            }
            let mut t = 1.0;
            loop {
                let cand: Vec<f64> = x.iter().zip(&step).map(|(a, s)| a + t * s).collect();
                let fc = self.objective(sigma, alpha, &cand);
                if fc >= f - 1e-12 || t < 1e-6 {
                    x = cand;
                    f = fc;
                    break;
                }
                t /= 2.0;
            }
        }
        let (_, arrow) = self.gradient_and_factor(sigma, alpha, &x);
        Some(Mode { x, f, arrow: arrow?, converged })
    }

    /// Laplace approximation of log p(u, v | y), up to a constant.
    fn log_marginal(&self, u: f64, mode: &Mode) -> f64 {
        self.f_prior_terms(u, mode.f) - 0.5 * mode.arrow.logdet()
    }

    fn f_prior_terms(&self, u: f64, f: f64) -> f64 {
        f + self.half_logdet_q(u.exp())
    }

    /// Evaluate the Laplace marginal on the grid us x vs, warm-starting each Newton run.
    pub fn grid_pass(&self, us: &[f64], vs: &[f64], start: &[f64]) -> Grid {
        let mut log_marg = vec![f64::NEG_INFINITY; us.len() * vs.len()];
        let mut modes = vec![start.to_vec(); us.len() * vs.len()];
        let (mut unconverged, mut not_pd) = (0, 0);
        let mut row_start = start.to_vec();
        for (i, &u) in us.iter().enumerate() {
            let mut x_start = row_start.clone();
            for (j, &v) in vs.iter().enumerate() {
                match self.find_mode(u.exp(), alpha_of(v), &x_start) {
                    Some(m) => {
                        unconverged += usize::from(!m.converged);
                        log_marg[i * vs.len() + j] = self.log_prior_hyper(u, v) + self.log_marginal(u, &m);
                        x_start = m.x.clone();
                        if j == 0 {
                            row_start = m.x.clone();
                        }
                        modes[i * vs.len() + j] = m.x;
                    }
                    None => not_pd += 1,
                }
            }
        }
        Grid { us: us.to_vec(), vs: vs.to_vec(), log_marg, modes, unconverged, not_pd }
    }

    /// Start vector: the prior mean with the intercept at the log of the mean count.
    fn start(&self) -> Vec<f64> {
        let mut x = vec![0.0; self.p()];
        x[..self.nf].copy_from_slice(&self.prior_mean);
        let seen: Vec<f64> = self.y.iter().copied().filter(|v| !v.is_nan()).collect();
        let mean = seen.iter().sum::<f64>() / seen.len().max(1) as f64;
        // Shift the first fixed effect (the intercept column) so exp(eta) starts near the mean.
        x[0] = (mean + 0.1).ln() - (1..self.nf).map(|k| self.f[k] * self.prior_mean[k]).sum::<f64>();
        x
    }

    /// Coarse grid, then a grid centred on the coarse posterior, then `n_draws` importance draws.
    pub fn fit(&self, n_draws: usize, seed: u64, cfg: &GridConfig) -> Fit {
        let start = self.start();
        let coarse = self.grid_pass(
            &linspace(cfg.u_range.0, cfg.u_range.1, cfg.coarse.0),
            &if self.poisson { vec![0.0] } else { linspace(cfg.v_range.0, cfg.v_range.1, cfg.coarse.1) },
            &start,
        );
        let (mu_u, mu_v, sd_u, sd_v) = grid_moments(&coarse);
        let fine = self.grid_pass(
            &linspace(mu_u - cfg.span * sd_u.max(cfg.min_sd), mu_u + cfg.span * sd_u.max(cfg.min_sd), cfg.fine.0),
            &if self.poisson {
                vec![0.0]
            } else {
                linspace(mu_v - cfg.span * sd_v.max(cfg.min_sd), mu_v + cfg.span * sd_v.max(cfg.min_sd), cfg.fine.1)
            },
            &start,
        );
        self.draw(&fine, n_draws, seed, coarse.unconverged + fine.unconverged, coarse.not_pd + fine.not_pd)
    }

    fn draw(&self, grid: &Grid, n_draws: usize, seed: u64, unconverged: usize, not_pd: usize) -> Fit {
        let (nu, nv, p) = (grid.us.len(), grid.vs.len(), self.p());
        let du = grid.us[1] - grid.us[0];
        // A single column (Poisson) has no dispersion to draw: v is a placeholder.
        let dv = if nv > 1 { grid.vs[1] - grid.vs[0] } else { 1.0 };
        let pg = normalized(&grid.log_marg);
        let edge_mass = edge_mass(&pg, nu, nv);
        let cdf: Vec<f64> = pg
            .iter()
            .scan(0.0, |acc, &q| {
                *acc += q;
                Some(*acc)
            })
            .collect();
        let mut rng = Rng::new(seed);
        let mut xs = Vec::with_capacity(n_draws * p);
        let mut hyper = Vec::with_capacity(n_draws * 2);
        let mut log_w = Vec::with_capacity(n_draws);
        let mut draw_failures = 0;
        for _ in 0..n_draws {
            let r = rng.uniform() * cdf[cdf.len() - 1];
            let cell = cdf.partition_point(|&c| c < r).min(pg.len() - 1);
            let (i, j) = (cell / nv, cell % nv);
            let u = grid.us[i] + (rng.uniform() - 0.5) * du;
            let v = if nv > 1 { grid.vs[j] + (rng.uniform() - 0.5) * dv } else { grid.vs[j] };
            let (sigma, alpha) = (u.exp(), alpha_of(v));
            let z: Vec<f64> = (0..p).map(|_| rng.normal()).collect();
            match self.find_mode(sigma, alpha, &grid.modes[cell]) {
                Some(m) if m.converged => {
                    let dev = m.arrow.sample_from_z(&z);
                    let x: Vec<f64> = m.x.iter().zip(&dev).map(|(a, b)| a + b).collect();
                    let target = self.log_prior_hyper(u, v)
                        + self.objective(sigma, alpha, &x)
                        + self.half_logdet_q(sigma);
                    let zz: f64 = z.iter().map(|a| a * a).sum();
                    let log_q = pg[cell].ln() - (du * dv).ln() - 0.5 * zz + 0.5 * m.arrow.logdet();
                    let lw = target - log_q;
                    log_w.push(if lw.is_nan() { f64::NEG_INFINITY } else { lw });
                    xs.extend(x);
                }
                _ => {
                    draw_failures += 1;
                    log_w.push(f64::NEG_INFINITY);
                    xs.extend(&grid.modes[cell]);
                }
            }
            hyper.extend([u, if self.poisson { f64::NAN } else { v }]);
        }
        let (weights, khat, ess) = if log_w.iter().any(|w| w.is_finite()) {
            let r = psis(&log_w);
            (r.weights, r.khat, r.ess)
        } else {
            (vec![1.0 / n_draws as f64; n_draws], f64::INFINITY, 0.0)
        };
        // Multinomial resampling to equal weights.
        let wcdf: Vec<f64> = weights
            .iter()
            .scan(0.0, |acc, &q| {
                *acc += q;
                Some(*acc)
            })
            .collect();
        let mut draws = Vec::with_capacity(n_draws * p);
        let mut hyper_out = Vec::with_capacity(n_draws * 2);
        for _ in 0..n_draws {
            let r = rng.uniform() * wcdf[wcdf.len() - 1];
            let k = wcdf.partition_point(|&c| c < r).min(n_draws - 1);
            draws.extend(&xs[k * p..(k + 1) * p]);
            hyper_out.extend(&hyper[k * 2..k * 2 + 2]);
        }
        Fit {
            draws,
            hyper: hyper_out,
            diagnostics: Diagnostics {
                khat,
                ess,
                edge_mass,
                unconverged,
                not_pd,
                draw_failures,
                grid_points: nu * nv,
            },
        }
    }
}

/// Probabilities proportional to exp(log_marg), summing to 1.
fn normalized(log_marg: &[f64]) -> Vec<f64> {
    let max = log_marg.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut pg: Vec<f64> = log_marg.iter().map(|l| (l - max).exp()).collect();
    let total: f64 = pg.iter().sum();
    for q in pg.iter_mut() {
        *q /= total;
    }
    pg
}

fn edge_mass(pg: &[f64], nu: usize, nv: usize) -> f64 {
    let row = |i: usize| (0..nv).map(|j| pg[i * nv + j]).sum::<f64>();
    let col = |j: usize| (0..nu).map(|i| pg[i * nv + j]).sum::<f64>();
    let rows = row(0).max(row(nu - 1));
    if nv > 1 {
        rows.max(col(0)).max(col(nv - 1))
    } else {
        rows
    }
}

/// Posterior mean and standard deviation of u and v on the grid.
fn grid_moments(grid: &Grid) -> (f64, f64, f64, f64) {
    let nv = grid.vs.len();
    let pg = normalized(&grid.log_marg);
    let (mut mu, mut mv) = (0.0, 0.0);
    for (k, q) in pg.iter().enumerate() {
        mu += q * grid.us[k / nv];
        mv += q * grid.vs[k % nv];
    }
    let (mut su, mut sv) = (0.0, 0.0);
    for (k, q) in pg.iter().enumerate() {
        su += q * (grid.us[k / nv] - mu).powi(2);
        sv += q * (grid.vs[k % nv] - mv).powi(2);
    }
    (mu, mv, su.sqrt(), sv.sqrt())
}

/// Fit the structural series and return (latent draws, hyperparameter draws, diagnostics).
///
/// `design` is T x nf. Draws have shape (n_draws, nf + T): the fixed effects, then the drift.
/// `likelihood` is "negative_binomial" or "poisson". Hyperparameter draws have shape
/// (n_draws, 2): log sigma and log phi (NaN for Poisson). Diagnostics are
/// (k-hat, ESS, edge mass, unconverged grid points, non-PD grid points, failed draws, grid size).
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn fit_structural<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    design: PyReadonlyArray2<'py, f64>,
    prior_mean: PyReadonlyArray1<'py, f64>,
    prior_prec: PyReadonlyArray1<'py, f64>,
    sigma_scale: f64,
    phi_scale: f64,
    likelihood: &str,
    n_draws: usize,
    seed: u64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>, (f64, f64, f64, f64, f64, f64, f64))> {
    let yv = y.as_slice()?.to_vec();
    let dv = design.as_array();
    let (t_len, nf) = dv.dim();
    if t_len != yv.len() || t_len < 2 || nf == 0 {
        return Err(PyValueError::new_err(format!(
            "design is {t_len} x {nf} for {} observations",
            yv.len()
        )));
    }
    if yv.iter().any(|v| v.is_infinite() || *v < 0.0) {
        return Err(PyValueError::new_err("y must be finite (or NaN for not observed) and non-negative"));
    }
    if yv.iter().filter(|v| !v.is_nan()).count() < 2 {
        return Err(PyValueError::new_err("y needs at least 2 observed values"));
    }
    let (pm, pp) = (prior_mean.as_slice()?.to_vec(), prior_prec.as_slice()?.to_vec());
    if pm.len() != nf || pp.len() != nf || pp.iter().any(|v| v.is_nan() || *v <= 0.0) {
        return Err(PyValueError::new_err("priors need one mean and one positive precision per column"));
    }
    if !(sigma_scale > 0.0 && phi_scale > 0.0) || n_draws < 2 {
        return Err(PyValueError::new_err("prior scales must be > 0 and n_draws at least 2"));
    }
    if likelihood != "negative_binomial" && likelihood != "poisson" {
        return Err(PyValueError::new_err("likelihood must be 'negative_binomial' or 'poisson'"));
    }
    let spec = Spec {
        y: yv,
        f: dv.iter().cloned().collect(),
        nf,
        prior_mean: pm,
        prior_prec: pp,
        sigma_scale,
        phi_scale,
        poisson: likelihood == "poisson",
    };
    let fit = py.allow_threads(|| spec.fit(n_draws, seed, &GridConfig::default()));
    let p = nf + t_len;
    let d = &fit.diagnostics;
    let draws = numpy::ndarray::Array2::from_shape_vec((n_draws, p), fit.draws)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let hyper = numpy::ndarray::Array2::from_shape_vec((n_draws, 2), fit.hyper)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let diag = (
        d.khat,
        d.ess,
        d.edge_mass,
        d.unconverged as f64,
        d.not_pd as f64,
        d.draw_failures as f64,
        d.grid_points as f64,
    );
    Ok((draws.into_pyarray_bound(py), hyper.into_pyarray_bound(py), diag))
}

#[cfg(test)]
mod tests {
    use super::*;

    const PERIOD: f64 = 12.0;

    /// Intercept, slope and two harmonics, as the Python side builds them by default.
    fn design(t_len: usize) -> Vec<f64> {
        let mut f = Vec::new();
        for t in 0..t_len {
            let tt = t as f64;
            f.push(1.0);
            f.push((tt - (t_len as f64 - 1.0) / 2.0) / 12.0);
            for k in 1..=2 {
                let a = 2.0 * std::f64::consts::PI * k as f64 * tt / PERIOD;
                f.push(a.cos());
                f.push(a.sin());
            }
        }
        f
    }

    fn spec(y: Vec<f64>) -> Spec {
        let t = y.len();
        Spec {
            y,
            f: design(t),
            nf: 6,
            prior_mean: vec![0.0; 6],
            prior_prec: vec![0.25, 1.0, 1.0, 1.0, 1.0, 1.0],
            sigma_scale: 0.2,
            phi_scale: 1.0,
            poisson: false,
        }
    }

    /// NB counts from the model, trend = level + slope, so truth is known.
    fn simulate(t_len: usize, level: f64, alpha: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let mut rng = Rng::new(seed);
        let mut trend = Vec::new();
        let mut y = Vec::new();
        let mut walk = 0.0;
        for t in 0..t_len {
            walk += 0.04 * rng.normal();
            let slope = 0.05 * (t as f64 - (t_len as f64 - 1.0) / 2.0) / 12.0;
            let tr = level.ln() + slope + walk;
            let season = 0.5 * (2.0 * std::f64::consts::PI * t as f64 / PERIOD).cos();
            let mu = (tr + season).exp();
            // Gamma-Poisson mixture: lambda ~ Gamma(alpha, mu / alpha), y ~ Poisson(lambda).
            let lambda = rng.gamma(alpha) * mu / alpha;
            let mut k = 0.0;
            let (mut prod, limit) = (rng.uniform(), (-lambda).exp());
            while prod > limit {
                k += 1.0;
                prod *= rng.uniform();
            }
            y.push(k);
            trend.push(tr);
        }
        (y, trend)
    }

    #[test]
    fn mode_has_zero_gradient_and_arrow_matches_dense_hessian() {
        let (y, _) = simulate(10, 5.0, 8.0, 1);
        let s = spec(y);
        let (sigma, alpha) = (0.15, 6.0);
        let m = s.find_mode(sigma, alpha, &s.start()).unwrap();
        assert!(m.converged);
        let (g, _) = s.gradient_and_factor(sigma, alpha, &m.x);
        assert!(g.iter().all(|v| v.abs() < 1e-3), "{g:?}");
        // Finite-difference Hessian of the objective, compared through log det.
        let p = s.p();
        let h = 1e-5;
        let mut hess = vec![0.0; p * p];
        for j in 0..p {
            let (mut xp, mut xm) = (m.x.clone(), m.x.clone());
            xp[j] += h;
            xm[j] -= h;
            let (gp, _) = s.gradient_and_factor(sigma, alpha, &xp);
            let (gm, _) = s.gradient_and_factor(sigma, alpha, &xm);
            for i in 0..p {
                hess[i * p + j] = -(gp[i] - gm[i]) / (2.0 * h);
            }
        }
        let l = crate::linalg::cholesky(&hess, p).unwrap();
        let dense: f64 = 2.0 * (0..p).map(|i| l[i * p + i].ln()).sum::<f64>();
        assert!((m.arrow.logdet() - dense).abs() < 1e-5 * dense.abs().max(1.0), "{} {dense}", m.arrow.logdet());
    }

    #[test]
    fn a_month_that_was_not_observed_adds_nothing_to_the_likelihood_and_the_mode_is_still_exact() {
        let (mut y, _) = simulate(14, 5.0, 8.0, 3);
        let full = spec(y.clone());
        y[6] = f64::NAN;
        let s = spec(y.clone());
        let (sigma, alpha) = (0.15, 6.0);
        let eta = s.eta(&s.start());
        let expected: f64 = (0..y.len())
            .filter(|&t| t != 6)
            .map(|t| logpmf(y[t], eta[t], alpha))
            .sum();
        assert!((s.loglik(&eta, alpha) - expected).abs() < 1e-9);
        assert!(s.loglik(&eta, alpha).is_finite() && full.loglik(&eta, alpha).is_finite());
        let m = s.find_mode(sigma, alpha, &s.start()).unwrap();
        assert!(m.converged);
        let (g, _) = s.gradient_and_factor(sigma, alpha, &m.x);
        assert!(g.iter().all(|v| v.abs() < 1e-3), "{g:?}");
    }

    #[test]
    fn fit_recovers_the_trend_on_high_counts() {
        let (y, trend) = simulate(60, 40.0, 20.0, 7);
        let s = spec(y);
        let fit = s.fit(1000, 3, &GridConfig::default());
        let d = &fit.diagnostics;
        assert_eq!((d.unconverged, d.not_pd), (0, 0));
        assert!(d.khat < 0.7, "khat {}", d.khat);
        assert!(d.ess > 300.0, "ess {}", d.ess);
        let p = s.p();
        let nf = s.nf;
        let mut err2 = 0.0;
        for t in 0..60 {
            let mean: f64 = (0..1000)
                .map(|k| fit.draws[k * p] + fit.draws[k * p + 1] * s.f[t * nf + 1] + fit.draws[k * p + nf + t])
                .sum::<f64>()
                / 1000.0;
            err2 += (mean - trend[t]).powi(2);
        }
        assert!((err2 / 60.0).sqrt() < 0.15, "rmse {}", (err2 / 60.0).sqrt());
    }

    #[test]
    fn same_seed_same_draws() {
        let (y, _) = simulate(24, 3.0, 5.0, 2);
        let s = spec(y);
        let a = s.fit(50, 9, &GridConfig::default());
        let b = s.fit(50, 9, &GridConfig::default());
        assert_eq!(a.draws, b.draws);
    }
}
