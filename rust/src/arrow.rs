//! Cholesky factor of a symmetric positive definite "arrowhead" matrix: a small dense block of
//! `nf` fixed effects next to a tridiagonal block of `n` drift terms.
//!
//! ```text
//!     H = [ B   C ]      B: nf x nf dense, C: nf x n dense, E: n x n tridiagonal
//!         [ C^T E ]
//! ```
//!
//! E is factored as L_E L_E^T (bidiagonal), then the Schur complement S = B - G^T G with
//! G = L_E^-1 C^T is factored densely. The cost is O(n nf^2) instead of O((n + nf)^3).
//! Block elimination: Golub & Van Loan (2013), "Matrix Computations", 4th ed., sections 4.3 and
//! 4.5. Used by the nested Laplace marginal of decision D-1.

use crate::linalg::{cho_solve, cholesky};

/// Lower factor of `H` in the order (drift, fixed): H = L L^T with
/// L = [[L_E, 0], [G^T, L_S]].
pub struct Arrow {
    pub nf: usize,
    pub n: usize,
    /// Diagonal of L_E.
    ld: Vec<f64>,
    /// Sub-diagonal of L_E: `ls[t]` couples row t+1 to row t (length n - 1).
    ls: Vec<f64>,
    /// G = L_E^-1 C^T, row-major n x nf.
    g: Vec<f64>,
    /// Dense lower factor of the Schur complement, nf x nf.
    lschur: Vec<f64>,
}

impl Arrow {
    /// Factor H from its blocks. `b` is nf x nf row-major, `c` is nf x n row-major, `e_diag`
    /// has n entries and `e_off` has n - 1 (the entries next to the diagonal).
    /// Returns None when H is not positive definite.
    pub fn factor(
        nf: usize,
        b: &[f64],
        c: &[f64],
        e_diag: &[f64],
        e_off: &[f64],
    ) -> Option<Arrow> {
        let n = e_diag.len();
        let mut ld = vec![0.0; n];
        let mut ls = vec![0.0; n.saturating_sub(1)];
        for t in 0..n {
            let mut d = e_diag[t];
            if t > 0 {
                ls[t - 1] = e_off[t - 1] / ld[t - 1];
                d -= ls[t - 1] * ls[t - 1];
            }
            if d.is_nan() || d <= 0.0 {
                return None;
            }
            ld[t] = d.sqrt();
        }
        // G column k = L_E^-1 (row k of C), stored row-major n x nf.
        let mut g = vec![0.0; n * nf];
        for k in 0..nf {
            let mut prev = 0.0;
            for t in 0..n {
                let mut v = c[k * n + t];
                if t > 0 {
                    v -= ls[t - 1] * prev;
                }
                prev = v / ld[t];
                g[t * nf + k] = prev;
            }
        }
        let mut s = b.to_vec();
        for i in 0..nf {
            for j in 0..=i {
                let mut acc = 0.0;
                for t in 0..n {
                    acc += g[t * nf + i] * g[t * nf + j];
                }
                s[i * nf + j] -= acc;
                s[j * nf + i] = s[i * nf + j];
            }
        }
        let lschur = if nf == 0 { Vec::new() } else { cholesky(&s, nf)? };
        Some(Arrow { nf, n, ld, ls, g, lschur })
    }

    /// log det H.
    pub fn logdet(&self) -> f64 {
        let a: f64 = self.ld.iter().map(|v| v.ln()).sum();
        let b: f64 = (0..self.nf).map(|i| self.lschur[i * self.nf + i].ln()).sum();
        2.0 * (a + b)
    }

    /// Solve H x = rhs. The vectors are ordered (fixed effects, drift).
    pub fn solve(&self, rhs: &[f64]) -> Vec<f64> {
        let (nf, n) = (self.nf, self.n);
        let (rf, rd) = rhs.split_at(nf);
        // Forward: L_E y_d = rd, then L_S y_f = rf - G^T y_d.
        let mut yd = vec![0.0; n];
        for t in 0..n {
            let mut v = rd[t];
            if t > 0 {
                v -= self.ls[t - 1] * yd[t - 1];
            }
            yd[t] = v / self.ld[t];
        }
        let mut tf = rf.to_vec();
        for t in 0..n {
            for k in 0..nf {
                tf[k] -= self.g[t * nf + k] * yd[t];
            }
        }
        let xf = if nf == 0 { Vec::new() } else { cho_solve(&self.lschur, nf, &tf) };
        // Backward: L_E^T x_d = y_d - G x_f.
        let mut xd = vec![0.0; n];
        for t in (0..n).rev() {
            let mut v = yd[t];
            for k in 0..nf {
                v -= self.g[t * nf + k] * xf[k];
            }
            if t + 1 < n {
                v -= self.ls[t] * xd[t + 1];
            }
            xd[t] = v / self.ld[t];
        }
        let mut out = xf;
        out.extend(xd);
        out
    }

    /// Solve L^T x = z, so that x ~ N(0, H^-1) when z ~ N(0, I). Ordered (fixed, drift) on both
    /// sides; the draw uses the fixed-first order of `solve`.
    pub fn sample_from_z(&self, z: &[f64]) -> Vec<f64> {
        let (nf, n) = (self.nf, self.n);
        let (zf, zd) = z.split_at(nf);
        // L^T = [[L_E^T, G], [0, L_S^T]] in (drift, fixed) order: x_f first.
        let mut xf = zf.to_vec();
        for i in (0..nf).rev() {
            for k in i + 1..nf {
                xf[i] -= self.lschur[k * nf + i] * xf[k];
            }
            xf[i] /= self.lschur[i * nf + i];
        }
        let mut xd = vec![0.0; n];
        for t in (0..n).rev() {
            let mut v = zd[t];
            for k in 0..nf {
                v -= self.g[t * nf + k] * xf[k];
            }
            if t + 1 < n {
                v -= self.ls[t] * xd[t + 1];
            }
            xd[t] = v / self.ld[t];
        }
        let mut out = xf;
        out.extend(xd);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::cholesky as dense_cholesky;

    /// A small H built densely, with a matching arrowhead factor.
    fn example() -> (Vec<f64>, Arrow, usize, usize) {
        let (nf, n) = (2, 5);
        let p = nf + n;
        let mut h = vec![0.0; p * p];
        let b = [3.0, 0.5, 0.5, 2.5];
        let c: Vec<f64> = (0..nf * n).map(|i| 0.1 * ((i % 4) as f64) - 0.12).collect();
        let e_diag = [2.5, 3.0, 2.0, 2.8, 2.2];
        let e_off = [-0.7, 0.4, -0.5, 0.6];
        for i in 0..nf {
            for j in 0..nf {
                h[i * p + j] = b[i * nf + j];
            }
            for t in 0..n {
                h[i * p + nf + t] = c[i * n + t];
                h[(nf + t) * p + i] = c[i * n + t];
            }
        }
        for t in 0..n {
            h[(nf + t) * p + nf + t] = e_diag[t];
            if t + 1 < n {
                h[(nf + t) * p + nf + t + 1] = e_off[t];
                h[(nf + t + 1) * p + nf + t] = e_off[t];
            }
        }
        let arrow = Arrow::factor(nf, &b, &c, &e_diag, &e_off).unwrap();
        (h, arrow, nf, n)
    }

    #[test]
    fn logdet_and_solve_match_dense() {
        let (h, arrow, nf, n) = example();
        let p = nf + n;
        let l = dense_cholesky(&h, p).unwrap();
        let dense_logdet: f64 = 2.0 * (0..p).map(|i| l[i * p + i].ln()).sum::<f64>();
        assert!((arrow.logdet() - dense_logdet).abs() < 1e-12);
        let rhs: Vec<f64> = (0..p).map(|i| (i as f64 + 1.0).sin()).collect();
        let want = cho_solve(&l, p, &rhs);
        let got = arrow.solve(&rhs);
        for i in 0..p {
            assert!((got[i] - want[i]).abs() < 1e-12, "{i}");
        }
    }

    #[test]
    fn sample_has_covariance_h_inverse() {
        // Push the unit vectors through L^-T: the columns form a square root of H^-1.
        let (h, arrow, nf, n) = example();
        let p = nf + n;
        let mut cols = Vec::new();
        for j in 0..p {
            let mut z = vec![0.0; p];
            z[j] = 1.0;
            cols.push(arrow.sample_from_z(&z));
        }
        // (sum_j col_j col_j^T) H should be the identity.
        for i in 0..p {
            for k in 0..p {
                let mut cov_hk = 0.0;
                for m in 0..p {
                    let cov_im: f64 = (0..p).map(|j| cols[j][i] * cols[j][m]).sum();
                    cov_hk += cov_im * h[m * p + k];
                }
                let want = if i == k { 1.0 } else { 0.0 };
                assert!((cov_hk - want).abs() < 1e-12, "{i} {k}");
            }
        }
    }

    #[test]
    fn refuses_indefinite() {
        assert!(Arrow::factor(1, &[0.1], &[1.0, 1.0], &[1.0, 1.0], &[0.0]).is_none());
    }
}
