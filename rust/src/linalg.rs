//! Small dense linear algebra for symmetric positive definite matrices (row-major, p x p).
//!
//! Cholesky factorization as in Golub & Van Loan (2013), "Matrix Computations", 4th ed.,
//! Johns Hopkins University Press, ISBN 978-1-4214-0794-4, section 4.2. Decision D-9.

/// Lower Cholesky factor L with A = L L^T, or None when A is not positive definite.
pub fn cholesky(a: &[f64], p: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; p * p];
    for j in 0..p {
        let mut d = a[j * p + j];
        for k in 0..j {
            d -= l[j * p + k] * l[j * p + k];
        }
        if d.is_nan() || d.is_infinite() || d <= 0.0 {
            return None;
        }
        let djj = d.sqrt();
        l[j * p + j] = djj;
        for i in j + 1..p {
            let mut s = a[i * p + j];
            for k in 0..j {
                s -= l[i * p + k] * l[j * p + k];
            }
            l[i * p + j] = s / djj;
        }
    }
    Some(l)
}

/// Solve (L L^T) x = b given the lower factor L.
pub fn cho_solve(l: &[f64], p: usize, b: &[f64]) -> Vec<f64> {
    let mut y = b.to_vec();
    for i in 0..p {
        for k in 0..i {
            y[i] -= l[i * p + k] * y[k];
        }
        y[i] /= l[i * p + i];
    }
    for i in (0..p).rev() {
        for k in i + 1..p {
            y[i] -= l[k * p + i] * y[k];
        }
        y[i] /= l[i * p + i];
    }
    y
}

/// Inverse of A from its lower factor, column by column.
pub fn cho_inverse(l: &[f64], p: usize) -> Vec<f64> {
    let mut inv = vec![0.0; p * p];
    for j in 0..p {
        let mut e = vec![0.0; p];
        e[j] = 1.0;
        let col = cho_solve(l, p, &e);
        for i in 0..p {
            inv[i * p + j] = col[i];
        }
    }
    inv
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_a_known_system() {
        let a = [4.0, 2.0, 2.0, 3.0];
        let l = cholesky(&a, 2).unwrap();
        let x = cho_solve(&l, 2, &[2.0, 1.0]);
        // 4x + 2y = 2, 2x + 3y = 1  ->  x = 0.5, y = 0
        assert!((x[0] - 0.5).abs() < 1e-14 && x[1].abs() < 1e-14);
        let inv = cho_inverse(&l, 2);
        let det = 8.0;
        assert!((inv[0] - 3.0 / det).abs() < 1e-14 && (inv[1] + 2.0 / det).abs() < 1e-14);
    }

    #[test]
    fn refuses_indefinite() {
        assert!(cholesky(&[1.0, 2.0, 2.0, 1.0], 2).is_none());
    }
}
