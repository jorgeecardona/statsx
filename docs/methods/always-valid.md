# Always-valid inference

Use this when you look at your data while it arrives and want to stop as soon as the answer is
clear, without inflating the error rate. The functions are `statsx.alwaysvalid.confidence_sequence`
and `statsx.alwaysvalid.e_process`.

## The problem

A 95% confidence interval is valid at one sample size fixed in advance. Let \(X_1, X_2, \dots\) have
mean \(\mu\) and let \(S_n = \sum_{i \le n} (X_i - \mu)\). By the law of the iterated logarithm,

\[
\limsup_{n \to \infty} \frac{|S_n|}{\sqrt{2 n \log\log n}} = \sigma \quad \text{almost surely,}
\]

so \(|S_n| / (\sigma\sqrt{n})\) crosses 1.96 sooner or later, even when the claim \(\mu = \mu_0\) is
true. If you stop at the first crossing, the false stop rate is far above 5%. In 4000 simulated
normal paths of 400 observations, a usual test checked after every observation stopped in 47.7%
of the paths.

## What it assumes

Each \(X_i\) is conditionally \(\sigma\)-sub-Gaussian given the past:

\[
\mathbb{E}\!\left[ e^{\lambda (X_i - \mu)} \mid X_1, \dots, X_{i-1} \right] \le e^{\lambda^2 \sigma^2 / 2}
\quad \text{for all } \lambda \in \mathbb{R}.
\]

- Normal data with standard deviation \(\sigma\) satisfy it.
- Data in \([0, 1]\), such as 0/1 outcomes, satisfy it with \(\sigma = 1/2\) (Hoeffding's lemma).
- \(\sigma\) is an input. The package does not estimate it.

## The method

For each \(\lambda\), the process \(\exp(\lambda S_n - \lambda^2 \sigma^2 n / 2)\) is a non-negative
supermartingale with initial value 1. A mixture of supermartingales is a supermartingale. Mix over
\(\lambda \sim \mathcal{N}(0, 1/\rho)\) and integrate:

\[
M_n(\mu) = \sqrt{\frac{\rho}{\rho + n\sigma^2}}
\exp\!\left( \frac{(S_n(\mu))^2}{2(\rho + n\sigma^2)} \right),
\qquad S_n(\mu) = \sum_{i \le n} X_i - n\mu .
\]

\(M_n(\mu)\) is the **e-value** for the claim "the mean is \(\mu\)". Under the claim,
\(\mathbb{E}[M_\tau] \le 1\) for every stopping time \(\tau\).

**Ville's inequality.** For a non-negative supermartingale with \(M_0 = 1\),

\[
\mathbb{P}\!\left( \exists n : M_n \ge 1/\alpha \right) \le \alpha .
\]

So the claim can be rejected as soon as \(M_n(\mu_0) \ge 1/\alpha\), and the chance of ever
rejecting a true claim is at most \(\alpha\), whatever the stopping rule.

**Confidence sequence.** Solving \(M_n(\mu) < 1/\alpha\) for \(\mu\) gives the interval

\[
C_n = \bar X_n \pm \frac{1}{n}\sqrt{(\rho + n\sigma^2)\,
\log\frac{\rho + n\sigma^2}{\rho\,\alpha^2}} ,
\qquad
\mathbb{P}\!\left( \forall n \ge 1 : \mu \in C_n \right) \ge 1 - \alpha .
\]

The two tools agree exactly: \(\mu_0 \notin C_n\) if and only if \(M_n(\mu_0) \ge 1/\alpha\).

**Choice of \(\rho\).** The interval is valid for every \(\rho > 0\). The package picks \(\rho\) so
that the interval is nearly the tightest at a chosen sample size \(n^\ast\) (`n_target`, default
100):

\[
\rho = \frac{n^\ast \sigma^2}{2\log(1/\alpha) + \log\!\left(1 + 2\log(1/\alpha)\right)} .
\]

## The price

The radius is of order \(\sigma\sqrt{\log n / n}\), against \(1.96\,\sigma/\sqrt{n}\) for a
fixed-sample interval. At \(n = 100\) and \(n = 400\), with \(\alpha = 0.05\), the interval is 1.55
and 1.60 times wider than the fixed-sample one. This is the price of validity at every \(n\).

## Example

```python
import numpy as np
from statsx.alwaysvalid import confidence_sequence, e_process

x = np.random.default_rng(7).normal(0.5, 1.0, 300)

cs = confidence_sequence(x, sigma=1.0, alpha=0.05)
for n in (10, 50, 100, 300):
    print(n, round(cs.mean[n - 1], 3), round(cs.lower[n - 1], 3), round(cs.upper[n - 1], 3))
print(cs.first_exclusion(0.0))

e = e_process(x, 0.0, sigma=1.0, alpha=0.05)
print(e.first_rejection())
```

```text
10 0.298 -0.921 1.517
50 0.207 -0.229 0.643
100 0.327 0.024 0.631
300 0.368 0.189 0.547
89
89
```

The interval for \(\mu\) excludes 0 for the first time at \(n = 89\). The e-value for
\(\mu_0 = 0\) reaches \(1/\alpha = 20\) at the same \(n\).

## Checked in the tests

- The confidence sequence misses the true mean at some \(n \le 400\) in 2.8% of 4000 simulated
  paths (at most 5%).
- The expected e-value under the claim is 1 to within \(10^{-6}\), by numerical integration over
  \(S_n\).
- The e-value and the interval agree at every \(n\), for five values of \(\mu_0\).

## What the library checks and does not check

Checked: the observations are a non-empty array of finite numbers, `sigma` is finite and above 0,
`alpha` is in (0, 1), and `n` is at least 1.

Not checked:

- That \(\sigma\) is large enough. If it is too small, the guarantee no longer holds.
- Observations that depend on each other beyond the martingale condition above.
- A mean that changes over time. The interval then covers the mean of a mixture, not a fixed
  \(\mu\).

## References

- Robbins, H. (1970). Statistical methods related to the law of the iterated logarithm. *Annals of
  Mathematical Statistics* 41(5):1397-1409. <https://doi.org/10.1214/aoms/1177696786>
- Howard, S. R., Ramdas, A., McAuliffe, J., and Sekhon, J. (2021). Time-uniform, nonparametric,
  nonasymptotic confidence sequences. *Annals of Statistics* 49(2):1055-1080.
  <https://doi.org/10.1214/20-AOS1991>
- Ville, J. (1939). *Etude critique de la notion de collectif*. Gauthier-Villars.
- Ramdas, A., Grunwald, P., Vovk, V., and Shafer, G. (2023). Game-theoretic statistics and safe
  anytime-valid inference. *Statistical Science* 38(4):576-601.
  <https://doi.org/10.1214/23-STS894>
