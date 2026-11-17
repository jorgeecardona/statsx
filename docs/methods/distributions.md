# Distributions and goodness of fit

Use this to fit a distribution from a few summary numbers, and to check the fit when you have
very many rows. The modules are `statsx.distributions` and `statsx.goodness`.

## Fit from sufficient statistics

A statistic \(T(x)\) is sufficient when the likelihood depends on the rows only through \(T(x)\).
You can fit from \(T\) when the rows are gone or spread over many machines. Two shards merge into
the statistics of the rows together.

### Normal and lognormal

For \(x_1, \dots, x_n\) the sufficient statistics are the count \(n\), the mean \(\bar x\) and the
sum of squared deviations \(M_2 = \sum_i (x_i - \bar x)^2\). Two shards \(A\) and \(B\) merge by
the rule of Chan, Golub and LeVeque (1983), with \(\delta = \bar x_B - \bar x_A\):

\[
n = n_A + n_B, \qquad
\bar x = \bar x_A + \delta\,\frac{n_B}{n}, \qquad
M_2 = M_{2,A} + M_{2,B} + \delta^2\,\frac{n_A n_B}{n} .
\]

The maximum likelihood fit is

\[
\hat\mu = \bar x, \qquad \hat\sigma^2 = \frac{M_2}{n} .
\]

With `bias_corrected=True` the variance is \(M_2 / (n - 1)\), which is unbiased. For a lognormal
distribution, \(\log X\) is normal: use the statistics of the logarithms (`log_moments`).

### Shifted exponential

The model is \(X = \theta + E\), with \(E\) exponential with rate \(\lambda\), so that
\(f(x) = \lambda e^{-\lambda (x - \theta)}\) for \(x \ge \theta\). The sufficient statistics are the
count, the minimum \(X_{(1)}\) and the mean. Let \(T = \sum_i (X_i - X_{(1)}) = n(\bar x - X_{(1)})\).

The likelihood \(\lambda^n e^{-\lambda \sum_i (x_i - \theta)}\) increases in \(\theta\) up to
\(X_{(1)}\), so the maximum likelihood fit is

\[
\hat\theta = X_{(1)}, \qquad \hat\lambda = \frac{n}{T} .
\]

By the Renyi representation of exponential spacings, \(X_{(1)} - \theta \sim
\mathrm{Exp}(n\lambda)\), and \(T \sim \mathrm{Gamma}(n - 1, \lambda)\) independently of
\(X_{(1)}\). Hence

\[
\mathbb{E}[X_{(1)}] = \theta + \frac{1}{n\lambda}, \qquad
\mathbb{E}[T] = \frac{n - 1}{\lambda}, \qquad
\mathbb{E}\!\left[\frac{1}{T}\right] = \frac{\lambda}{n - 2} \quad (n \ge 3) .
\]

The maximum likelihood shift is biased upwards by \(1 / (n\lambda)\). The bias-corrected fit is

\[
\tilde\theta = X_{(1)} - \frac{T}{n(n - 1)}, \qquad \tilde\lambda = \frac{n - 2}{T} ,
\]

which is unbiased for \(\theta\) and \(\lambda\). It needs \(n \ge 3\).

## Goodness of fit at an effective sample size

With a very large \(n\), a chi-square test rejects almost any fit. The result is a true statement
("the fit is not exact") that you already knew. `goodness_of_fit` and `chi_square_test` report the
test, two distances that do not grow with \(n\), and the sample size at which the test would reject.

### The test

Put the rows in \(k\) categories with observed shares \(\hat p_i\) and expected shares \(\pi_i\). For
a stated effective sample size \(n_e \le n\), the statistic is

\[
X^2 = n_e \sum_{i=1}^{k} \frac{(\hat p_i - \pi_i)^2}{\pi_i} .
\]

At \(n_e = n\) this is Pearson's statistic. Under the fit it is approximately
\(\chi^2_{k - 1 - m}\), where \(m\) is the number of parameters estimated from the same rows. The
result states both \(n\) and \(n_e\). Use \(n_e < n\) when rows are not independent
(\(n_e = n / \text{design effect}\)), or to ask whether the misfit would show at the sample size you
care about.

For continuous data the bins have equal probability under the fit, \(\pi_i = 1/k\), and the number
of bins follows Mann and Wald (1942):

\[
k = \left\lceil 4 \left( \frac{2 (n_e - 1)^2}{z_{1-\alpha}^2} \right)^{1/5} \right\rceil ,
\]

reduced so that every bin expects at least 5 rows.

### The distances

**Total variation distance.** \(\mathrm{TVD} = \tfrac12 \sum_i |\hat p_i - \pi_i|\). It is the
largest difference in the probability of any event. For continuous data it is measured on the bins,
so it is a lower bound of the true distance, plus sampling noise of about \(0.4\sqrt{k/n}\).

**Wasserstein-1 distance.** On the line, with \(F_n\) the empirical distribution function and \(F\)
the fit,

\[
W_1 = \int_{-\infty}^{\infty} |F_n(x) - F(x)|\,dx
    = \int_0^1 \left| F_n^{-1}(u) - F^{-1}(u) \right| du
\]

(Vallender 1973). It has the units of the data. The package computes it exactly, with no bins. On
the cell \(u \in (a, b] = ((i-1)/n,\, i/n]\) the empirical quantile is the constant
\(c = x_{(i)}\). With \(M(u) = \int_0^u F^{-1}(s)\,ds\) and \(u^\ast = \mathrm{clip}(F(c), a, b)\),

\[
\int_a^b |c - F^{-1}(u)|\,du
= c(u^\ast - a) - \big(M(u^\ast) - M(a)\big) + \big(M(b) - M(u^\ast)\big) - c(b - u^\ast) .
\]

\(M\) has a closed form for each fit. With \(\varphi\) and \(\Phi\) the standard normal density and
distribution function:

| Fit | \(M(u)\) |
|---|---|
| Normal \((\mu, \sigma)\) | \(\mu u - \sigma\,\varphi(\Phi^{-1}(u))\) |
| Lognormal \((\mu, \sigma)\) | \(e^{\mu + \sigma^2/2}\,\Phi(\Phi^{-1}(u) - \sigma)\) |
| Shifted exponential \((\theta, \lambda)\) | \(\theta u + \lambda^{-1}\big[u + (1 - u)\ln(1 - u)\big]\) |

### The sample size that rejects

Let \(\hat\phi = \sum_i (\hat p_i - \pi_i)^2 / \pi_i\). Because \(X^2 = n_e \hat\phi\), the same shares
are rejected at level \(\alpha\) when \(n_e \ge \chi^2_{k-1-m,\,1-\alpha} / \hat\phi\). The result
reports this value as `n_to_reject`. A value near your real sample size means the misfit is large. A
value far above it means the misfit is too small to matter at that size. It includes sampling noise
in \(\hat\phi\) of about \(k / n\), so for data that come exactly from the fit it is about \(n\).

## Example

```python
import numpy as np
from statsx.distributions import Moments, fit_normal
from statsx.goodness import goodness_of_fit

rng = np.random.default_rng(0)
shards = [rng.normal(3.0, 2.0, n) for n in (10, 400, 37)]

merged = Moments.from_values(shards[0])
for shard in shards[1:]:
    merged = merged.merge(Moments.from_values(shard))
fit = fit_normal(merged)
print(merged.n, round(fit.mean, 3), round(fit.sd, 3))

# 200 000 rows, mean shifted by 3% of a standard deviation, tested against the unshifted fit.
x = np.random.default_rng(1).normal(0.03, 1.0, 200_000)
base = fit_normal(Moments(1000, 0.0, 1000.0))
for n_effective in (None, 1000):
    r = goodness_of_fit(x, base, n_effective=n_effective)
    print(r.n_effective, r.n_bins, round(r.p_value, 6), round(r.tvd, 4),
          round(r.wasserstein, 4), round(r.n_to_reject))
```

```text
447 2.976 2.019
200000.0 497 1e-05 0.0225 0.0276 171001
1000.0 60 1.0 0.013 0.0276 76907
```

The first line is the pooled count, mean and standard deviation of three shards. The next two lines
are the same data tested at all 200 000 rows and at an effective size of 1000. The Wasserstein
distance is the same in both, close to the true shift of 0.03 (the difference is sampling noise).

## Checked in the tests

- A fit from statistics equals the fit from the rows to \(10^{-12}\), and a merge of three shards
  equals the statistics of the pooled rows to \(10^{-12}\).
- For \(\theta = 10\), \(\lambda = 0.5\) and \(n = 8\), over 40000 samples the maximum likelihood
  shift averages 10.25 (the bias \(1 / (n\lambda)\)), the corrected shift 10.00 and the corrected
  rate 0.500.
- The regularized incomplete gamma function behind the p-value matches mpmath to \(10^{-12}\) at
  nine points, including an upper tail of \(1.6 \times 10^{-23}\).
- The die example (13.4 on 5 degrees of freedom) gives \(p = 0.0199052203347744\).
- \(W_1\) matches the numeric area between the two distribution functions to \(2 \times 10^{-5}\).
- For 300 lognormal samples of 1000 rows drawn from the fit, 6.3% had \(p < 0.05\).

## When it fails

- Rows that are not independent: set `n_effective`.
- A fit estimated from the same rows: set `n_params`. The p-value is then approximate.
- Fewer than 25 rows, or fewer than 5 expected per bin.

## References

- Chan, T. F., Golub, G. H., and LeVeque, R. J. (1983). Algorithms for computing the sample
  variance: analysis and recommendations. *The American Statistician* 37(3):242-247.
  <https://doi.org/10.1080/00031305.1983.10483115>
- Casella, G., and Berger, R. L. (2002). *Statistical Inference*, 2nd ed., chapter 7. Duxbury.
- Lehmann, E. L., and Casella, G. (1998). *Theory of Point Estimation*, 2nd ed., chapter 2.
  Springer. <https://doi.org/10.1007/b98854>
- Pearson, K. (1900). On the criterion that a given system of deviations from the probable in the
  case of a correlated system of variables is such that it can be reasonably supposed to have
  arisen from random sampling. *Philosophical Magazine* 50(302):157-175.
  <https://doi.org/10.1080/14786440009463897>
- Mann, H. B., and Wald, A. (1942). On the choice of the number of class intervals in the
  application of the chi square test. *Annals of Mathematical Statistics* 13(3):306-317.
  <https://doi.org/10.1214/aoms/1177731569>
- Vallender, S. S. (1973). Calculation of the Wasserstein distance between probability
  distributions on the line. *Theory of Probability and its Applications* 18(4):784-786.
  <https://doi.org/10.1137/1118101>
- Gautschi, W. (1979). A computational procedure for incomplete gamma functions. *ACM Transactions
  on Mathematical Software* 5(4):466-481. <https://doi.org/10.1145/355853.355863>
