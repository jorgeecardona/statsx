# Drift between two samples

Use this to decide whether new data comes from the same distribution as a reference sample. The
module is `statsx.drift`. Each test looks for a different change, so read the last column of the
table before you pick one.

| Test | Data | Statistic | Finds well | Misses |
|---|---|---|---|---|
| `ks_test` | one variable | largest gap between the distribution functions | any change in the centre | a change in the far tails only |
| `mmd_test` | several columns | maximum mean discrepancy | shifts and shape changes, with a good bandwidth | high dimension with few rows |
| `classifier_test` | several columns | held-out AUC | what the classifier can learn | a change the classifier cannot represent |
| `psi` | one variable or bin counts | population stability index | changes in the bin shares | changes inside a bin |

## Kolmogorov-Smirnov

For samples of sizes \(n\) and \(m\) with distribution functions \(F_a\) and \(F_b\),

\[
D = \sup_x \left| F_a(x) - F_b(x) \right| .
\]

Without ties, \(D\) takes the form \(g / (nm)\) for an integer \(g\), and the distribution of \(D\)
under no drift is a count of lattice paths. A path goes from \((0, 0)\) to \((n, m)\), and the step
at the \(k\)-th pooled value goes right if it belongs to \(a\) and up if it belongs to \(b\). The path
stays inside the band \(|im - jn| < g\) exactly when \(D < g/(nm)\). Let \(f(i, j)\) be the share of
the paths to \((i, j)\) that stay inside. Then

\[
f(i, j) = \frac{i\, f(i-1, j) + j\, f(i, j-1)}{i + j}, \qquad
\mathbb{P}(D \ge g/(nm)) = 1 - f(n, m) .
\]

This is Hodges (1958). The package uses it for \(nm \le 10000\) without ties. Otherwise it uses the
Smirnov limit

\[
\mathbb{P}(D \ge d) \approx 2 \sum_{k \ge 1} (-1)^{k-1} e^{-2k^2 \lambda^2}, \qquad
\lambda = d \sqrt{\frac{nm}{n + m}} .
\]

## Maximum mean discrepancy

For a kernel \(k\), the unbiased estimate of the squared distance between the two mean embeddings is

\[
\widehat{\mathrm{MMD}}^2 =
\frac{1}{n(n-1)} \sum_{i \ne j} k(x_i, x_j)
+ \frac{1}{m(m-1)} \sum_{i \ne j} k(y_i, y_j)
- \frac{2}{nm} \sum_{i, j} k(x_i, y_j) .
\]

The kernel is \(k(x, y) = \exp(-\lVert x - y \rVert^2 / (2h^2))\), with \(h\) the median of the pairwise
distances of the pooled rows (Gretton et al. 2012). The estimate is near 0 without drift and can be
negative. The p-value comes from relabelling the pooled rows at random \(B\) times:

\[
p = \frac{1 + \#\{b : \widehat{\mathrm{MMD}}^2_b \ge \widehat{\mathrm{MMD}}^2\}}{1 + B} ,
\]

which is never 0 (Phipson and Smyth 2010). The test holds the \((n+m)^2\) kernel matrix in memory.

## Classifier two-sample test

A logistic regression with an L2 penalty learns to tell sample \(a\) from sample \(b\) on half of each
sample. The statistic is its area under the ROC curve (AUC) on the other half. An AUC of 0.5 means
no drift.

Without drift, the held-out scores of the two samples are exchangeable, so the AUC is the
Mann-Whitney statistic and its p-value is exact for any class sizes (normal approximation with a tie
and continuity correction). A linear classifier finds a shift of the mean. For one variable with
equal spread and a shift of \(\delta\) standard deviations, the best AUC is \(\Phi(\delta / \sqrt 2)\),
0.760 for \(\delta = 1\). Set `quadratic=True` to add squares and products of the columns, so that a
change of spread or correlation can be found as well.

## Population stability index

Cut the reference into \(k\) bins of equal share. With reference shares \(a_i\) and new shares \(b_i\),

\[
\mathrm{PSI} = \sum_{i=1}^{k} (b_i - a_i) \ln \frac{b_i}{a_i} .
\]

This is the symmetrized Kullback-Leibler divergence \(\mathrm{KL}(a \| b) + \mathrm{KL}(b \| a)\) of the
binned distributions. To second order, \((b - a)\ln(b/a) \approx (b - a)^2 / a\), so
\(n_e \cdot \mathrm{PSI}\) behaves as a chi-square statistic, with \(n_e = nm/(n + m)\):

\[
n_e \cdot \mathrm{PSI} \;\approx\; \chi^2_{k-1} \quad \text{without drift.}
\]

The p-value uses this approximation. The usual thresholds of 0.1 and 0.25 are conventions of credit
scoring, not tests. Empty bins are raised to a share of \(10^{-4}\), because the logarithm of 0 is
infinite. `psi_from_shares` takes shares you already binned.

## Example

```python
import numpy as np
from statsx.drift import classifier_test, ks_test, mmd_test, psi

rng = np.random.default_rng(0)
reference = rng.normal(0.0, 1.0, 500)
new = rng.normal(0.3, 1.0, 500)  # the mean moved by 0.3 standard deviations

for result in (
    ks_test(reference, new),
    psi(reference, new),
    mmd_test(reference, new, n_permutations=499),
    classifier_test(reference, new),
):
    print(f"{result.test:36s} {result.statistic:7.4f}  p = {result.p_value:.5f}")
```

```text
Kolmogorov-Smirnov                    0.1400  p = 0.00011
population stability index            0.0817  p = 0.01551
MMD (Gaussian kernel, permutation)    0.0127  p = 0.00200
classifier two-sample test (AUC)      0.5702  p = 0.00664
```

## Checked in the tests

With no drift, a 5% test rejected in 4.5% of 1500 runs (KS, exact branch), 4.9% (KS, limit branch),
5.7% of 300 (MMD), 4.0% of 400 (classifier) and 5.5% of 1500 (PSI).

- The exact KS p-value equals the count over every possible labelling, for three small cases.
- The unbiased MMD averages the closed-form value for two normal distributions.
- The classifier AUC for a shift of one standard deviation is 0.756 against 0.760 in theory.
- A change of spread (standard deviation 2 against 1) has \(p = 0.13\) for the linear classifier and
  \(p < 10^{-80}\) with `quadratic=True`.

## When it fails

- Rows that are not independent (the same user twice, a time series): every p-value is too small.
- Many comparisons (one test per column): adjust with `statsx.multiple.adjust`.
- With a very large \(n\), every test finds a tiny drift. Read the statistic, not only the p-value.

## References

- Smirnov, N. (1948). Table for estimating the goodness of fit of empirical distributions. *Annals of
  Mathematical Statistics* 19(2):279-281. <https://doi.org/10.1214/aoms/1177730256>
- Hodges, J. L. (1958). The significance probability of the Smirnov two-sample test. *Arkiv for
  Matematik* 3(5):469-486. <https://doi.org/10.1007/BF02589501>
- Gretton, A., Borgwardt, K. M., Rasch, M. J., Scholkopf, B., and Smola, A. (2012). A kernel
  two-sample test. *Journal of Machine Learning Research* 13:723-773.
  <https://jmlr.org/papers/v13/gretton12a.html>
- Phipson, B., and Smyth, G. K. (2010). Permutation p-values should never be zero. *Statistical
  Applications in Genetics and Molecular Biology* 9(1):Article 39.
  <https://doi.org/10.2202/1544-6115.1585>
- Lopez-Paz, D., and Oquab, M. (2017). Revisiting classifier two-sample tests. *ICLR*.
  <https://openreview.net/forum?id=SJkXfE5xx>
- Mann, H. B., and Whitney, D. R. (1947). On a test of whether one of two random variables is
  stochastically larger than the other. *Annals of Mathematical Statistics* 18(1):50-60.
  <https://doi.org/10.1214/aoms/1177730491>
- Siddiqi, N. (2006). *Credit Risk Scorecards*. Wiley.
