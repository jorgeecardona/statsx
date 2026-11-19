# Check MCMC draws and their Monte Carlo error

Use this when a sampler gave you draws of a parameter and you must decide whether to trust them,
and how many digits of a summary are real. The module is `statsx.diagnostics`.

## The data

An array of shape (chains, draws) for each parameter. Use 4 chains or more, and at least 100 draws
per chain. Every chain is split in half, so a chain that drifts shows up as two halves that differ.

## Convergence: R-hat and effective sample size

Split chains give \(M\) chains of \(N\) draws. With \(B\) the variance between the chain means (times
\(N\)) and \(W\) the mean within-chain variance,

\[
\widehat{\operatorname{var}}^{+} = \frac{N - 1}{N} W + \frac{B}{N}, \qquad
\widehat{R} = \sqrt{\widehat{\operatorname{var}}^{+} / W} .
\]

Before the computation the draws are replaced by their normal scores (the pooled rank \(r\) of a draw
among \(S\) draws becomes \(\Phi^{-1}((r - 3/8)/(S + 1/4))\)), so heavy tails cannot hide a problem.
A second \(\widehat{R}\) uses the folded draws \(|x - \operatorname{median}|\) to catch a difference in
spread. The reported value is the larger of the two (Vehtari et al. 2021).

The effective sample size (ESS) is the number of independent draws that carry the same information
as your autocorrelated ones:

\[
\mathrm{ESS} = \frac{MN}{\hat\tau}, \qquad
\hat\tau = -1 + 2 \sum_{t=0}^{T} \hat P_t, \qquad \hat P_t = \hat\rho_{2t} + \hat\rho_{2t+1},
\]

with the pooled autocorrelations \(\hat\rho_t\) summed while the pair sums stay positive and do not
increase (Geyer 1992). `check_convergence` reports bulk ESS (rank-normalized draws) and tail ESS (the
smaller ESS of the 5% and 95% quantile indicators). The default limits are \(\widehat{R} < 1.01\) and
ESS above 400.

## Monte Carlo standard error

The posterior mean you compute from \(S\) draws is not the exact posterior mean. Its error comes from
the draws, and it does not fall below \(\sigma / \sqrt{\mathrm{ESS}}\) for any sampler. Write
\(\sigma\) for the posterior standard deviation.

| Function | Summary | Formula |
|---|---|---|
| `mcse_mean` | posterior mean | \(\sigma / \sqrt{\mathrm{ESS}(x)}\) |
| `mcse_sd` | posterior standard deviation | delta method below |
| `mcse_quantile` | posterior quantile at `prob` | Beta rule below |

For the standard deviation, let \(d = (x - \bar x)^2\). The variance of the sample variance is
\((\overline{d^2} - \bar d^2) / \mathrm{ESS}(d)\), and the delta method gives

\[
\operatorname{se}(s) = \sqrt{\frac{\operatorname{Var}(s^2)}{4 s^2}} .
\]

For the quantile at level \(p\), take the ESS \(n_p\) of the indicator \(\mathbb{1}\{x \le q_p\}\). The
level of the true quantile among your draws is about \(\mathrm{Beta}(n_p p + 1,\, n_p (1 - p) + 1)\). The
standard error is half the distance between the sorted draws at the 16th and 84th percentiles of that
Beta distribution.

All ESS values here are for the draws as they are, split in half, and not rank-normalized.

## Example

Draws of an AR(1) chain with autocorrelation 0.8, which a sampler with sticky steps can produce.
The naive error \(\sigma / \sqrt{S}\) treats them as independent and is too small by a factor of 3.

```python
import numpy as np
from statsx.diagnostics import check_convergence, ess_mean, mcse_mean, mcse_quantile

rng = np.random.default_rng(0)
draws = np.zeros((4, 2000))
state = rng.normal(size=4)
for t in range(2000):
    state = 0.8 * state + np.sqrt(1 - 0.8**2) * rng.normal(size=4)
    draws[:, t] = state

check = check_convergence({"mu": draws})
print("passed:", check.passed)
print("ESS as a share of the draws:", round(ess_mean(draws) / draws.size, 1))
naive = draws.std(ddof=1) / np.sqrt(draws.size)
print("naive error is too small by:", round(mcse_mean(draws) / naive, 1))
print("median error over mean error:", round(mcse_quantile(draws, 0.5) / mcse_mean(draws), 1))
```

```text
passed: True
ESS as a share of the draws: 0.1
naive error is too small by: 3.0
median error over mean error: 1.0
```

## Checked in the tests

Each Monte Carlo standard error was compared with the spread of the summary over 300 repeated runs
of 4 chains with 500 draws (AR(1) draws, so the true value is known):

| Summary | Autocorrelation | Reported (mean of 60 runs) | Spread over 300 runs |
|---|---|---|---|
| mean | 0 | 0.0224 | 0.0224 |
| mean | 0.5 | 0.0385 | 0.0386 |
| mean | 0.8 | 0.0671 | 0.0666 |
| standard deviation | 0.5 | 0.0201 | 0.0212 |
| 5% quantile | 0.5 | 0.0635 | 0.0604 |
| median | 0.5 | 0.0416 | 0.0438 |
| 95% quantile | 0.5 | 0.0633 | 0.0573 |

The error of the mean also matches the theory \(\sqrt{(1 + \phi)/(1 - \phi)/N}\). The ESS of the mean
is a third of the draws for \(\phi = 0.5\). Constant draws give NaN, and a wrong shape is refused.

## What it assumes

- The draws are a matrix of shape (chains, draws), from chains started at different points.
- Each chain is stationary after the draws you pass. Discard the warm-up first.
- The Monte Carlo standard error (MCSE) is for the draws. It is not an error of the model.

## What the library checks and does not check

Checked: the shape (chains, draws), that every draw is finite, and that `prob` is in (0, 1).

Not checked:

- Whether the sampler mixed. A sampler that has not mixed can look fine to every number here. Run 4
  chains from different starting points, and read \(\widehat{R}\) first.
- Whether the ESS estimate is stable. It is noisy when it is small. Below 100 draws per chain, do not
  trust a small MCSE.
- The far tails. The quantile rule uses an indicator ESS, which is rough with few draws.
- How well the model fits. MCSE says nothing about it.

## References

- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., and Burkner, P.-C. (2021). Rank-normalization,
  folding, and localization: an improved \(\widehat{R}\) for assessing convergence of MCMC.
  *Bayesian Analysis* 16(2):667-718. <https://doi.org/10.1214/20-BA1221>
- Geyer, C. J. (1992). Practical Markov chain Monte Carlo. *Statistical Science* 7(4):473-483.
  <https://doi.org/10.1214/ss/1177011137>
- Kenney, J. F., and Keeping, E. S. (1951). *Mathematics of Statistics, Part Two*, 2nd ed., p. 141.
  Van Nostrand.
