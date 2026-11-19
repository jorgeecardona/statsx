# Priors, conjugate updates and posterior summaries

Use this to turn a judgement or a count into a distribution, update it with data in closed form, and
read the result. The modules are `statsx.prior`, `statsx.conjugate` and `statsx.posterior`. For a model
with no closed form, see [Check MCMC draws](convergence.md).

## A prior from an expert

An expert gives a lower and an upper bound and says how sure they are that the true value lies between
them. The default is 90%. `lognormal_from_interval(low, high, level)` returns the lognormal distribution
whose central interval at that level is exactly those bounds: the log of the value is normal, its median
is the middle of the two logs, and the standard deviation follows from the normal quantile
(O'Hagan et al. 2006).

The expert must be calibrated: the interval holds the value as often as they claim. The function cannot
check that.

## Conjugate updates

Each prior is a frozen dataclass, and each update returns a new object of the same family. The
arithmetic runs in the Rust core (Gelman et al. 2013).

| Update | Prior | Data | Posterior |
|---|---|---|---|
| `beta_binomial` | `Beta(alpha, beta)` | successes out of trials | `Beta` |
| `gamma_poisson` | `Gamma(shape, rate)` | a count over an exposure | `Gamma` |
| `normal_normal` | `Normal(mean, sd)` | the mean of `n` values with known `sigma` | `Normal` |
| `dirichlet_multinomial` | `Dirichlet` | counts per category | `Dirichlet` |
| `NormalInverseGamma.update` | `NormalInverseGamma` | values of a normal sample, mean and variance both unknown | `NormalInverseGamma` |

Each prior has `sample(n, seed=)`, so its draws are reproducible. `NormalInverseGamma.reference(values)` is the posterior under the
reference prior `1 / sigma2`. Its interval for the mean is the classical t interval, and it keeps its coverage at
small `n` (Gelman et al. 2013, section 3.2).

## Summaries of posterior draws

The functions work on any sample of draws of one quantity.

- `quantiles(draws, probs)`: the type 7 quantiles, by linear interpolation between order statistics
  (Hyndman and Fan 1996).
- `credible_interval(draws, level)`: the equal-tailed interval.
- `hdi(draws, level)`: the shortest interval that holds the stated share of the sorted draws
  (Chen and Shao 1999).
- `compare(a, b, margin=)`: for two independent posteriors, \(P(A > B)\), \(P(B > A)\), the probability that
  the difference is inside the margin of practical equivalence, and an interval for \(A - B\). It also
  gives the Monte Carlo error of the three probabilities. The margin is the `EquivalenceMargin` of
  [the tests page](tests.md).

## What it assumes

- The prior is a statement about what you knew before the data. A narrow prior pulls the result toward
  it. Look at the prior as well as the posterior.
- Conjugate updates assume the model of the family: Bernoulli trials for the Beta, Poisson counts for the
  Gamma, a normal with known variance for the Normal. The data must follow it.
- `compare` assumes the two posteriors are independent.
- Posterior summaries are only as good as the draws. They are Monte Carlo estimates, with an error.

## What the library checks and does not check

Checked: priors have positive parameters, successes are not above the trials, Dirichlet counts match the
number of categories, expert bounds are finite with 0 < low < high, levels and probabilities are in range,
draws are finite with at least 2 values, and a reference posterior has at least 2 values that are not all
equal. `compare` reports its Monte Carlo error.

Not checked:

- That an expert is calibrated, or that two experts would give the same interval.
- That the data follow the conjugate model. A Poisson update on overdispersed counts gives an interval
  that is too narrow.
- That two posteriors are independent. Two arms that share data are not.

## Example

An expert says the cost of a run lies between 2 and 8 with 90% confidence. Two systems are then
compared: A has 7 successes in 40 trials, B has 12 in 40, both with a uniform Beta(1, 1) prior.

```python
from statsx.conjugate import Beta, Gamma, Normal, beta_binomial, gamma_poisson, normal_normal
from statsx.posterior import compare, credible_interval, hdi
from statsx.prior import lognormal_from_interval

cost = lognormal_from_interval(2.0, 8.0, 0.9)
print("median cost:", round(cost.median, 2), "mean cost:", round(cost.mean, 2))

a = beta_binomial(Beta(1.0, 1.0), 7, 40)
b = beta_binomial(Beta(1.0, 1.0), 12, 40)
print("A:", a, "mean", round(a.mean, 3))
draws = a.sample(20000, seed=1)
print("90% credible interval:", [round(x, 2) for x in (credible_interval(draws, 0.9).lower, credible_interval(draws, 0.9).upper)])
print("90% HDI:", [round(x, 2) for x in (hdi(draws, 0.9).lower, hdi(draws, 0.9).upper)])

result = compare(a, b, margin=0.05)
print("P(A > B):", round(result.p_a_greater, 2), "P(B > A):", round(result.p_b_greater, 2))
print("P(equivalent within 0.05):", round(result.p_equivalent, 2))

print(gamma_poisson(Gamma(2.0, 1.0), 30, 10.0))
print(normal_normal(Normal(0.0, 10.0), 2.0, 25, 1.4))
```

```text
median cost: 4.0 mean cost: 4.37
A: Beta(alpha=8.0, beta=34.0) mean 0.19
90% credible interval: [0.1, 0.3]
90% HDI: [0.09, 0.29]
P(A > B): 0.1 P(B > A): 0.9
P(equivalent within 0.05): 0.19
Gamma(shape=32.0, rate=11.0)
Normal(mean=1.3977635782747604, sd=0.39968038348871576)
```

The HDI is shorter than the equal-tailed interval because the posterior is skewed. B is more likely to
be better than A (0.9), and the probability that they differ by less than 0.05 is 0.19.

## Checked in the tests

- Each conjugate update returns the closed-form posterior as a named object, and the posterior moments are
  right. More successes than trials, non-positive priors and mismatched Dirichlet lengths are refused.
- A 90% expert interval gives the matching lognormal prior, and a lower level gives a wider one.
- Quantiles match linear interpolation, the credible interval has equal tails, and the HDI is the shortest
  interval with the stated mass (and equals the equal-tailed one for a symmetric sample).
- `compare` matches the exact value of `P(A > B)` for two Betas, and the exact probabilities for two normals.
  A wider margin gives a higher probability of equivalence, and the same seed gives the same comparison.
- The reference posterior gives the classical t interval, and covers at the stated rate at 5 values.

## References

- O'Hagan, A., Buck, C. E., Daneshkhah, A., Eiser, J. R., Garthwaite, P. H., Jenkinson, D. J., Oakley,
  J. E., and Rakow, T. (2006). *Uncertain Judgements: Eliciting Experts' Probabilities*. Wiley.
  <https://doi.org/10.1002/0470033312>
- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., and Rubin, D. B. (2013). *Bayesian
  Data Analysis*, 3rd ed. CRC Press. <https://doi.org/10.1201/b16018>
- Murphy, K. P. (2007). Conjugate Bayesian analysis of the Gaussian distribution. Technical report,
  University of British Columbia.
- Hyndman, R. J., and Fan, Y. (1996). Sample quantiles in statistical packages. *The American
  Statistician* 50(4):361-365. <https://doi.org/10.1080/00031305.1996.10473566>
- Chen, M.-H., and Shao, Q.-M. (1999). Monte Carlo estimation of Bayesian credible and HPD intervals.
  *Journal of Computational and Graphical Statistics* 8(1):69-92.
  <https://doi.org/10.1080/10618600.1999.10474802>
