# Compare two timing runs

Use this when you have a few repeated timings of version A and of version B, and you want the
probability that B is faster, the speedup with an interval, and the probability that the two are
within a margin you accept as equal.

## The model

Timings are multiplicative: noise scales with the time. Work on the log scale. For version
\(v \in \{A, B\}\), the log times are

\[
y_{v,i} \mid \mu_v, \sigma_v^2 \sim \mathcal{N}(\mu_v, \sigma_v^2), \qquad i = 1, \dots, n_v ,
\]

so \(e^{\mu_v}\) is the geometric mean time and \(e^{\mu_A - \mu_B}\) is the speedup of B over A.
The two versions have their own noise \(\sigma_v^2\).

## The posterior

Give \((\mu, \sigma^2)\) a Normal-Inverse-Gamma prior:
\(\sigma^2 \sim \mathrm{IG}(\alpha_0, \beta_0)\) and \(\mu \mid \sigma^2 \sim
\mathcal{N}(m_0, \sigma^2/\kappa_0)\). With \(n\) values, mean \(\bar y\) and
\(M_2 = \sum_i (y_i - \bar y)^2\), the posterior is Normal-Inverse-Gamma with

\[
\kappa_n = \kappa_0 + n, \qquad
m_n = \frac{\kappa_0 m_0 + n\bar y}{\kappa_n}, \qquad
\alpha_n = \alpha_0 + \frac{n}{2}, \qquad
\beta_n = \beta_0 + \frac{M_2}{2} + \frac{\kappa_0\, n\, (\bar y - m_0)^2}{2\kappa_n} .
\]

The marginal posterior of \(\mu\) is a Student t:

\[
\mu \mid y \sim t_{2\alpha_n}\!\left( m_n,\; \frac{\beta_n}{\alpha_n \kappa_n} \right) .
\]

`NormalInverseGamma.update` does this update, and `sample` draws \(\mu\) so the object goes
straight into `compare`. Two independent posteriors give

\[
\mathbb{P}(B \text{ faster}) = \mathbb{P}(\mu_A - \mu_B > 0), \qquad
\mathbb{P}(\text{within } 2\%) = \mathbb{P}\big( |\mu_A - \mu_B| \le \log 1.02 \big) ,
\]

estimated from paired draws of the two posteriors.

## Which prior

**Reference prior** \(p(\mu, \sigma^2) \propto 1/\sigma^2\). Use `NormalInverseGamma.reference`. The
posterior is \((\bar y,\, n,\, (n-1)/2,\, M_2/2)\), so the marginal of \(\mu\) is exactly the
classical \(t_{n-1}\) interval, \(\bar y \pm t_{n-1,\,0.975}\, s / \sqrt{n}\). At \(n = 5\), its 95%
interval covered the true mean in 95% of simulated samples for every noise level tried.

**Informative noise prior.** If you know the noise level from earlier runs, say \(\sigma \approx
2\%\), you may put it in the prior: `NormalInverseGamma(0.0, 1e-6, 1.0, 0.02**2)`. It narrows the
interval at \(n = 5\), but only if the true noise is near the prior. Coverage of the 95% interval
at \(n = 5\), by simulation:

| True noise sd | 0.01 | 0.02 | 0.03 | 0.04 | 0.06 |
|---|---|---|---|---|---|
| Coverage | 99.8% | 95.3% | 90.9% | 88.1% | 87.1% |

A prior that is too confident about low noise gives intervals that are too narrow when the real
noise is higher. Use the reference prior unless you can defend the number.

## Example

```python
import math
import numpy as np
from statsx.conjugate import NormalInverseGamma
from statsx.posterior import compare

a = [13.71, 13.58, 14.39, 13.80, 13.66]  # version A, decode ms per token
b = [13.21, 13.30, 13.12, 13.45, 13.26]  # version B

pa, pb = (NormalInverseGamma.reference(np.log(t)) for t in (a, b))
c = compare(pa, pb, margin=math.log(1.02))  # 2% counts as equal
print("P(B faster)  ", round(c.p_a_greater, 3))
print("P(within 2%) ", round(c.p_equivalent, 3))
low, high = math.exp(c.difference.lower), math.exp(c.difference.upper)
print("speedup of B  %.3f to %.3f" % (low, high))
```

```text
P(B faster)   0.99
P(within 2%)  0.063
speedup of B  1.010 to 1.075
```

The speedup is \(e^{\mu_A - \mu_B}\), so a value above 1 means B is faster. The interval is the 95%
credible interval of the ratio.

## How many runs do you need

Plan before you measure. For two groups of \(n\) timings with within-group noise \(\sigma\) (on the
log scale), the two-sample t statistic under a true log-difference \(\Delta\) is noncentral t with
\(2(n-1)\) degrees of freedom and noncentrality \(\Delta / (\sigma\sqrt{2/n})\). The power is its
tail beyond the critical value, and `minimum_detectable_effect_two_sample` inverts it.

```python
import math
from statsx.design import (
    minimum_detectable_effect_two_sample,
    sample_size_equivalence_two_sample,
    sample_size_two_sample,
)

sigma = 0.02  # noise sd of log time, about 2%
for n in (5, 10, 20):
    d = minimum_detectable_effect_two_sample(sigma, n)  # 80% power, alpha 0.05
    print(n, "per version finds a speedup of", round(100 * (math.exp(d.effect) - 1), 1), "%")
print("n for a 2% difference:", sample_size_two_sample(math.log(1.02), sigma).n)
print("n to show equivalence within 2%:", sample_size_equivalence_two_sample(sigma, math.log(1.02)).n)
```

```text
5 per version finds a speedup of 4.1 %
10 per version finds a speedup of 2.7 %
20 per version finds a speedup of 1.8 %
n for a 2% difference: 18
n to show equivalence within 2%: 19
```

The noise \(\sigma\) is a guess. Take it from earlier runs of the same setup, and plan with a
larger value than you expect.

## What this does not tell you

- The unit of replication is a **run**. If the 5 timings of a version come from one process, they
  share the run's start-up state, clock speed and cache, so they are not independent. The interval
  then describes that one run, not the version. Restart the process for each timing, or model the
  run effect.
- The model assumes timings of one version are exchangeable. A drift over time (thermal
  throttling, other jobs) breaks it. Interleave A and B (ABAB...) so a drift hits both.
- With \(n = 5\) the interval is wide whatever the method. Check the width before you decide.

## References

- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., and Rubin, D. B. (2013).
  *Bayesian Data Analysis*, 3rd ed., sections 3.2 and 3.3. CRC Press.
  <https://doi.org/10.1201/b16018>
- Murphy, K. P. (2007). *Conjugate Bayesian analysis of the Gaussian distribution*. Technical
  report, University of British Columbia.
