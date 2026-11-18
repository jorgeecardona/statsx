# Check that a posterior method is correct

Use this to test an inference method: your own sampler, an approximation, or a conjugate update you
wrote. The test finds a method that returns a posterior that is too narrow, too wide or biased,
even when its fits look fine. The module is `statsx.sbc`.

## The idea

If the model is right, this procedure gives a rank that is uniform:

1. Draw a parameter \(\theta^*\) from the prior.
2. Simulate data \(y\) from the model with \(\theta^*\).
3. Run your inference on \(y\) and keep \(L\) independent posterior draws \(\theta_1, \dots, \theta_L\).
4. Record the rank \(r = \#\{\ell : \theta_\ell < \theta^*\}\), an integer from \(0\) to \(L\).

When the inference is correct, \(r\) is uniform on \(\{0, \dots, L\}\) for every prior and model
(Cook, Gelman and Rubin 2006; Talts et al. 2018). Repeat the steps \(N\) times, for example 400, and
test whether the \(N\) ranks are uniform.

## What the test reports

`check_calibration` runs three tests on each parameter and compares each with \(\alpha / (3k)\) for
\(k\) parameters (Bonferroni).

| Test | Rejects when | It means |
|---|---|---|
| chi-square on the rank histogram | the bin counts differ from equal counts | the ranks are not uniform |
| mean rank | the mean rank is not the middle one | the posterior is biased |
| spread of the ranks | ranks pile at both ends, or in the middle | the posterior is too narrow, or too wide |

The two last tests use the exact mean and variance of a uniform rank, so they are valid for any
\(L\). Write \(u_i = (r_i + \tfrac12)/(L + 1) - \tfrac12\). The statistics are

\[
z_{\text{mean}} = \frac{\bar u}{\sqrt{\mathbb{E}[v^2] / N}}, \qquad
z_{\text{spread}} = \frac{\overline{u^2} - \mathbb{E}[v^2]}{\sqrt{\mathbb{V}[v^2] / N}},
\]

where \(v\) is uniform on the same \(L + 1\) points. The chi-square test uses at most 20 bins, and a
bin count that divides \(L + 1\) with at least 5 expected ranks per bin.

## Example

The model is a normal mean with a normal prior. Its posterior has a closed form, so the exact one
must pass, and a posterior with 0.6 of the right spread must fail.

```python
import numpy as np
from statsx.sbc import check_calibration, sbc_ranks

rng = np.random.default_rng(0)
n, n_data, n_draws = 400, 5, 99
mu = rng.normal(size=n)  # step 1: draw from the prior N(0, 1)
y = rng.normal(mu[:, None], 1.0, size=(n, n_data))  # step 2: simulate data
centre = y.sum(axis=1) / (n_data + 1)  # the exact posterior is N(centre, 1 / (n_data + 1))
sd = 1.0 / np.sqrt(n_data + 1)

for label, scale in (("exact", 1.0), ("too narrow", 0.6)):
    draws = rng.normal(centre[:, None], scale * sd, size=(n, n_draws))  # step 3
    check = check_calibration({"mu": sbc_ranks(mu, draws)}, n_draws)  # step 4 and the test
    print(label, "passed:", check.passed, [r.split(" (p")[0] for r in check.reasons])
```

```text
exact passed: True []
too narrow passed: False ['mu: rank histogram is not uniform', 'mu: the posterior is too narrow']
```

## Checked in the tests

- An exact posterior was rejected in 3% of 200 runs of 400 replications (the family-wise level is 5%).
- With 400 replications, a posterior with 0.6, 0.8 or 1.3 times the right spread, or shifted by 0.2
  (0.5 posterior standard deviations), was rejected in all 100 runs. A shift of 0.1 (0.24 posterior
  standard deviations) was rejected in 99.
- A discrete parameter, where the truth equals the draws, stays uniform. The papers assume a
  continuous parameter and do not treat ties. statsx breaks them at random, which keeps the rank
  uniform on \(0, \dots, L\).

## When it fails

- The posterior draws inside one replication must be independent. Thin MCMC draws until their
  autocorrelation is small, or the test rejects a correct sampler.
- It checks the method and the model together, on data the model itself simulates. It says nothing
  about whether the model fits real data.
- The test sees the marginal of each parameter. A method that is wrong only in the dependence
  between parameters can pass. Test a function of the parameters too (their sum, a prediction).
- Few replications give little power. With fewer than about 100, only large errors show.

## References

- Talts, S., Betancourt, M., Simpson, D., Vehtari, A., and Gelman, A. (2018). Validating Bayesian
  inference algorithms with simulation-based calibration. arXiv:1804.06788.
  <https://arxiv.org/abs/1804.06788>
- Cook, S. R., Gelman, A., and Rubin, D. B. (2006). Validation of software for Bayesian models using
  posterior quantiles. *Journal of Computational and Graphical Statistics* 15(3):675-692.
  <https://doi.org/10.1198/106186006X136976>
