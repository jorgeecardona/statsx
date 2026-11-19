# Effect of an intervention on a count series

Use this to estimate what an intervention did to a monthly count series, when you have control series
that the intervention did not touch. The module is `statsx.impact`.

## The method

`causal_impact` fits [`fit_trend`](trend.md) on the months before the intervention, with the controls
as regressors. It then forecasts the months after the intervention, using the control values of those
months. That forecast is the counterfactual: what the series would have been without the intervention.
The effect in a month is the observed count minus the counterfactual, as posterior draws.

`cumulative()` sums the effect over all months after the intervention. `average()` gives the effect
per month. `band()` gives a band for every month. `p_positive()` is the share of draws in which the
cumulative effect is above 0. Each summary is an `Estimate` with its interval, the number of months and
the assumption below.

The controls enter as \(\log(1 + x)\) by default, so a control acts on the log of the mean. The
coefficients of the controls have a Gaussian prior. This is not the spike-and-slab prior of the paper.

## What it assumes

- The controls are not changed by the intervention.
- The relation between the controls and the series is the same after the intervention as before it.
- The model of [the trend page](trend.md) holds in the months before the intervention: a season of fixed
  shape, a smooth level, and negative binomial counts.

The first two cannot be checked from the data. They are the assumptions of Brodersen et al. (2015).

## What the library checks and does not check

Checked: the shapes, that at least \(2 \times\) `period` months come before the intervention and at least
1 after it, that no count after the intervention is missing, and that log controls are not negative. The
fit on the months before the intervention must pass its check (`FitCheckError`, see the trend page).

Not checked:

- That the controls are unaffected by the intervention. A control that the intervention moved biases the
  effect, and no number here shows it.
- That the relation between the controls and the series did not change at the intervention. A change in
  it looks like an effect.
- That something else did not happen at the same time. The method sees one change and gives it to the
  intervention.
- Whether the model fits. A passing fit check is a check of the computation (principle P-3).

## Example

The data are simulated. The series has a multiplicative effect of \(e^{0.8}\) from month 48, and three
control series share its hidden factor.

```python
import numpy as np
from statsx.impact import causal_impact

rng = np.random.default_rng(1)
months, start = 60, 48
t = np.arange(months)
factor = np.cumsum(rng.normal(0.0, 0.08, months))
season = 0.4 * np.cos(2 * np.pi * t / 12)
controls = np.column_stack(
    [
        rng.negative_binomial(20.0, 20.0 / (20.0 + np.exp(np.log(40.0) + factor + season)))
        for _ in range(3)
    ]
)
without = np.exp(np.log(30.0) + 0.8 * factor + season)
mean = without * np.exp(0.8 * (t >= start))
counts = rng.negative_binomial(20.0, 20.0 / (20.0 + mean))
truth = float((mean - without)[start:].sum())

impact = causal_impact(counts, controls, start, drift_scale=0.05, seed=0)
total = impact.cumulative(0.95)
print("true cumulative effect:", round(truth, -1))
low, high = round(total.interval.lower, -1), round(total.interval.upper, -1)
print("estimate:", round(total.estimate, -1), "interval:", (low, high))
print("interval holds the truth:", total.interval.contains(truth))
print("P(effect above 0) above 0.95:", impact.p_positive() > 0.95)
```

```text
true cumulative effect: 400.0
estimate: 380.0 interval: (220.0, 500.0)
interval holds the truth: True
P(effect above 0) above 0.95: True
```

## Checked in the tests

- A large effect (\(e^{0.8}\)) is found: the interval of the cumulative effect is above 0, and the
  estimate is within 35% of the true cumulative effect.
- With no effect, the 95% interval excluded 0 in at most 20% of 15 simulated series. The nominal rate is
  5%. The test allows extra for 15 series and for the prior.
- `average()` is `cumulative()` divided by the number of months, and the band ends keep their order.
- The result states its assumption and the number of months.

These tests assert inequalities. They show that the method works on this kind of data, not that your
controls satisfy the assumptions.

## References

- Brodersen, K. H., Gallusser, F., Koehler, J., Remy, N., and Scott, S. L. (2015). Inferring causal
  impact using Bayesian structural time-series models. *Annals of Applied Statistics* 9(1):247-274.
  <https://doi.org/10.1214/14-AOAS788>
- Scott, S. L., and Varian, H. R. (2014). Predicting the present with Bayesian structural time series.
  *International Journal of Mathematical Modelling and Numerical Optimisation* 5(1-2):4-23.
  <https://doi.org/10.1504/IJMMNO.2014.059942>
