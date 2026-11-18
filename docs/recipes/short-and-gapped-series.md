# Fit a short series, or a series with holes

Use this when your monthly counts have less than two seasons of history, or when some months were
not observed. The function is `statsx.trend.fit_trend`.

## The model

The log mean of month \(t\) is a level, a slope, a Fourier season, and a random-walk drift:

\[
\log \mu_t = \beta_0 + \beta_1 \tilde t + \sum_{k=1}^{H} \Big[ a_k \cos\frac{2\pi k t}{P}
+ b_k \sin\frac{2\pi k t}{P} \Big] + d_t, \qquad d_t = d_{t-1} + \eta_t, \quad \eta_t \sim \mathcal{N}(0, \sigma^2),
\]

with \(P\) the period (12 for months) and \(H\) the number of harmonics. The counts are negative
binomial (or Poisson) given \(\mu_t\).

## Months that were not observed

Let \(O\) be the set of observed months. The likelihood is a sum over \(O\) only:

\[
\log p(y \mid \theta) = \sum_{t \in O} \log \mathrm{NB}(y_t \mid \mu_t, \phi) .
\]

The drift \(d_t\) and the season still run over every month, so the phase of the season and the
elapsed time stay right across a hole. Two ways to say which months are missing:

- Put `NaN` in `counts` at the month that was not observed.
- Pass the month of each count as `index`, for counts that come from a grouped query. The series is
  laid out from the first to the last index, with NaN where a month is absent.

A month that you observed with no events is a 0, not a NaN. Do not fill a missing month with a
value: the posterior then treats it as seen, and its band is too narrow. Dropping the month is
worse: every later month moves one place early, and the season is fitted at the wrong phase.

## A short series

Each harmonic adds two parameters. With few months, the model uses at most one parameter per two
observations: the level, the slope and \(2H\) terms use at most \(n/2\), so

\[
H \le \left\lfloor \frac{n}{4} \right\rfloor - 1 \qquad (P \le n < 2P),
\]

every harmonic up to \(P/2\) from \(n = 2P\) observed months, and none below one season
(\(n < P\)): pass `harmonics=0` for a fit with a level, a slope and the drift only. This limit is a
convention chosen here, not a theorem. The fit check still decides whether the fit is accepted.
`supported_harmonics(n, period)` returns the limit for a length.

## Example

```python
import numpy as np
from statsx.trend import fit_trend, supported_harmonics

rng = np.random.default_rng(7)
t = np.arange(18, dtype=float)
counts = rng.poisson(np.exp(4.0 + 0.012 * t + 0.25 * np.sin(2 * np.pi * t / 12.0)))

print([supported_harmonics(n, 12) for n in (8, 12, 18, 23, 24)])
fit = fit_trend(counts, period=12, harmonics=3, seed=42)
print(fit.n_months, fit.n_observed, fit.check.reasons)

# The same series from a grouped query: months 5 and 6 produced no row.
months = np.array([m for m in range(18) if m not in (5, 6)])
gapped = fit_trend(counts[months], index=months, period=12, harmonics=2, seed=42)
print(gapped.n_months, gapped.n_observed)

try:
    fit_trend(counts, period=12, harmonics=5)
except ValueError as error:
    print(error)
```

```text
[0, 2, 3, 4, 6]
18 18 ()
18 16
18 observed months support at most 3 harmonics at period 12, not 5. Use harmonics=3 or harmonics=0 for a fit without a season; 24 observed months support every harmonic up to 6.
```

## What it does not do

- Months before the first or after the last observation are not part of the series: `index` starts at
  the first count. To forecast across a trailing gap, put NaN at the end.
- `backtest` refuses a NaN, and `causal_impact` refuses a NaN after the intervention, because
  their scores would be NaN.
- The regressors have one row per month of the dense series, including the months that were not
  observed.

## References

- Durbin, J., and Koopman, S. J. (2012). *Time Series Analysis by State Space Methods*, 2nd ed.,
  chapter 4 (missing observations). Oxford University Press.
  <https://doi.org/10.1093/acprof:oso/9780199641178.001.0001>
