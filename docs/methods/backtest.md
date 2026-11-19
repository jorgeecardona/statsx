# Backtest a count forecast

Use this to check how good the forecasts of `fit_trend` are on your own series, before you trust one.
The module is `statsx.backtest`.

## The method

`backtest(counts, horizon=, initial=, step=)` walks through the series. At each origin it refits
[`fit_trend`](trend.md) on the months before the origin, forecasts the next `horizon` months and scores the
forecast draws against the counts that followed (Tashman 2000). The first origin uses `initial` months, and
each next origin moves `step` months later.

Two scores are reported:

- CRPS, the continuous ranked probability score, for every forecast month. It is lower when the forecast
  is closer to the outcome and its spread is right (Gneiting and Raftery 2007). It is computed from draws
  with \(\mathrm{CRPS} = \mathbb{E}|X - y| - \tfrac12 \mathbb{E}|X - X'|\), where \(X\) and \(X'\) are independent
  draws and \(y\) is the observed count.
- Coverage: the share of observed counts inside each central band of the forecast (levels 0.5, 0.8 and 0.95
  by default). A 0.8 band that holds 80% of the counts is calibrated.

An origin whose fit fails its check is counted as refused and reported, not dropped. `backtest_all` runs
every series you give it and reports all of them. A forecast claim that rests on a chosen subset of series
is not a claim about the method (Roque et al. 2025).

## What it assumes

- The future looks like the past origins. A change in the process after the last origin is not in the
  score.
- The counts are observed. A missing month makes the scores undefined, so the series is refused.
- Origins overlap when `step` is smaller than `horizon` or the series is short. The scores of nearby
  origins share months, so they are not independent.

## What the library checks and does not check

Checked: the counts are one-dimensional, `horizon` and `step` are at least 1, `initial` is at least
\(2 \times\) `period`, there is room for at least one origin, regressors have one row per month, no month is
missing, and every refused fit is counted and shown.

Not checked:

- Whether a difference in CRPS between two methods is more than noise. No interval is given for it.
- That the coverage is estimated well. With 18 forecasts per series, the coverage of a band moves a lot.
- That your series are a fair sample of the series you will forecast.

## Example

The data are simulated: three monthly series of 72 months with different trends and seasons. Each is
backtested with origins every 6 months from month 36 and a horizon of 3 months.

```python
import numpy as np
from statsx.backtest import backtest_all

rng = np.random.default_rng(0)
t = np.arange(72)


def series(trend, amplitude):
    mean = np.exp(5.0 + trend * t + amplitude * np.cos(2 * np.pi * t / 12))
    return rng.poisson(mean * rng.gamma(40.0, 1 / 40.0, t.size))


data = {"a": series(0.004, 0.4), "b": series(-0.003, 0.2), "c": series(0.0, 0.6)}
report = backtest_all(data, horizon=3, initial=36, step=6)
print("origins:", report.n_origins, "refused:", report.n_refused)
print("mean CRPS:", round(report.mean_crps, 1))
for name, result in report.results.items():
    print(name, "coverage of the 50%, 80%, 95% bands:", np.round(result.coverage, 2))
```

```text
origins: 18 refused: 0
mean CRPS: 15.8
a coverage of the 50%, 80%, 95% bands: [0.5  0.89 1.  ]
b coverage of the 50%, 80%, 95% bands: [0.44 0.78 0.94]
c coverage of the 50%, 80%, 95% bands: [0.89 0.94 1.  ]
```

Each series has 6 origins and 18 forecast months. A coverage of 0.5 or 1.0 in such a small count is
noise, not a flaw of the method. Use more origins before you judge a band.

## Checked in the tests

- CRPS from draws matches the pairwise definition, and the closed form for a normal forecast.
- The backtest scores every origin with CRPS and coverage, and a better forecast has a lower CRPS.
- Every series is reported, and refused origins are counted.
- The regressors are sliced for each origin.
- Bad arguments are refused.

## References

- Gneiting, T., and Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation.
  *Journal of the American Statistical Association* 102(477):359-378.
  <https://doi.org/10.1198/016214506000001437>
- Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: an analysis and review.
  *International Journal of Forecasting* 16(4):437-450.
  <https://doi.org/10.1016/S0169-2070(00)00065-0>
- Roque, L., Cerqueira, V., Soares, C., and Torgo, L. (2025). Cherry-picking in time series forecasting:
  how to select datasets to make your model shine. *Proceedings of the AAAI Conference on Artificial
  Intelligence* 39(19):20192-20199.
