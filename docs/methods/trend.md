# Trend, season and forecast of a count series

Use this to get the trend of a monthly count series with the season and the noise removed, and to
forecast it. The module is `statsx.trend`.

## The model

`fit_trend` fits a negative binomial (or Poisson) structural time series. The log of the mean in
month \(t\) is

\[
\log \mu_t = a + b\,\frac{t - \bar t}{P} + \sum_{k=1}^{K}\bigl(c_k \cos\tfrac{2\pi k t}{P} + s_k \sin\tfrac{2\pi k t}{P}\bigr) + x_t^\top \beta + d_t,
\]

where \(P\) is the period, \(K\) the number of harmonics, \(x_t\) optional standardized regressors,
and \(d_t\) a random walk (the drift). The posterior comes from the nested Laplace approximation
(Rue, Martino and Chopin 2009), corrected by importance sampling with a Pareto smoothed tail
(Vehtari et al. 2024).

## What it assumes

A fit that returns has passed a check of the computation. It has not been checked against the model.
These are assumptions of the model, and the data may not hold them:

- The season repeats with the same shape every year (a fixed Fourier shape).
- The level changes smoothly, as a random walk with a small step.
- Given the mean, the counts are independent and negative binomial. No extra spread across months.
- The priors below are the defaults. A narrow prior on a coefficient is a statement you make about it.

## What the library checks and does not check

Checked, always: the shape of the data and the period (finite, at least 2), the scales (above 0), and
the harmonic bound. After the fit, a check of the computation: the Pareto k-hat, the importance
sampling ESS, the mass at the grid edge, Newton convergence and a positive definite Hessian. The fit
raises `FitCheckError` when it fails.

Not checked:

- That the level is a random walk, or that the season keeps its amplitude and phase. The drift
  follows such changes, so the model check below does not see them.
- That a regressor causes the change you attribute to it.

A passing fit says the computation worked, not that the model holds. `fit.predictive_check()` asks
the data about four of the assumptions (see "Does the model fit the data?").

## The period

`period` is a number of months, weeks or any step, and it need not be whole. A year of weekly data
is 365.2425 / 7 = 52.1775 weeks. Rounding it to 52 moves the season out of phase by about three
quarters of a week in four years, and the error is systematic. With 600 weeks of counts near 700 a
week, the season was recovered with an rms error of 0.0024 at 52.1775 and 0.0253 at 52.

`causal_impact` and `backtest` still take a whole `period`.

## How many harmonics

Pass `harmonics="auto"` when one call must serve series of different lengths. It takes as many
harmonics as the observed length supports (`supported_harmonics`), and never more than
`max_harmonics` (default 3). A long series supports 6, which is more season than most monthly
count series need. The number used is in `TrendFit.harmonics`.

| Observed months | Harmonics supported at period 12 |
|---|---|
| fewer than 12 | 0 |
| 12 to 23 | `months // 4 - 1`, at most 6 |
| 24 or more | 6 |

An integer is not changed: it raises when the series is too short, and the error names the bound.

## When the fit check fails

`fit_trend` raises `FitCheckError` when the computation fails its check (k-hat, effective sample
size, grid edge). The error names the cause and what to change. The default stays: no check, no fit.

Pass `require=False` to get the fit anyway, and read `fit.check`: `passed`, the numbers, and the
thresholds the library applies. Show such a forecast with a caveat. A check on the computation is
not a check on the model (see "What it assumes").

The series below is simulated: 96 months with a trend, a season of period 12 and extra spread.

```python
import numpy as np
from statsx.trend import fit_trend

rng = np.random.default_rng(3)
t = np.arange(96)
mean = np.exp(5.0 + 0.004 * t + 0.5 * np.cos(2 * np.pi * t / 12) + 0.3 * np.sin(2 * np.pi * t / 12))
counts = rng.poisson(mean * rng.gamma(40.0, 1 / 40.0, t.size))
fit = fit_trend(counts, require=False, seed=0)
print(fit.check.passed, fit.check.reasons)
```

```text
True ()
```

## Does the model fit the data?

`fit.predictive_check()` simulates new series from the fitted model, one per posterior draw, and
compares four statistics of the observed counts with the same statistics of the simulated ones
(Gelman, Meng and Stern 1996). Each statistic is "larger is worse". The p-value is the share of
simulated values at least as large as the observed one. A p-value below `level` (default 0.05) is
a flag.

| Check | Looks for |
|---|---|
| `tails` | an outlier: the largest absolute Pearson residual |
| `zeros` | a surplus of zero counts |
| `phase` | a season shape the harmonics cannot hold |
| `autocorrelation` | counts that alternate or follow each other from month to month |

The series below is the one above with one month multiplied by 4.

```python
bad = counts.astype(float)
bad[50] *= 4
print(fit.predictive_check().flagged)
print(fit_trend(bad, require=False, seed=0).predictive_check().flagged)
```

```text
()
('tails',)
```

Read a flag as "the model does not describe this". It does not say what to change. An outlier may
be a data error, or a real event for which you need a regressor.

What the check cannot do:

- It does not see a level that changes or jumps, a pulse, or a season whose amplitude or phase
  changes. The drift follows them. In simulation each of these passed every check.
- It does not see extra spread under a Poisson fit. The drift absorbs it.
- A negative binomial fit estimates its spread, so a spread check would stay near 0.5. There is none.
- The p-values are conservative (Bayarri and Berger 2000). A flag is evidence. No flag is weak
  evidence, not a proof that the model is right.
- The fit has seen these counts, so the check is not a test on new data. Use `backtest` for that.

In 40 simulated series that follow the model, 1 was flagged (a `tails` flag) at the 5% level. At the
same level the check found an outlier (3 times the mean), a surplus of 20% zeros, a season spike that two
harmonics cannot hold, and counts that alternate by 0.25 on the log scale, in every simulated series.

## Counts per unit of exposure

Pass `exposure` when each month had a different amount of exposure: population, person-time,
trials, operating hours. Its log enters the log mean with the coefficient fixed at 1 (an offset,
McCullagh and Nelder 1989). The trend and the season are then the log of the rate, not of the count.

```python
exposure = 2000.0 * np.exp(np.linspace(0.0, np.log(10.0), 72))
months = np.arange(72)
log_rate = -3.0 + 0.01 * months + 0.3 * np.sin(2 * np.pi * months / 12)
cases = np.random.default_rng(15).poisson(exposure * np.exp(log_rate))
rate_fit = fit_trend(cases, exposure=exposure, seed=0)
count_fit = fit_trend(cases, seed=0)
print(round(float(np.median(rate_fit.slope)), 2), round(float(np.median(count_fit.slope)), 2))
```

```text
0.12 0.51
```

The simulated rate grows by 0.12 a year on the log scale, and the fit with `exposure` finds that.
The counts grow faster, because the exposure grows tenfold in six years: the fit without `exposure`
gives the slope of the counts.

The forecast needs the future exposure, because it is your assumption and the fit cannot know it.
Pass ones to read `mean` as a rate per unit of exposure.

```python
future = rate_fit.forecast(12, exposure=np.full(12, 20000.0), seed=0)
print(future.mean.shape)
```

```text
(2000, 84)
```

The coefficient 1 is an assumption: the count is proportional to the exposure. To check it, fit
`np.log(exposure)` as a column of `regressors` (a free coefficient) and see whether its interval
holds 1. `regressors` cannot replace `exposure`: they are standardized and their coefficient is free.

## Reuse the season on another series

The seasonal shape of a count process often belongs to the whole population: weather and holidays
do not change when you filter to a slice. `fit.seasonal` gives the draws of the seasonal term of the
log mean for the fitted months, shape (draws, months). `fit.seasonal_at(positions)` gives it at any
month position, counted from the first fitted month (0), also past the end.

Fit the season once on the long series, then give it to a short slice as a regressor with no free
season of its own:

```python
slice_counts = rng.poisson(0.1 * mean[:18] * rng.gamma(40.0, 1 / 40.0, 18))
fleet = fit_trend(counts, period=12, harmonics=3, seed=0)
shape = fleet.seasonal.mean(axis=0)[:18]
slice_fit = fit_trend(slice_counts, period=12, harmonics=0, regressors=shape, seed=0)
amplitude = np.quantile(slice_fit.regressor_coefficients[:, 0], [0.05, 0.95])
print(np.round(amplitude, 1))
```

```text
[0.7 1.4]
```

The 90% interval of the amplitude holds 1, so the slice follows the aggregate season.

The slice then fits one amplitude instead of `2 * harmonics` coefficients. Two limits:

- This is complete pooling of the shape with a free amplitude. It is not partial pooling: the slice
  cannot differ in phase or in shape, only in scale.
- The prior of the coefficient is centred on zero, so it shrinks toward no season, not toward the
  aggregate. Whether the slice follows the aggregate is "does the interval hold 1", not "does it
  exclude 0". Over 30 seeds at 18 months, the zero test fired for 43% of slices that follow the
  aggregate and 10% of slices with the amplitude shifted by three months (GitLab issue #10). A
  coefficient interval that excludes 1 says only that the slice does not follow the amplitude.

## Forecast in coarser buckets

A monthly fit can show a quarter or a year. The months of one draw move together, so sum inside
each draw, then take quantiles. Summing the bands of the months is wrong: it adds quantiles, not
variables, and the interval comes out too wide.

```python
future = fit.forecast(12, seed=1)
quarters = future.bands((0.5, 0.8), every=3)  # bands of 4 quarterly totals
totals = future.aggregate(3)                  # the draws, and where each bucket starts
print(quarters[1].lower.round(-1), quarters[1].upper.round(-1))
print(totals.starts, totals.draws.shape)
```

```text
[1090.  640.  360.  630.] [1470.  890.  510.  890.]
[ 96  99 102 105] (2000, 4)
```

- Only the forecast months are bucketed, from the first one.
- A last bucket that would be partial is dropped. A horizon of 8 months with `every=3` gives 2 buckets.
- A bucket is labelled by its first month: `totals.starts` counts from the first fitted month.
- `of="counts"` sums new counts, `of="mean"` sums expected counts. The trend is on the log scale,
  where months do not add, so it is refused.

## Priors

Every prior is settable. The defaults are in the table. Change one when you port a model and need
the same priors, or when you know the scale from other series.

| Argument | Prior it sets | Default |
|---|---|---|
| `intercept_scale` | sd of the intercept, centred on the log of the mean count | 2 |
| `slope_scale` | sd of the slope, per `period` months | 1 |
| `seasonal_scale` | sd of each Fourier coefficient | 1 |
| `regressor_scale` | sd of each (standardized) regressor coefficient | 1 |
| `drift_scale` | scale of the half-normal on the drift sd | 0.2 |
| `dispersion_scale` | scale of the half-normal on the dispersion \(\phi\) | 1 |

The slope is per `period` months. A model with a slope sd of 0.05 per month has `slope_scale=0.6` at
period 12.

The family of the dispersion prior is fixed (half-normal). A model with a Gamma prior on the
dispersion does not give the same answer even at a matched scale.

## References

Rue, H., Martino, S., & Chopin, N. (2009). Approximate Bayesian inference for latent Gaussian models
by using integrated nested Laplace approximations. *Journal of the Royal Statistical Society B*
71(2):319-392. https://doi.org/10.1111/j.1467-9868.2008.00700.x.

Vehtari, A., Simpson, D., Gelman, A., Yao, Y., & Gabry, J. (2024). Pareto smoothed importance
sampling. *Journal of Machine Learning Research* 25(72):1-58.

McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models*, 2nd ed. Chapman and Hall,
London. https://doi.org/10.1007/978-1-4899-3242-6.

Gelman, A., Meng, X.-L., & Stern, H. (1996). Posterior predictive assessment of model fitness via
realized discrepancies. *Statistica Sinica* 6(4):733-760.

Bayarri, M. J., & Berger, J. O. (2000). P values for composite null models. *Journal of the
American Statistical Association* 95(452):1127-1142.
https://doi.org/10.1080/01621459.2000.10474309.
