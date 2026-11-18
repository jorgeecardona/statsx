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

## What the result assumes

A fit that returns has passed a check of the computation. It has not been checked against the model.
These are assumptions of the model, and the data may not hold them:

- The season repeats with the same shape every year (a fixed Fourier shape).
- The level changes smoothly, as a random walk with a small step.
- Given the mean, the counts are independent and negative binomial. No extra spread across months.
- The priors below are the defaults. A narrow prior on a coefficient is a statement you make about it.

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
not a check on the model (see "What the result assumes").

```python
fit = fit_trend(counts, require=False)
if not fit.check.passed:
    caveat = fit.check.reasons
```

## Reuse the season on another series

The seasonal shape of a count process often belongs to the whole population: weather and holidays
do not change when you filter to a slice. `fit.seasonal` gives the draws of the seasonal term of the
log mean for the fitted months, shape (draws, months). `fit.seasonal_at(positions)` gives it at any
month position, counted from the first fitted month (0), also past the end.

Fit the season once on the long series, then give it to a short slice as a regressor with no free
season of its own:

```python
fleet = fit_trend(aggregate_counts, period=12, harmonics=3, seed=0)
shape = fleet.seasonal.mean(axis=0)
slice_fit = fit_trend(slice_counts, period=12, harmonics=0, regressors=shape, seed=0)
```

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
