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
