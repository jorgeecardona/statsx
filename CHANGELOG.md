# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `statsx.impact.causal_impact`: the effect of an intervention on a count series from control
  series, as the effect for each month, the cumulative and average effect (with interval, sample
  size and assumption), and P(effect > 0).
- `fit_trend` takes `regressors` (terms in the log mean, needed again for the forecast months)
  and `likelihood="poisson"`. The negative binomial stays the default: a Poisson fit of
  overdispersed counts puts the extra variance into the trend.
- `TrendFit.anomalies`: flags the months whose count is outside the 95% band the model
  expected for them, with the band and the direction for every month.
- `TrendFit.forecast`: nested 50/80/95% credible bands for each fitted and future month
  (`Forecast.bands`), of the log trend, the expected count or a new count. The trend bands
  widen with the horizon.
- `statsx.trend.fit_trend`: the trend of a monthly count series with the season and the noise
  removed, as credible bands (`TrendFit.band`) and P(slope > 0) (`TrendFit.p_increasing`). A fit
  that fails its check raises `FitCheckError` with the reasons; it never returns a degraded fit.
- `Estimate` and `Interval`: every result gives the estimate, the interval, the sample size,
  and the assumption as fields of a frozen dataclass.
- NumPy is the only run-time dependency, and `import statsx` loads no torch, pandas, SciPy,
  plotting, database, network or cloud module. Tests check both.
- `statsx.conjugate`: Beta-Binomial, Gamma-Poisson, Normal-Normal and Dirichlet-Multinomial
  updates return the posterior as a named object (`Beta`, `Gamma`, `Normal`, `Dirichlet`).
- `statsx.bootstrap`: bootstrap with a named resample unit (`IID`, `Stratified`, `Clustered`,
  `Block`) and a percentile interval; the result records the unit.
- `statsx.diagnostics.check_convergence`: rank-normalized split R-hat, bulk and tail ESS, and
  divergences in one check; `require()` raises `ConvergenceError` with the reasons.
- `statsx.eta.fit_duration`: learn task duration from past runs and their parameters (for
  example file count and size); `predict` gives the duration with a credible interval.
- Initial release.
