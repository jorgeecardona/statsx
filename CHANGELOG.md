# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
