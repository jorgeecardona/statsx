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
- Initial release.
