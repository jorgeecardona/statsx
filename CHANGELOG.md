# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `fit_trend`: both ways of "cannot fit" now share a base, `statsx.errors.StatsxError`. `UnfittableInput` (a
  `ValueError`) is a series that cannot be fitted at all; `FitCheckError` (a `RuntimeError`) is a fit that did
  not pass its check. Old handlers keep working. Fixes GitLab issue #7.
- `fit_trend` no longer refuses a series whose counts are close to Poisson: mass on the Poisson end of the
  dispersion grid is a valid answer, not an edge failure (the grid now reaches it). Each fit-check
  reason now says what it means and what to change. Fixes GitLab issue #4.

### Added

- `statsx.pairwise`: a `judge` column in `Comparisons.from_long` gives each judge a position bias, so a
  judge that favours the first answer does not tilt the strengths (`position_bias`,
  `position_bias_interval`, `p_first_wins`). Data where the order is not mixed are refused, naming
  the judges.
- `statsx.diagnostics`: `mcse_mean`, `mcse_sd`, `mcse_quantile` (Monte Carlo standard error of a
  posterior mean, standard deviation and quantile) and `ess_mean`. Docs page "Check MCMC draws and
  their Monte Carlo error", which also explains R-hat and ESS.
- `statsx.evals`: an `infra` column (GPU, driver, host, run batch) in `Responses.from_long`, `fit_tokens`
  and `fit_rasch` fits one effect per infra, so a model ratio or ability is read with the infra
  effect removed (`TokenFit.infra_effect`, `RaschFit.infra_shift_difference`). Data where no model
  links the infras are refused, naming the models. `calibrate_infra` lists the bridge models.
  Fixes GitLab issue #3.
- `statsx.sbc`: simulation-based calibration. `sbc_ranks` and `check_calibration` test that a posterior
  method is correct, and say whether it is too narrow, too wide or biased. Docs page "Check that a
  posterior method is correct".
- `statsx.pairwise`: `Comparisons` and `fit_bradley_terry` (Bradley-Terry and Davidson with ties) with
  bootstrap intervals that resample whole items. Docs page "Rank systems from pairwise results".
- `fit_trend` fits series shorter than two seasons (`supported_harmonics` says how many harmonics a length
  carries; `harmonics=0` fits no season) and months that were not observed (NaN in `counts`, or an `index`).
  Its errors now name the failed condition. `causal_impact` and `backtest` refuse a NaN that would
  make their scores NaN. Fixes GitLab issues #1 and #2.
- `statsx.evals`: `Responses`, `fit_rasch` (abilities and difficulties of models on shared tasks, with
  intervals and `p_better`), `fit_tokens` (token ratios between models on the same tasks) and
  `cost_per_pass`. Docs page "Compare models on one scale".
- `statsx.drift`: `ks_test` (exact for small samples), `mmd_test`, `classifier_test` and `psi` /
  `psi_from_shares` to compare two samples for drift.
- `statsx.design`: power, sample size, minimum detectable effect and TOST for two independent groups
  of means (`power_two_sample`, `sample_size_two_sample`, `minimum_detectable_effect_two_sample`,
  `power_equivalence_two_sample`, `sample_size_equivalence_two_sample`).
- `statsx.conjugate.NormalInverseGamma`: posterior for the mean and variance of a normal sample, with
  `update`, an exact t credible interval, `sample` (works in `compare`) and `reference` (the
  classical t interval). Docs recipe "Compare two timing runs".
- Docs: methods pages with LaTeX (always-valid inference, group sequential bounds, bandits and best
  arm, distributions and goodness of fit), a full API reference, and a test that runs every page's
  example.
- `statsx.goodness`: `goodness_of_fit` and `chi_square_test` at a stated effective sample size, with
  total variation distance, exact Wasserstein distance and the sample size that would reject the fit.
  New Rust `chi2_cdf` and `chi2_sf`.
- `statsx.distributions`: `fit_normal`, `fit_lognormal` and `fit_shifted_exponential` from
  sufficient statistics (`Moments`, `ExponentialStats`) that merge across shards, with cdf, ppf and
  sample.
- `statsx.identify.best_arm`: find the best arm and stop when it is certain, with an error rate
  `delta` and an optional tolerance `epsilon`. `statsx.alwaysvalid.confidence_radius` gives the
  interval half-width for a running total.
- `statsx.bandit`: `thompson_allocation`, `bayes_ucb_indices`, `bayes_ucb_arm` and
  `successive_halving` to spend a budget over arms with Beta posteriors.
- `statsx.alwaysvalid.confidence_sequence` and `e_process`: always-valid confidence sequence for a
  mean and e-value for "the mean is mu0", valid when you look after every observation and stop early.
- `statsx.groupsequential.group_sequential_bounds`: O'Brien-Fleming and Pocock critical values for
  a planned number of looks, with the nominal p-value and the error spent at each look.
- `statsx.design`: power, sample size and minimum detectable effect for paired differences, two
  proportions, and equivalence (TOST) of a paired difference or two proportions. The noncentral t
  distribution is in Rust.
- `statsx.allocation.stratified_allocation`: how many items to draw in each stratum for a
  target margin of error (Neyman or proportional, with optional costs).
- `statsx.calibration.fit_platt` and `fit_isotonic`: recalibrate a classifier's scores with Platt
  scaling or isotonic regression, both written in Rust.
- `statsx.calibration.reliability`: reliability diagram numbers, ECE, MCE and the exact Brier
  decomposition of a classifier's probabilities.
- `statsx.agreement`: `cohen_kappa` (plain, linear or quadratic weights), `fleiss_kappa` and
  `krippendorff_alpha` (missing ratings, four levels of measurement), each with a bootstrap
  interval.
- `statsx.paired`: `mcnemar` (exact or chi-square), `sign_test`, `wilcoxon` (exact up to 50
  differences, ties included) and `bootstrap_difference` for paired results.
- `statsx.equivalence`: one `EquivalenceMargin`, a TOST test (`tost`, Welch or paired) and
  `rope_probability`. `compare` takes the same margin.
- `statsx.multiple.adjust`: Holm, Benjamini-Hochberg and Storey q-value corrections for many
  tests, with the adjusted values, the rejections and the estimated share of true nulls.
- `statsx.decision`: `simulate_cost` (Monte Carlo cost distribution with a fixed seed),
  `expected_value_of_information`, `break_even_probability` and `break_even`.
- `statsx.posterior.compare`: P(A > B) and the probability of practical equivalence of two
  posteriors, with the credible interval of the difference. `Beta`, `Gamma` and `Normal` have a
  `sample` method.
- `statsx.prior.lognormal_from_interval`: the lognormal prior that matches an expert's calibrated
  90% interval.
- `statsx.proportion.proportion_interval`: Wilson, Jeffreys, Clopper-Pearson and Agresti-Coull
  intervals for a proportion. The module notes state the small-N coverage of each, and a test
  checks it.
- `statsx.effect.effect_size`: Hedges' g, Cohen's d or the mean difference of two groups, with
  an analytic or bootstrap interval, the sample size and the assumption.
- `bootstrap(..., method="basic" | "bca")`: basic and BCa intervals next to the percentile
  interval. The result records the method. BCa needs `IID` or `Clustered` resampling.
- `statsx.posterior`: `quantiles`, `credible_interval` (equal-tailed) and `hdi` (highest density
  interval) of posterior draws, one function each.
- `statsx.progress.fit_progress`: the time left of a running task, with an interval, from its
  elapsed time and the fraction done, learned from past runs that reported progress.
- `DurationModel.predict` takes only the parameters known before the run. The model is refitted
  without the unknown ones, so the interval is wider, and the result names them.
- `DurationModel.update`: add a finished run to the duration model without refitting from all
  past runs; the result equals a refit up to rounding.
- `statsx.backtest`: rolling-origin backtest of a forecast with CRPS and interval coverage by
  step ahead. Refused fits are counted and reported, and `backtest_all` reports every series
  it is given.
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
