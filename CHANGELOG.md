# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.62] - 2026-11-20

### Added

- `UMAPFit.transform(x)` places new points in a finished UMAP map without changing it, from
  their exact nearest training points. No paper describes this step; the docs say which
  description it follows. `UMAPFit` now keeps the fitted data.
- `umap(x, metric="cosine")` maps rows by the angle between them, for embeddings whose length
  means nothing. `neighbor_preservation` takes the same `metric`.
- `statsx.embedding.umap`: UMAP from the paper, in Rust. It builds the weighted neighbour graph,
  starts from the eigenvectors of its Laplacian, and lays it out by sampled gradient descent. The
  neighbours are exact. The map is the same for any thread count. `neighbor_preservation` says how
  many neighbours a map keeps, and `UMAPFit.cross_entropy` gives the objective. No library besides
  NumPy.
- `statsx.clustering.dbscan`, `hdbscan` and `k_distances`: density clustering from the papers, in
  Rust, with exact distances. Groups have any shape, points in no group are noise (-1), and no
  number of clusters is given. HDBSCAN also finds groups of different density. No library besides
  NumPy.
- `statsx.clustering.gaussian_mixture`: a Gaussian mixture fitted by EM (full, diagonal or spherical
  covariance), with a probability for each point, `predict_proba`, `score_samples` and `sample`. It
  drops runs where a component collapses onto a few points. `choose_components` picks the number of
  components by BIC or AIC. No library besides NumPy.
- `statsx.clustering`: `kmeans` (k-means++ seeds, Lloyd iteration, a Rust kernel that gives the same
  fit for any thread count), `silhouette`, and `choose_k` by the gap statistic. No library besides
  NumPy.
- `Forecast.rates` divides each count draw by the exposure of its month. The docs now say that a
  forecast at exposure 1 gives a rate as `mean` but a band of counts that is nearly all 0 (GitLab
  issue #16).

## [0.1.61] - 2026-11-20

### Added

- `cost_at_ability` gives each model's cost per solved task at the quality of a reference model,
  counting the retries a weaker model needs (GitLab issue #14).

## [0.1.60] - 2026-11-20

### Added

- `fit_tokens(..., censored=...)` treats runs that hit the token budget as censored, so a model
  that needs more than the budget no longer looks cheaper (GitLab issue #13).

## [0.1.59] - 2026-11-20

### Added

- `cost_per_pass(..., tasks=...)` resamples whole tasks for its interval, so a few hard, costly
  tasks no longer make it too narrow (GitLab issue #12).

## [0.1.58] - 2026-11-20

### Added

- `fit_trend(exposure=...)` fits the trend of a rate: the log of the exposure (population,
  person-time, trials, hours) enters the model with its coefficient fixed at 1. `forecast` then
  takes the future exposure (GitLab issue #15).

## [0.1.57] - 2026-11-20

### Added

- `TrendFit.predictive_check()` asks the data whether the model holds. It simulates series from the
  fit and flags an outlier, a surplus of zeros, a season shape the harmonics cannot hold, and counts
  that alternate. The docs list what it cannot see: a changing level or season is absorbed by the
  drift.

## [0.1.56] - 2026-11-19

### Added

- `fit_by_judge` gives each judge its own strengths and tests whether the judges rank the systems
  alike. The test uses a covariance from resampling the items once for all judges.

## [0.1.55] - 2026-11-19

### Added

- Methods pages for the modules that had none: causal impact, paired tests with equivalence and
  multiple testing, calibration, intervals (proportion, effect size, bootstrap), power and
  allocation, time to finish a task, backtest, agreement, priors with conjugate updates, and
  decisions.
- Page "Which method do I use?": your situation in one column, the function and its page in the
  next.
- Every methods page has two headings, "What it assumes" and "What the library checks and does not
  check".
- Every python example in the README and the methods pages runs and shows its output. A block with
  no output fails the doc test.

## [0.1.54] - 2026-11-19

### Fixed

- `BradleyTerryFit.p_first_wins` gave the Bradley-Terry value even with Davidson ties, where a tie
  is not a win. It now follows the fitted model.
- The Cost-of-Pass authors and the PPI++ arXiv number were wrong in the docs and docstrings.

## [0.1.52] - 2026-11-18

### Added

- New module `statsx.rating`: online Elo (`elo_expected`, `elo_update`) and Glicko-2
  (`glicko2_update`) from a stream of results. The Glicko-2 update reproduces the worked example of
  Glickman (2022).

## [0.1.51] - 2026-11-18

### Added

- `BradleyTerryFit.elo` and `elo_interval` give a strength on the Elo scale (400 points are odds of
  10 to 1), with the bootstrap interval.

## [0.1.49] - 2026-11-18

### Added

- New page `ideas.md`: a running log of ideas and open questions. New principle P-3: a computation
  that runs is not evidence that its assumption holds.
- `fit_trend(require=False)` returns the fit with its failed check instead of raising, so a caller
  can show a forecast with a caveat. (#9)
- `TrendFit.seasonal` and `seasonal_at(positions)` give draws of the seasonal term, to reuse a
  season fitted on a long series as a regressor on a short one. The docs say it is complete pooling.
  (#10)
- `Forecast.aggregate(every)` and `bands(every=...)` sum forecast months into buckets inside each
  draw, so a monthly fit serves quarters and years. (#11)

## [0.1.48] - 2026-11-18

### Added

- `fit_trend(period=...)` takes a float, so a weekly series can use 365.2425 / 7 = 52.1775. A drift
  that is off (a random walk with no change of level) is a valid answer and no longer refused as a
  grid edge. (#8)
- `fit_trend(harmonics="auto", max_harmonics=3)` takes as many harmonics as the series length
  supports, up to the ceiling, so one call serves series of any length. (#6)

## [0.1.47] - 2026-11-18

### Added

- `fit_trend` takes `intercept_scale`, `slope_scale`, `seasonal_scale` and `regressor_scale`, the
  prior scales that were fixed, with today's values as defaults. New page `methods/trend.md` states
  the model, its priors and what it assumes. (#5)

## [0.1.46] - 2026-11-18

### Fixed

- `fit_trend`: both ways of "cannot fit" now share a base, `statsx.errors.StatsxError`.
  `UnfittableInput` (a `ValueError`) is a series that cannot be fitted at all; `FitCheckError` (a
  `RuntimeError`) is a fit that did not pass its check. Old handlers keep working. (#7)

## [0.1.45] - 2026-11-18

### Fixed

- `fit_trend` no longer refuses a series whose counts are close to Poisson: mass on the Poisson end
  of the dispersion grid is a valid answer, not an edge failure (the grid now reaches it). Each
  fit-check reason now says what it means and what to change. (#4)

## [0.1.44] - 2026-11-18

### Added

- `statsx.pairwise`: a `judge` column in `Comparisons.from_long` gives each judge a position bias,
  so a judge that favours the first answer does not tilt the strengths (`position_bias`,
  `position_bias_interval`, `p_first_wins`). Data where the order is not mixed are refused, naming
  the judges.

## [0.1.43] - 2026-11-18

### Added

- `statsx.diagnostics`: `mcse_mean`, `mcse_sd`, `mcse_quantile` (Monte Carlo standard error of a
  posterior mean, standard deviation and quantile) and `ess_mean`. Docs page "Check MCMC draws and
  their Monte Carlo error", which also explains R-hat and ESS.

## [0.1.42] - 2026-11-18

### Added

- `statsx.evals`: an `infra` column (GPU, driver, host, run batch) in `Responses.from_long`,
  `fit_tokens` and `fit_rasch` fits one effect per infra, so a model ratio or ability is read with
  the infra effect removed (`TokenFit.infra_effect`, `RaschFit.infra_shift_difference`). Data where
  no model links the infras are refused, naming the models. `calibrate_infra` lists the bridge
  models. (#3)

## [0.1.41] - 2026-11-18

### Added

- `statsx.sbc`: simulation-based calibration. `sbc_ranks` and `check_calibration` test that a
  posterior method is correct, and say whether it is too narrow, too wide or biased. Docs page
  "Check that a posterior method is correct".

## [0.1.40] - 2026-11-18

### Added

- `statsx.pairwise`: `Comparisons` and `fit_bradley_terry` (Bradley-Terry and Davidson with ties)
  with bootstrap intervals that resample whole items. Docs page "Rank systems from pairwise
  results".

## [0.1.39] - 2026-11-18

### Added

- `fit_trend` fits series shorter than two seasons (`supported_harmonics` says how many harmonics a
  length carries; `harmonics=0` fits no season) and months that were not observed (NaN in `counts`,
  or an `index`). Its errors now name the failed condition. `causal_impact` and `backtest` refuse a
  NaN that would make their scores NaN. (#1, #2)

## [0.1.38] - 2026-11-18

### Added

- `statsx.evals`: `Responses`, `fit_rasch` (abilities and difficulties of models on shared tasks,
  with intervals and `p_better`), `fit_tokens` (token ratios between models on the same tasks) and
  `cost_per_pass`. Docs page "Compare models on one scale".

## [0.1.37] - 2026-11-18

### Added

- `statsx.drift`: `ks_test` (exact for small samples), `mmd_test`, `classifier_test` and `psi` /
  `psi_from_shares` to compare two samples for drift.

## [0.1.36] - 2026-11-18

### Added

- `statsx.design`: power, sample size, minimum detectable effect and TOST for two independent groups
  of means (`power_two_sample`, `sample_size_two_sample`, `minimum_detectable_effect_two_sample`,
  `power_equivalence_two_sample`, `sample_size_equivalence_two_sample`).

## [0.1.35] - 2026-11-18

### Added

- `statsx.conjugate.NormalInverseGamma`: posterior for the mean and variance of a normal sample,
  with `update`, an exact t credible interval, `sample` (works in `compare`) and `reference` (the
  classical t interval). Docs recipe "Compare two timing runs".

## [0.1.34] - 2026-11-17

### Added

- Docs: methods pages with LaTeX (always-valid inference, group sequential bounds, bandits and best
  arm, distributions and goodness of fit), a full API reference, and a test that runs every page's
  example.
- `statsx.goodness`: `goodness_of_fit` and `chi_square_test` at a stated effective sample size, with
  total variation distance, exact Wasserstein distance and the sample size that would reject the
  fit. New Rust `chi2_cdf` and `chi2_sf`.

## [0.1.33] - 2026-11-17

### Added

- `statsx.distributions`: `fit_normal`, `fit_lognormal` and `fit_shifted_exponential` from
  sufficient statistics (`Moments`, `ExponentialStats`) that merge across shards, with cdf, ppf and
  sample.

## [0.1.32] - 2026-11-17

### Added

- `statsx.identify.best_arm`: find the best arm and stop when it is certain, with an error rate
  `delta` and an optional tolerance `epsilon`. `statsx.alwaysvalid.confidence_radius` gives the
  interval half-width for a running total.

## [0.1.31] - 2026-11-17

### Added

- `statsx.bandit`: `thompson_allocation`, `bayes_ucb_indices`, `bayes_ucb_arm` and
  `successive_halving` to spend a budget over arms with Beta posteriors.

## [0.1.30] - 2026-11-17

### Added

- `statsx.alwaysvalid.confidence_sequence` and `e_process`: always-valid confidence sequence for a
  mean and e-value for "the mean is mu0", valid when you look after every observation and stop
  early.

## [0.1.29] - 2026-11-17

### Added

- `statsx.groupsequential.group_sequential_bounds`: O'Brien-Fleming and Pocock critical values for a
  planned number of looks, with the nominal p-value and the error spent at each look.

## [0.1.28] - 2026-11-17

### Added

- `statsx.design`: power, sample size and minimum detectable effect for paired differences, two
  proportions, and equivalence (TOST) of a paired difference or two proportions. The noncentral t
  distribution is in Rust.

## [0.1.27] - 2026-11-17

### Added

- `statsx.allocation.stratified_allocation`: how many items to draw in each stratum for a target
  margin of error (Neyman or proportional, with optional costs).

## [0.1.26] - 2026-11-17

### Added

- `statsx.calibration.fit_platt` and `fit_isotonic`: recalibrate a classifier's scores with Platt
  scaling or isotonic regression, both written in Rust.

## [0.1.25] - 2026-11-17

### Added

- `statsx.calibration.reliability`: reliability diagram numbers, ECE, MCE and the exact Brier
  decomposition of a classifier's probabilities.

## [0.1.24] - 2026-11-17

### Added

- `statsx.agreement`: `cohen_kappa` (plain, linear or quadratic weights), `fleiss_kappa` and
  `krippendorff_alpha` (missing ratings, four levels of measurement), each with a bootstrap
  interval.

## [0.1.23] - 2026-11-17

### Added

- `statsx.paired`: `mcnemar` (exact or chi-square), `sign_test`, `wilcoxon` (exact up to 50
  differences, ties included) and `bootstrap_difference` for paired results.

## [0.1.22] - 2026-11-17

### Added

- `statsx.equivalence`: one `EquivalenceMargin`, a TOST test (`tost`, Welch or paired) and
  `rope_probability`. `compare` takes the same margin.

## [0.1.21] - 2026-11-17

### Added

- `statsx.multiple.adjust`: Holm, Benjamini-Hochberg and Storey q-value corrections for many tests,
  with the adjusted values, the rejections and the estimated share of true nulls.

## [0.1.20] - 2026-11-17

### Added

- `statsx.decision`: `simulate_cost` (Monte Carlo cost distribution with a fixed seed),
  `expected_value_of_information`, `break_even_probability` and `break_even`.

## [0.1.19] - 2026-11-17

### Added

- `statsx.posterior.compare`: P(A > B) and the probability of practical equivalence of two
  posteriors, with the credible interval of the difference. `Beta`, `Gamma` and `Normal` have a
  `sample` method.
- `statsx.prior.lognormal_from_interval`: the lognormal prior that matches an expert's calibrated
  90% interval.

## [0.1.18] - 2026-11-17

### Added

- `statsx.proportion.proportion_interval`: Wilson, Jeffreys, Clopper-Pearson and Agresti-Coull
  intervals for a proportion. The module notes state the small-N coverage of each, and a test checks
  it.

## [0.1.17] - 2026-11-17

### Added

- `statsx.effect.effect_size`: Hedges' g, Cohen's d or the mean difference of two groups, with an
  analytic or bootstrap interval, the sample size and the assumption.

## [0.1.16] - 2026-11-17

### Added

- `bootstrap(..., method="basic" | "bca")`: basic and BCa intervals next to the percentile interval.
  The result records the method. BCa needs `IID` or `Clustered` resampling.

## [0.1.15] - 2026-11-17

### Added

- `statsx.posterior`: `quantiles`, `credible_interval` (equal-tailed) and `hdi` (highest density
  interval) of posterior draws, one function each.

## [0.1.14] - 2026-11-17

### Added

- `statsx.progress.fit_progress`: the time left of a running task, with an interval, from its
  elapsed time and the fraction done, learned from past runs that reported progress.

## [0.1.13] - 2026-11-17

### Added

- `DurationModel.predict` takes only the parameters known before the run. The model is refitted
  without the unknown ones, so the interval is wider, and the result names them.

## [0.1.12] - 2026-11-17

### Added

- `DurationModel.update`: add a finished run to the duration model without refitting from all past
  runs; the result equals a refit up to rounding.

## [0.1.11] - 2026-11-17

### Added

- `statsx.backtest`: rolling-origin backtest of a forecast with CRPS and interval coverage by step
  ahead. Refused fits are counted and reported, and `backtest_all` reports every series it is given.

## [0.1.10] - 2026-11-17

### Added

- `statsx.impact.causal_impact`: the effect of an intervention on a count series from control
  series, as the effect for each month, the cumulative and average effect (with interval, sample
  size and assumption), and P(effect > 0).

## [0.1.9] - 2026-11-17

### Added

- `fit_trend` takes `regressors` (terms in the log mean, needed again for the forecast months) and
  `likelihood="poisson"`. The negative binomial stays the default: a Poisson fit of overdispersed
  counts puts the extra variance into the trend.

## [0.1.8] - 2026-11-17

### Added

- `TrendFit.anomalies`: flags the months whose count is outside the 95% band the model expected for
  them, with the band and the direction for every month.

## [0.1.7] - 2026-11-17

### Added

- `TrendFit.forecast`: nested 50/80/95% credible bands for each fitted and future month
  (`Forecast.bands`), of the log trend, the expected count or a new count. The trend bands widen
  with the horizon.

## [0.1.6] - 2026-11-17

### Added

- `statsx.trend.fit_trend`: the trend of a monthly count series with the season and the noise
  removed, as credible bands (`TrendFit.band`) and P(slope > 0) (`TrendFit.p_increasing`). A fit
  that fails its check raises `FitCheckError` with the reasons; it never returns a degraded fit.

## [0.1.5] - 2026-11-16

### Added

- `statsx.eta.fit_duration`: learn task duration from past runs and their parameters (for example
  file count and size); `predict` gives the duration with a credible interval.

## [0.1.4] - 2026-11-16

### Added

- `statsx.diagnostics.check_convergence`: rank-normalized split R-hat, bulk and tail ESS, and
  divergences in one check; `require()` raises `ConvergenceError` with the reasons.

## [0.1.3] - 2026-11-16

### Added

- `statsx.bootstrap`: bootstrap with a named resample unit (`IID`, `Stratified`, `Clustered`,
  `Block`) and a percentile interval; the result records the unit.

## [0.1.2] - 2026-11-16

### Added

- `statsx.conjugate`: Beta-Binomial, Gamma-Poisson, Normal-Normal and Dirichlet-Multinomial updates
  return the posterior as a named object (`Beta`, `Gamma`, `Normal`, `Dirichlet`).

## [0.1.1] - 2026-11-16

### Added

- `Estimate` and `Interval`: every result gives the estimate, the interval, the sample size, and the
  assumption as fields of a frozen dataclass.
- NumPy is the only run-time dependency, and `import statsx` loads no torch, pandas, SciPy,
  plotting, database, network or cloud module. Tests check both.

## [0.1.0] - 2026-11-16

### Added

- Initial release.
