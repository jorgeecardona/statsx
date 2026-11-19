# Which method do I use?

Find your situation in the first column. The last column is the page to read, or the API entry when
the module has no page yet.

Every method assumes something about your data that the library cannot check. Read the page before
you trust a number.

## Forecast and explain a count series

| You have | You want | Use | Read |
|---|---|---|---|
| Monthly counts | Trend, season, forecast, anomalies | `statsx.trend` | [Trend, season and forecast](methods/trend.md) |
| Counts for a treated series and control series | The effect of an intervention | `statsx.impact` | [API](api.md#statsximpact) |
| A monthly count series | A rolling-origin test of its forecast: CRPS and interval coverage | `statsx.backtest` | [API](api.md#statsxbacktest) |

## Compare two groups or two estimates

| You have | You want | Use | Read |
|---|---|---|---|
| A count of successes out of trials | An interval for the proportion | `statsx.proportion` | [API](api.md#statsxproportion) |
| Two groups of values | An effect size with an interval | `statsx.effect` | [API](api.md#statsxeffect) |
| Any statistic and a sample | A bootstrap interval, with a named resample unit | `statsx.bootstrap` | [API](api.md#statsxbootstrap) |
| Paired results (same items, two systems) | McNemar, sign, Wilcoxon, or a bootstrap difference | `statsx.paired` | [API](api.md#statsxpaired) |
| Many p-values | Holm, Benjamini-Hochberg, q-values | `statsx.multiple` | [API](api.md#statsxmultiple) |
| A margin of practical equivalence | TOST or the ROPE probability | `statsx.equivalence` | [API](api.md#statsxequivalence) |
| Two posteriors | P(A > B) and a summary of the difference | `statsx.posterior` | [API](api.md#statsxposterior) |

## Stop an experiment early, or choose between arms

| You have | You want | Use | Read |
|---|---|---|---|
| A stream of data you look at often | A confidence sequence or e-value that stays valid | `statsx.alwaysvalid` | [Always-valid inference](methods/always-valid.md) |
| A fixed number of planned looks | Stopping boundaries | `statsx.groupsequential` | [Group sequential bounds](methods/group-sequential.md) |
| A budget and several arms | Thompson sampling, Bayes-UCB, successive halving | `statsx.bandit` | [Spending a budget over arms](methods/arms.md) |
| Several arms and a target error | The best arm, with a stop rule | `statsx.identify` | [Spending a budget over arms](methods/arms.md) |

## Plan a run

| You have | You want | Use | Read |
|---|---|---|---|
| An expected effect and a variance | Power, sample size, minimum detectable effect | `statsx.design` | [API](api.md#statsxdesign) |
| An evaluation set with strata | How many items per stratum | `statsx.allocation` | [API](api.md#statsxallocation) |

## Compare models and systems

| You have | You want | Use | Read |
|---|---|---|---|
| Results of several models on shared tasks | One scale for the models, with costs | `statsx.evals` | [Compare models on one scale](methods/model-comparison.md) |
| Pairwise judgements (A beats B), possibly with judges | One strength per system, with intervals | `statsx.pairwise` | [Rank systems from pairwise results](methods/pairwise.md) |
| A stream of match results | Online Elo or Glicko-2 ratings | `statsx.rating` | [Update ratings from a stream](methods/rating.md) |
| Ratings from several raters | Cohen's kappa, Fleiss' kappa, Krippendorff's alpha | `statsx.agreement` | [API](api.md#statsxagreement) |

## Check a classifier, a distribution or a drift

| You have | You want | Use | Read |
|---|---|---|---|
| Predicted probabilities and outcomes | Calibration, Brier decomposition, Platt, isotonic | `statsx.calibration` | [API](api.md#statsxcalibration) |
| Sufficient statistics of a sample | A fitted normal, lognormal or shifted exponential | `statsx.distributions` | [Distributions and goodness of fit](methods/distributions.md) |
| A fitted distribution and a sample | Goodness of fit at an effective sample size | `statsx.goodness` | [Distributions and goodness of fit](methods/distributions.md) |
| Two samples | Whether they drifted: KS, MMD, classifier test, PSI | `statsx.drift` | [Drift between two samples](methods/drift.md) |

## Bayes and decisions

| You have | You want | Use | Read |
|---|---|---|---|
| Counts and a conjugate model | The closed-form posterior | `statsx.conjugate` | [API](api.md#statsxconjugate) |
| An expert's calibrated interval | A lognormal prior | `statsx.prior` | [API](api.md#statsxprior) |
| Draws from an MCMC run | Convergence checks and Monte Carlo error | `statsx.diagnostics` | [Check MCMC draws](methods/convergence.md) |
| A posterior method | A test that it is correct | `statsx.sbc` | [Check a posterior method](methods/sbc.md) |
| Uncertain costs and benefits | A cost distribution, break-even, value of information | `statsx.decision` | [API](api.md#statsxdecision) |

## Time to finish a task

| You have | You want | Use | Read |
|---|---|---|---|
| Past tasks, with their input parameters and durations | The duration of a new task, with a credible interval | `statsx.eta` | [API](api.md#statsxeta) |
| A running task with its progress | An updated ETA | `statsx.progress` | [API](api.md#statsxprogress) |
