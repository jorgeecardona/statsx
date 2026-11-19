# Intervals for a proportion, an effect size and any statistic

Use this to put an interval on a proportion, on the difference between two groups, or on any statistic
of your own. Each result carries its estimate, its interval, the sample size and the assumption. The
modules are `statsx.proportion`, `statsx.effect` and `statsx.bootstrap`.

## A proportion

`proportion_interval(successes, trials, method=...)` gives the sample proportion and an interval. The
methods differ most with few trials. Coverage is the share of samples whose interval holds the true
proportion. These numbers are exact from the binomial distribution, for true proportions from 0.0005 to
0.9995, at 95% and 10 trials (a test checks each line):

| Method | Mean coverage | Lowest coverage | Mean width |
|---|---|---|---|
| `wilson` (default) | 95.4% | 83.6%, for a proportion near 0 | 0.44 |
| `jeffreys` | 95.3% | 86.8% | |
| `clopper_pearson` | 98.4% | 96.1% | 0.51 |
| `agresti_coull` | 96.5% | 92.4% | 0.46 |

`wilson` is never empty and stays inside [0, 1]. `jeffreys` sets the ends to 0 for 0 successes and to 1
for all successes. `clopper_pearson` is exact, so its coverage is at least 95%: use it when under-coverage
is not acceptable, and accept the wider interval. The lower end of `agresti_coull` can be below 0 and is
then cut to 0. With 30 trials the lowest coverage is 84.1% (Wilson), 88.8% (Jeffreys), 95.1%
(Clopper-Pearson) and 93.4% (Agresti-Coull).

## The difference between two groups

`effect_size(a, b)` returns Cohen's d or Hedges' g with an interval. Cohen's d is the mean difference
over the pooled standard deviation. Hedges' g is d times a small-sample correction \(J\), which removes the
upward bias of d. The analytic interval is the large-sample normal interval with variance

\[
\frac{n_1 + n_2}{n_1 n_2} + \frac{d^2}{2(n_1 + n_2)} .
\]

On normal data the 90% interval held the true effect in 89%, 91% and 91% of 1000 samples with 10, 20 and
40 observations per group. For skewed data, or when equal variances are in doubt, use
`method="bootstrap"`.

## Any statistic: the bootstrap

`bootstrap(data, statistic, unit)` resamples `data`, applies `statistic` to each resample and takes the
interval from the replicates. The resample unit is part of the result, so a reader can see what was
resampled:

| Unit | Resamples | Use when |
|---|---|---|
| `IID()` | single observations | rows are independent |
| `Stratified(labels)` | observations within each stratum, keeping stratum sizes | strata are fixed by design |
| `Clustered(labels)` | whole clusters, observations kept together | rows of one cluster are related |
| `Block(length)` | blocks of consecutive observations | the rows are a time series |

The interval method is `percentile` (the quantiles of the replicates), `basic` (the quantiles reflected
around the estimate) or `bca` (corrected for bias and skew, with the acceleration from a jackknife).
`bca` needs `IID` or `Clustered`, since the jackknife drops one observation or one cluster.

## What it assumes

- Proportion: the trials are independent and share one success probability.
- Effect size: two independent samples. The analytic interval assumes about normal data and, for the
  pooled standard deviation, equal variances.
- Bootstrap: the unit you name is the unit that is independent. A bootstrap over rows when rows come in
  clusters gives an interval that is too narrow (see the example).
- Bootstrap with few observations: the replicates only show what the sample shows. Few observations or a
  statistic that depends on the extremes make the percentile interval unreliable.

## What the library checks and does not check

Checked: counts are integers with 0 <= successes <= trials, each group has at least 2 finite
observations and the two groups are not both constant, the analytic interval is asked only for
Cohen's d or Hedges' g, the level is in (0, 1), the method is known, the data are not empty, labels
match the length of the data, a block length is in [1, n], at least 2 clusters, and `bca` is not
asked with `Stratified` or `Block`.

Not checked:

- That the unit you named is the right one. The result records it, and nothing tests it.
- That the data are about normal, for the analytic effect interval.
- That a bootstrap interval has its stated coverage for your statistic and sample size.

## Example

The data are simulated. The first part shows the proportion intervals for 0 successes in 10 trials. The
second compares the analytic and bootstrap intervals of Hedges' g on skewed data. The third resamples 200
rows that come in 20 groups of 10, once by row and once by group.

```python
import numpy as np
from statsx.bootstrap import Clustered, IID, bootstrap
from statsx.effect import effect_size
from statsx.proportion import proportion_interval

for method in ("wilson", "jeffreys", "clopper_pearson", "agresti_coull"):
    e = proportion_interval(0, 10, method=method)
    print(f"{method:16s} 0 of 10: [{e.interval.lower:.3f}, {e.interval.upper:.3f}]")

rng = np.random.default_rng(0)
a = rng.lognormal(0.5, 1.0, 25)
b = rng.lognormal(0.0, 1.0, 25)
for method in ("analytic", "bootstrap"):
    e = effect_size(a, b, method=method)
    print(f"effect {method:9s} g {e.estimate:.2f} [{e.interval.lower:.2f}, {e.interval.upper:.2f}]")

groups = np.repeat(np.arange(20), 10)
data = rng.normal(0.0, 1.0, 20)[groups] + rng.normal(0.0, 0.5, 200)
for name, unit in (("rows", IID()), ("groups", Clustered(groups))):
    r = bootstrap(data, lambda x: float(np.mean(x)), unit, seed=1)
    width = r.estimate.interval.upper - r.estimate.interval.lower
    print(f"mean, resampling {name:6s} interval width {width:.2f}")
```

```text
wilson           0 of 10: [0.000, 0.278]
jeffreys         0 of 10: [0.000, 0.217]
clopper_pearson  0 of 10: [0.000, 0.308]
agresti_coull    0 of 10: [0.000, 0.321]
effect analytic  g 0.01 [-0.53, 0.56]
effect bootstrap g 0.01 [-0.53, 0.57]
mean, resampling rows   interval width 0.31
mean, resampling groups interval width 0.87
```

With no successes in 10 trials the upper ends range from 0.22 (Jeffreys) to 0.32 (Agresti-Coull). The
two effect intervals agree here. Resampling by row gives an interval about a third as wide as resampling
by group, because the rows of one group share a level. The wider one is right for this data.

## Checked in the tests

- Each proportion interval matches reference values, the ends are set at 0 and at all successes, and the
  small-sample coverage claims above are true at 10 trials.
- Cohen's d matches its definition, and Hedges' g is d times \(J\).
- The analytic effect interval covers the true effect at about the nominal rate, and the bootstrap one
  agrees with it.
- The percentile bootstrap covers the mean at about the nominal rate with IID data. Stratified
  resampling keeps the stratum sizes, clustered resampling keeps clusters together, and block
  replicates are runs of consecutive rows.
- The basic and BCa intervals follow their definitions. BCa with clusters drops one cluster at a time.
- The same seed gives the same result.

## References

- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal
  of the American Statistical Association* 22(158):209-212. <https://doi.org/10.1080/01621459.1927.10502953>
- Clopper, C. J., and Pearson, E. S. (1934). The use of confidence or fiducial limits illustrated in the
  case of the binomial. *Biometrika* 26(4):404-413. <https://doi.org/10.1093/biomet/26.4.404>
- Agresti, A., and Coull, B. A. (1998). Approximate is better than "exact" for interval estimation of
  binomial proportions. *The American Statistician* 52(2):119-126.
  <https://doi.org/10.1080/00031305.1998.10480550>
- Brown, L. D., Cai, T. T., and DasGupta, A. (2001). Interval estimation for a binomial proportion.
  *Statistical Science* 16(2):101-133. <https://doi.org/10.1214/ss/1009213286>
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*, 2nd ed. Lawrence Erlbaum.
- Hedges, L. V. (1981). Distribution theory for Glass's estimator of effect size and related estimators.
  *Journal of Educational Statistics* 6(2):107-128. <https://doi.org/10.3102/10769986006002107>
- Hedges, L. V., and Olkin, I. (1985). *Statistical Methods for Meta-Analysis*. Academic Press.
  <https://doi.org/10.1016/C2009-0-03396-0>
- Efron, B. (1979). Bootstrap methods: another look at the jackknife. *Annals of Statistics* 7(1):1-26.
  <https://doi.org/10.1214/aos/1176344552>
- Efron, B. (1987). Better bootstrap confidence intervals. *Journal of the American Statistical
  Association* 82(397):171-185. <https://doi.org/10.1080/01621459.1987.10478410>
- Efron, B., and Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman and Hall.
  <https://doi.org/10.1201/9780429246593>
- Davison, A. C., and Hinkley, D. V. (1997). *Bootstrap Methods and their Application*. Cambridge
  University Press. <https://doi.org/10.1017/CBO9780511802843>
- Kunsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *Annals of
  Statistics* 17(3):1217-1241. <https://doi.org/10.1214/aos/1176347265>
