# Tests for paired results, equivalence and many tests

Use this to test whether two systems differ on the same items, to show that they are the same in
practice, or to correct many tests at once. The modules are `statsx.paired`, `statsx.equivalence` and
`statsx.multiple`.

## Paired results

Paired means each item was measured twice: two systems on the same items, or one system before and
after. The tests look at the differences, so the spread between items does not enter.

| Function | Input | Uses | Weak when |
|---|---|---|---|
| `mcnemar` | right or wrong, per item and system | only the items where the systems disagree | most items agree: few items count |
| `sign_test` | a score per item and system | the direction of each difference | the size of the differences matters |
| `wilcoxon` | a score per item and system | direction and size (signed ranks) | the differences are far from symmetric |
| `bootstrap_difference` | a score per item and system | the mean difference, resampled in pairs | very few items |

`mcnemar` is a binomial test of \(b\) out of \(b + c\) against one half, where \(b\) counts the items only
A gets right and \(c\) the items only B gets right (McNemar 1947). `exact=False` gives the chi-square
form with the continuity correction of Edwards (1948). `wilcoxon` drops zero differences. Its p-value is
exact for up to 50 non-zero differences, with ties, and a normal approximation with a tie correction
above that.

Turning a score into right or wrong throws away information. The example shows it.

## Equivalence

A test of "no difference" cannot show that two systems are the same. A small sample gives a wide
interval, and a wide interval proves nothing. An `EquivalenceMargin` states which differences count as
practically zero, and `tost` tests that the difference lies inside it (Schuirmann 1987).

Two one-sided t tests reject "the difference is at or below the lower margin" and "at or above the upper
margin". Both must reject, so the p-value is the larger of the two. This is the same as the
\(1 - 2\alpha\) interval of the difference lying inside the margin. `rope_probability` is the Bayesian
counterpart: the share of posterior draws of the difference inside the margin (Kruschke 2018). One
margin serves both, so the two answers mean the same thing.

Failing to show equivalence does not show a difference.

## Many tests

`adjust` takes the p-values of \(m\) tests and returns the adjusted values in the input order and which
tests are rejected at level `alpha`.

| Method | Controls | Needs |
|---|---|---|
| `holm` | the chance of any false rejection (family-wise error) | nothing about the dependence |
| `bh` | the expected share of false rejections among the rejections | independent or positively dependent tests |
| `storey` | the same, scaled by an estimate \(\hat\pi_0\) of the share of true nulls | many tests |

`storey` rejects more than `bh` when many tests have a real effect. It estimates \(\hat\pi_0\) from the
p-values above a threshold `lambda_` (default 0.5), capped at 1. With a handful of tests that estimate is
unstable, as the example shows.

## What it assumes

- The items are independent. Two rows from one prompt, one user or one session break every p-value here.
- Paired data: the pair is the unit. The two measurements of an item are the only link between rows.
- `mcnemar` and `sign_test`: under no difference each item favours A or B with probability one half.
- `wilcoxon`: the differences are centred on zero under no difference. The test is strongest when they are
  roughly symmetric.
- `tost`: the differences are independent and about normal. The margin is a choice you make from the
  problem, not from the data.
- `holm`: nothing about the dependence. `bh`: independent or positively dependent tests. `storey`: a
  large number of tests, so that \(\hat\pi_0\) is stable.

## What the library checks and does not check

Checked: paired arrays are non-empty, finite and of one length; right-or-wrong arrays are 0 or 1; each
group has at least 2 finite observations and the data have spread; `alpha` is in range; the margin
contains 0; p-values are numbers in [0, 1]; the method is one of the three.

Not checked:

- That the items are independent.
- That the margin is a sensible size. A margin wider than any real difference makes everything
  equivalent.
- That the tests of one family are the family you meant. The correction counts the p-values you pass.
- That the tests are positively dependent, for `bh`, or numerous enough, for `storey`.

## Example

The data are simulated. System A is better than B by 0.4 on a score. A score above 0.3 counts as
right. The example runs each test.

```python
import numpy as np
from statsx.equivalence import tost
from statsx.multiple import adjust
from statsx.paired import bootstrap_difference, mcnemar, sign_test, wilcoxon

rng = np.random.default_rng(0)
difficulty = rng.normal(0.0, 1.0, 80)
a = difficulty + rng.normal(0.4, 0.5, 80)
b = difficulty + rng.normal(0.0, 0.5, 80)

print("McNemar on right or wrong:", round(mcnemar(a > 0.3, b > 0.3).p_value, 2))
print("sign test:", round(sign_test(a, b).p_value, 3))
print("Wilcoxon below 0.001:", wilcoxon(a, b).p_value < 0.001)
diff = bootstrap_difference(a, b)
low, high = round(diff.interval.lower, 2), round(diff.interval.upper, 2)
print("mean difference:", round(diff.estimate, 2), "interval:", low, high)
print("equivalent within 0.2:", tost(a, b, 0.2, paired=True).equivalent)

p = np.array([0.001, 0.008, 0.039, 0.041, 0.2, 0.7])
for method in ("holm", "bh", "storey"):
    print(method, "rejects", adjust(p, method=method).n_rejected)
```

```text
McNemar on right or wrong: 0.21
sign test: 0.002
Wilcoxon below 0.001: True
mean difference: 0.41 interval: 0.25 0.56
equivalent within 0.2: False
holm rejects 2
bh rejects 2
storey rejects 4
```

McNemar on right or wrong has a large p-value because only a few items disagree. The sign test and
Wilcoxon, which use the scores, find the difference. The two groups are not equivalent within 0.2. With
6 p-values, `storey` rejects 4 where `holm` and `bh` reject 2, because \(\hat\pi_0\) comes from a single
p-value above the threshold. Use `storey` with many tests.

## Checked in the tests

- Holm and Benjamini-Hochberg reject what the example of Benjamini and Hochberg (1995) says, and the
  adjusted values match a brute-force definition.
- Results come back in the input order, and adjusted values never shrink.
- Benjamini-Hochberg keeps the false discovery rate below `alpha`, and Holm the family-wise error rate,
  in simulation.
- The sign test matches the binomial sum, and McNemar's exact test is a binomial test on the
  disagreements. The chi-square form matches the textbook value.
- Wilcoxon matches the enumeration of all sign assignments, with and without ties, and the example of
  Hollander and Wolfe. The normal approximation agrees with the exact test near the limit of 50.
- The paired tests keep their size and have power in simulation.
- TOST is the same as the \(1 - 2\alpha\) interval lying inside the margin, and the error rate at the
  edge of the margin is at most `alpha`.
- One margin gives the TOST test and the ROPE probability.

## References

- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or
  percentages. *Psychometrika* 12(2):153-157. <https://doi.org/10.1007/BF02295996>
- Edwards, A. L. (1948). Note on the "correction for continuity" in testing the significance of the
  difference between correlated proportions. *Psychometrika* 13(3):185-187.
  <https://doi.org/10.1007/BF02289261>
- Dixon, W. J., and Mood, A. M. (1946). The statistical sign test. *Journal of the American Statistical
  Association* 41(236):557-566. <https://doi.org/10.1080/01621459.1946.10501898>
- Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics Bulletin* 1(6):80-83.
  <https://doi.org/10.2307/3001968>
- Hollander, M., Wolfe, D. A., and Chicken, E. (2014). *Nonparametric Statistical Methods*, 3rd ed.
  Wiley. <https://doi.org/10.1002/9781119196037>
- Schuirmann, D. J. (1987). A comparison of the two one-sided tests procedure and the power approach for
  assessing the equivalence of average bioavailability. *Journal of Pharmacokinetics and
  Biopharmaceutics* 15(6):657-680. <https://doi.org/10.1007/BF01068419>
- Lakens, D. (2017). Equivalence tests: a practical primer for t tests, correlations, and meta-analyses.
  *Social Psychological and Personality Science* 8(4):355-362. <https://doi.org/10.1177/1948550617697177>
- Kruschke, J. K. (2018). Rejecting or accepting parameter values in Bayesian estimation. *Advances in
  Methods and Practices in Psychological Science* 1(2):270-280. <https://doi.org/10.1177/2515245918771304>
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of
  Statistics* 6(2):65-70.
- Benjamini, Y., and Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful
  approach to multiple testing. *Journal of the Royal Statistical Society B* 57(1):289-300.
  <https://doi.org/10.1111/j.2517-6161.1995.tb02031.x>
- Storey, J. D. (2002). A direct approach to false discovery rates. *Journal of the Royal Statistical
  Society B* 64(3):479-498. <https://doi.org/10.1111/1467-9868.00346>
