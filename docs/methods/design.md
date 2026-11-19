# Plan a run: power, sample size and allocation

Use this before you collect data, to decide how many items you need and how to split them. The modules
are `statsx.design` and `statsx.allocation`.

## Power, sample size and the smallest effect

Every test has three functions. Each returns a `Design` with the test, `n`, the effect, the power and
`alpha`.

| Function | You give | You get |
|---|---|---|
| `power_*` | the effect and `n` | the chance the test finds the effect |
| `sample_size_*` | the effect and the power | the smallest `n` that reaches it |
| `minimum_detectable_effect_*` | `n` and the power | the smallest effect that `n` finds with that power |

The tests covered:

| Test | Suffix | `n` is | Power |
|---|---|---|---|
| Paired differences, or one sample against a value | `_paired` | the number of pairs | exact, noncentral t |
| Two groups of means | `_two_sample` | the items in each group | exact, noncentral t |
| Two proportions | `_two_proportions` | the items in each group | normal approximation (Fleiss, Levin and Paik 2003); `exact=True` counts every binomial outcome |
| Equivalence (TOST) of a paired difference, two means, two proportions | `_equivalence_*` | as above | exact for means (Owen 1965), normal approximation for proportions (Blackwelder 1982) |

For `_paired`, `effect` is the mean difference and `sd` is the standard deviation of the differences.
For `_two_sample`, `effect` is the difference of the means and `sd` the standard deviation within a
group. Exact power for two proportions is not a smooth function of `n`, so
`sample_size_two_proportions` gives the `n` of the approximation.

Equivalence tests take an `EquivalenceMargin`, the same object as `statsx.equivalence` (see
[the tests page](tests.md)).

## Allocation over strata

An evaluation set often has strata: classes, languages, difficulty levels. `stratified_allocation`
finds the smallest total sample size whose split between strata reaches a target margin of error. The
variance of the combined mean is

\[
\sum_h W_h^2 S_h^2 \Bigl(\frac{1}{n_h} - \frac{1}{N_h}\Bigr), \qquad W_h = N_h / N ,
\]

so the split changes the precision.

- `"neyman"`: \(n_h\) proportional to \(N_h S_h\). Larger or more variable strata get more items. This is
  the smallest variance for a fixed total (Neyman 1934). With a cost \(c_h\) per item, \(n_h\) is
  proportional to \(N_h S_h / \sqrt{c_h}\).
- `"proportional"`: \(n_h\) proportional to \(N_h\).

For a proportion, give the stratum proportions \(p_h\) and the standard deviation is
\(\sqrt{p_h (1 - p_h)}\). A stratum is capped at its size, and the rest is spread over the other strata.

## What it assumes

- The planning inputs are guesses: the effect, the spread, the baseline rate, the stratum variances.
  The power holds only if they are right. Use a cautious spread.
- The test you plan is the test you run. The power of a t test is not the power of a rank test.
- The items are independent. Clusters need more items than the formulas give.
- Allocation: the strata are fixed by design, and the standard deviation of each stratum is known
  before you sample.

## What the library checks and does not check

Checked: `n` is a whole number of at least the minimum, `alpha` is in (0, 0.5), the power is in (0, 1),
`sd` is above 0, the effect is not 0 where that is needed, the true difference is inside the margin for
equivalence, and a sample size above 10 million is refused. For allocation: sizes and spreads have one
value per stratum, each stratum has at least 2 items, costs are above 0, and the margin is above 0.

Not checked:

- That your planning effect and `sd` are the true ones.
- That the data will follow the model of the test (normal differences for the t test).
- That the strata you chose are the ones that matter.

## Example

A two-group comparison with an effect of 0.5 and a standard deviation of 1, then the same plan with a
larger `sd`. Then two proportions, and an allocation of 10000 items over three strata.

```python
from statsx.allocation import stratified_allocation
from statsx.design import (
    minimum_detectable_effect_two_sample,
    power_two_sample,
    sample_size_two_proportions,
    sample_size_two_sample,
)

plan = sample_size_two_sample(effect=0.5, sd=1.0, power=0.8)
print("items per group:", plan.n, "power", round(plan.power, 3))
print("power with 40 per group:", round(power_two_sample(0.5, 1.0, 40).power, 2))
smallest = minimum_detectable_effect_two_sample(1.0, 100).effect
print("smallest effect with 100 per group:", round(smallest, 2))
print("if the sd is 1.5, items per group:", sample_size_two_sample(effect=0.5, sd=1.5).n)
print("proportions 0.5 and 0.6, items per group:", sample_size_two_proportions(0.5, 0.6).n)

for method in ("neyman", "proportional"):
    a = stratified_allocation(
        [5000, 3000, 2000], margin=0.03, proportions=[0.5, 0.2, 0.1], method=method
    )
    print(f"{method:12s} sizes {a.sizes} total {a.total}")
```

```text
items per group: 64 power 0.801
power with 40 per group: 0.6
smallest effect with 100 per group: 0.4
if the sd is 1.5, items per group: 143
proportions 0.5 and 0.6, items per group: 388
neyman       sizes (424, 204, 102) total 730
proportional sizes (377, 226, 151) total 754
```

The effect 0.5 with `sd` 1 is Cohen's medium effect, and 64 per group is the classic answer. A spread of
1.5 instead of 1 needs more than twice as many items. Neyman allocation needs fewer items than
proportional allocation for the same margin, because the strata with proportions near 0.5 vary more.

## Checked in the tests

- The power matches Cohen's classic sample sizes, and a simulation of the two-sample test has the
  analytic power.
- The paired power matches a published value and a simulation.
- A one-sided test has more power than a two-sided one, and no effect gives `alpha`.
- Two proportions match the Fleiss formula and an exact count.
- Equivalence power matches a simulation and the TOST test, and the sample size reaches the power.
- A sample size is the smallest that reaches the power, and the minimum detectable effect reaches the
  power at that effect and shrinks with `n`.
- Neyman allocation follows \(n_h \propto N_h S_h\), is the smallest total that meets the margin, and needs
  fewer items than proportional allocation for the same margin.
- A small stratum is capped at its size.

## References

- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*, 2nd ed. Lawrence Erlbaum.
- Fleiss, J. L., Levin, B., and Paik, M. C. (2003). *Statistical Methods for Rates and Proportions*, 3rd
  ed. Wiley. <https://doi.org/10.1002/0471445428>
- Schuirmann, D. J. (1987). A comparison of the two one-sided tests procedure and the power approach for
  assessing the equivalence of average bioavailability. *Journal of Pharmacokinetics and
  Biopharmaceutics* 15(6):657-680. <https://doi.org/10.1007/BF01068419>
- Blackwelder, W. C. (1982). "Proving the null hypothesis" in clinical trials. *Controlled Clinical
  Trials* 3(4):345-353. <https://doi.org/10.1016/0197-2456(82)90024-1>
- Owen, D. B. (1965). A special case of a bivariate non-central t-distribution. *Biometrika*
  52(3-4):437-446. <https://doi.org/10.1093/biomet/52.3-4.437>
- Neyman, J. (1934). On the two different aspects of the representative method: the method of stratified
  sampling and the method of purposive selection. *Journal of the Royal Statistical Society*
  97(4):558-625. <https://doi.org/10.2307/2342192>
- Cochran, W. G. (1977). *Sampling Techniques*, 3rd ed., section 5.5. Wiley.
