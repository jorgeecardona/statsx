# Agreement between raters

Use this to measure how much two or more raters agree on the same items: people labelling data, or a
model judging against a person. The module is `statsx.agreement`.

All three measures compare the agreement seen with the agreement expected by chance. 1 is perfect
agreement, 0 is the level of chance, and a negative value is worse than chance. Each function returns an
`Estimate` with a percentile bootstrap interval that resamples the items.

| Function | Raters | Ratings | Use when |
|---|---|---|---|
| `cohen_kappa` | two | labels | two raters rate every item. `weights="linear"` or `"quadratic"` makes a near miss count less than a far miss, for ordered categories |
| `fleiss_kappa` | any fixed number | counts of each category per item | the same number of raters rates every item, nominal categories |
| `krippendorff_alpha` | any number | a matrix (raters, units), missing allowed | ratings are missing, or the scale is ordinal, interval or ratio |

For `krippendorff_alpha`, a unit needs at least two ratings, and `level_of_measurement` is
`nominal`, `ordinal`, `interval` or `ratio`.

Kappa is low when one category is very common, even with high raw agreement. Read it with the category
shares (see the example).

## What it assumes

- The items are independent, and the raters rate each item without seeing each other.
- Cohen and Fleiss: one chance level for all items, set by the category shares. Krippendorff: the
  difference function of the level of measurement.
- The items are a sample of the items you care about. The interval resamples items, not raters.
- The raters are the raters you will use. With 2 or 3 raters the result says little about raters in
  general.

## What the library checks and does not check

Checked: both raters have the same number of items, and at least 2; every item has the same number of
raters in `fleiss_kappa`, at least 2; counts are non-negative whole numbers with at least 2 categories;
ratings have the shape (raters, units); weights need numeric labels; a measure that is undefined (one
category only, fewer than 2 units with 2 ratings) is refused.

Not checked:

- That the raters did not influence each other.
- That a high kappa means the labels are right. Raters can agree and be wrong together.
- That the categories are well defined. Vague categories give a low kappa that more raters will not fix.

## Example

The data are simulated: two raters label 100 items in 3 categories, and the second rater copies the first
70% of the time. A third rater is added with 20% of the ratings missing. The last lines show the paradox
of kappa: 90% raw agreement with one very common category.

```python
import numpy as np
from statsx.agreement import cohen_kappa, krippendorff_alpha

rng = np.random.default_rng(0)
a = rng.integers(0, 3, 100)
b = np.where(rng.random(100) < 0.7, a, rng.integers(0, 3, 100))
c = np.where(rng.random(100) < 0.7, a, rng.integers(0, 3, 100)).astype(float)
c[rng.random(100) < 0.2] = np.nan

kappa = cohen_kappa(a, b)
low, high = round(kappa.interval.lower, 2), round(kappa.interval.upper, 2)
print("Cohen kappa:", round(kappa.estimate, 2), "interval", (low, high))
alpha = krippendorff_alpha(np.vstack([a, b, c]).astype(float))
low, high = round(alpha.interval.lower, 2), round(alpha.interval.upper, 2)
print("Krippendorff alpha:", round(alpha.estimate, 2), "interval", (low, high))

x = np.array([1] * 90 + [0] * 10)
y = np.array([1] * 85 + [0] * 5 + [1] * 5 + [0] * 5)
print("raw agreement:", (x == y).mean(), "kappa:", round(cohen_kappa(x, y).estimate, 2))
```

```text
Cohen kappa: 0.52 interval (0.37, 0.65)
Krippendorff alpha: 0.49 interval (0.38, 0.6)
raw agreement: 0.9 kappa: 0.44
```

The rater copies the first one 70% of the time and guesses otherwise, so raw agreement is about 80%. Chance
agreement for 3 equal categories is 1/3, so kappa and alpha are about 0.5, not 0.8. In the last example 90% of
the items agree, but category 1 is so common that chance alone would agree on most of them, so kappa is 0.44.

## Checked in the tests

- Cohen's kappa matches the textbook example, and weighted kappa gives partial credit to near misses.
- Fleiss' kappa matches the worked example and agrees with Cohen's kappa for two raters.
- Krippendorff's alpha matches the reference values with missing ratings, at each level, and follows its
  definition over pairs of values.
- Perfect agreement gives 1 and random ratings give about 0. The interval narrows with more items.

## References

- Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological
  Measurement* 20(1):37-46. <https://doi.org/10.1177/001316446002000104>
- Cohen, J. (1968). Weighted kappa: nominal scale agreement with provision for scaled disagreement or
  partial credit. *Psychological Bulletin* 70(4):213-220. <https://doi.org/10.1037/h0026256>
- Fleiss, J. L. (1971). Measuring nominal scale agreement among many raters. *Psychological Bulletin*
  76(5):378-382. <https://doi.org/10.1037/h0031619>
- Krippendorff, K. (2011). Computing Krippendorff's alpha-reliability. Departmental Papers (ASC),
  University of Pennsylvania. <https://repository.upenn.edu/asc_papers/43>
- Krippendorff, K. (2016). Bootstrapping distributions for Krippendorff's alpha.
  <https://www.asc.upenn.edu/sites/default/files/2021-03/Alphabootstrap.pdf>
