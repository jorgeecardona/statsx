# Calibration of predicted probabilities

Use this to check whether the probabilities of a classifier are right, and to fix them when they are
not. The module is `statsx.calibration`.

A classifier is calibrated when, among the items it gives probability \(p\), a share \(p\) are
positive. A model can rank items well and still be badly calibrated.

## Measure it

`reliability(y_true, p)` bins the predicted probabilities and compares the mean prediction in each bin
with the share of positives. These are the numbers of a reliability diagram (DeGroot and Fienberg 1983).
From the same bins it gives:

- ECE, the expected calibration error: the gaps between the two, weighted by the size of the bin. MCE is
  the largest gap among bins that hold items (Naeini et al. 2015).
- The Brier score and its decomposition (Murphy 1973) into reliability (the calibration error),
  resolution (how much the bins separate positives from negatives) and uncertainty (the variance of the
  outcome). With bins there are two more terms for the spread of the predictions inside a bin
  (Stephenson et al. 2008). They are zero when every bin holds one predicted value. The identity is exact:

\[
\text{Brier} = \text{reliability} - \text{resolution} + \text{uncertainty} + \text{within-bin} .
\]

`strategy="uniform"` cuts bins of equal width on [0, 1]. `strategy="quantile"` cuts bins with about the
same number of items.

ECE depends on the number of bins and on how they are cut. It is biased upward with few items per bin.
For a perfectly calibrated classifier with 10 bins, the mean ECE is about 0.10 at 100 items, 0.03 at
1000 and 0.01 at 10000 (see the example). Read ECE with the bin counts, and the Brier score as the
overall check.

## Fix it

Both fixes learn a map from the scores of the classifier to probabilities.

- `fit_platt` is a sigmoid of the score, with 2 parameters (Platt 1999; Lin, Lin and Weng 2007). It is
  stable with few items. It can only stretch and shift the logit of the score.
- `fit_isotonic` is any non-decreasing curve, by pool adjacent violators (Zadrozny and Elkan 2002). It
  needs more items. Between two training scores the map is linear, and outside them it is constant.

When the miscalibration is a sigmoid of the logit, as with an overconfident classifier, isotonic
regression had a worse test Brier score than Platt scaling with 40 and 200 training items in the
tests, and the same from 1000.

Fit the map on data the classifier was not trained on. Check the result on a third set.

## What it assumes

- The items are independent, and the labels are 0 or 1.
- The calibration data look like the data you will score. A map fitted on one population is wrong on
  another one.
- Platt scaling: the miscalibration is a sigmoid of the score. Isotonic regression: the true
  probability never decreases as the score rises.
- ECE and MCE depend on the binning. They are not properties of the classifier alone.

## What the library checks and does not check

Checked: the arrays are non-empty and of one length, labels are 0 or 1, probabilities are in [0, 1],
scores are finite, `n_bins` is at least 1, `strategy` is known, and both classes are present when you
fit a map.

Not checked:

- That the calibration set was not used to train the classifier.
- That a bin holds enough items. A bin with 3 items has a noisy rate. Read `counts`.
- That the map still holds after the data change. Check calibration again on new data.

## Example

The data are simulated. The scores are logits of an overconfident classifier: the true probability is
the sigmoid of half the logit. The maps are fitted on 400 items and checked on 2000 new items.

```python
import numpy as np
from statsx.calibration import fit_isotonic, fit_platt, reliability

rng = np.random.default_rng(0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def draw(n):
    logit = rng.normal(0.0, 2.0, n)
    return logit, (rng.random(n) < sigmoid(0.5 * logit)).astype(float)


train, y_train = draw(400)
test, y_test = draw(2000)
raw = reliability(y_test, sigmoid(test))
print(f"raw      ECE {raw.ece:.2f}  Brier {raw.brier.brier:.3f}")
maps = {"platt": fit_platt(train, y_train), "isotonic": fit_isotonic(train, y_train)}
for name, fitted in maps.items():
    r = reliability(y_test, fitted.predict(test))
    print(f"{name:8s} ECE {r.ece:.2f}  Brier {r.brier.brier:.3f}")

for n in (100, 1000, 10000):
    eces = []
    for _ in range(20):
        p = rng.random(n)
        eces.append(reliability((rng.random(n) < p).astype(float), p).ece)
    print(f"calibrated classifier, {n} items: mean ECE {np.mean(eces):.2f}")
```

```text
raw      ECE 0.10  Brier 0.214
platt    ECE 0.03  Brier 0.205
isotonic ECE 0.05  Brier 0.208
calibrated classifier, 100 items: mean ECE 0.10
calibrated classifier, 1000 items: mean ECE 0.03
calibrated classifier, 10000 items: mean ECE 0.01
```

Both maps repair the classifier. Platt scaling has the better Brier score here, as the miscalibration is
a sigmoid of the logit. The last lines show the upward bias of ECE with few items.

## Checked in the tests

- The reliability numbers are the bin means, and ECE is their weighted gap.
- The Brier score equals its decomposition exactly, and the within-bin terms vanish with one value per
  bin.
- A calibrated classifier has a small ECE and an overconfident one a large ECE. Quantile bins hold about
  the same number of items.
- Platt scaling and isotonic regression repair an overconfident classifier. The isotonic map never
  decreases and stays a probability, and it is the best non-decreasing fit.
- Platt scaling is better than isotonic regression with few items.

## References

- DeGroot, M. H., and Fienberg, S. E. (1983). The comparison and evaluation of forecasters. *The
  Statistician* 32(1/2):12-22. <https://doi.org/10.2307/2987588>
- Murphy, A. H. (1973). A new vector partition of the probability score. *Journal of Applied Meteorology*
  12(4):595-600. <https://doi.org/10.1175/1520-0450(1973)012<0595:ANVPOT>2.0.CO;2>
- Stephenson, D. B., Coelho, C. A. S., and Jolliffe, I. T. (2008). Two extra components in the Brier
  score decomposition. *Weather and Forecasting* 23(4):752-757. <https://doi.org/10.1175/2007WAF2006116.1>
- Platt, J. C. (1999). Probabilistic outputs for support vector machines and comparisons to regularized
  likelihood methods. In *Advances in Large Margin Classifiers*, 61-74. MIT Press.
- Lin, H.-T., Lin, C.-J., and Weng, R. C. (2007). A note on Platt's probabilistic outputs for support
  vector machines. *Machine Learning* 68(3):267-276. <https://doi.org/10.1007/s10994-007-5018-6>
- Zadrozny, B., and Elkan, C. (2002). Transforming classifier scores into accurate multiclass probability
  estimates. *Proceedings of KDD 2002*, 694-699. <https://doi.org/10.1145/775047.775151>
- Barlow, R. E., Bartholomew, D. J., Bremner, J. M., and Brunk, H. D. (1972). *Statistical Inference under
  Order Restrictions*. Wiley.
- Naeini, M. P., Cooper, G. F., and Hauskrecht, M. (2015). Obtaining well calibrated probabilities using
  Bayesian binning into quantiles. *Proceedings of the AAAI Conference on Artificial Intelligence* 29(1).
