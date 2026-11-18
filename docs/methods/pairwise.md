# Rank systems from pairwise results

Use this when a judge (a person, or a model) compares two systems on the same item and says which is
better, or that they tie, and you want one strength per system with intervals. The module is
`statsx.pairwise`.

## The data

One row per comparison: the two systems, the outcome (the first wins, the second wins, or a tie),
and the item (the prompt, the task, the match). Rows of the same item are resampled together, as
explained below. This is the `Comparisons` table.

## Bradley-Terry

Give system \(i\) a strength \(\theta_i\). With \(d = \theta_a - \theta_b\), the chance that \(a\) beats
\(b\) is

\[
\mathbb{P}(a \succ b) = \frac{e^{\theta_a}}{e^{\theta_a} + e^{\theta_b}} = \sigma(d) .
\]

With `ties="half"`, a tie counts as half a win and half a loss: a row with outcome \(y \in \{1, \tfrac12, 0\}\)
adds \(y \log\sigma(d) + (1 - y)\log\sigma(-d)\) to the log likelihood.

## Davidson: ties as an outcome

Davidson (1970) gives a tie its own probability. With \(Z = e^{d/2} + e^{-d/2} + \nu\) and a tie
parameter \(\nu \ge 0\),

\[
\mathbb{P}(a \succ b) = \frac{e^{d/2}}{Z}, \qquad
\mathbb{P}(b \succ a) = \frac{e^{-d/2}}{Z}, \qquad
\mathbb{P}(\text{tie}) = \frac{\nu}{Z} .
\]

With \(\nu = 0\) this is the Bradley-Terry model. The tie probability is largest for equal systems
(\(\nu / (2 + \nu)\) at \(d = 0\)) and falls as the systems move apart. With \(\tau = \log\nu\), a row adds
\(\tfrac{d}{2} - \log Z\) (first wins), \(-\tfrac{d}{2} - \log Z\) (second wins) or \(\tau - \log Z\) (tie).

## The fit

The strengths have a Gaussian prior \(\theta_i \sim \mathcal{N}(0, s^2)\) (`prior_sd`, default 3), and
\(\tau \sim \mathcal{N}(0, 3^2)\). The prior keeps a strength finite for a system that never loses, and
fixes the mean of the strengths at 0. Newton's method maximizes the log posterior, with the exact
gradient and Hessian. With \(g_r = \partial \ell_r / \partial d\), the gradient of strength \(i\) is
\(\sum_{r: a_r = i} g_r - \sum_{r: b_r = i} g_r - \theta_i/s^2\), and the second derivatives of a row
enter the Hessian in the same pattern, with \(\partial^2 \ell / \partial d^2 = -\big(\tfrac{\cosh(d/2)}{2Z}
- \tfrac{\sinh^2(d/2)}{Z^2}\big)\) for Davidson's model.

For two systems the fit has a closed form. With shares \(f_w, f_l, f_t\) of wins, losses and ties and a flat prior,

\[
\hat d = \log\frac{f_w}{f_l}, \qquad
\hat\nu = \frac{f_t\,(e^{\hat d/2} + e^{-\hat d/2})}{1 - f_t} .
\]

## Intervals: resample the items

Rows of one item are not independent: the same prompt judged for several pairs, or by one judge in one
mood. Treating the rows as independent gives intervals that are too narrow. The package resamples
the **items** with replacement, and refits each time (as a weight per row, the number of times its
item was drawn). The percentile interval of the resampled strengths is the interval
(Efron 1979; for whole clusters, Field and Welsh 2007).

`p_better(a, b)` is the share of resamples in which \(\theta_a > \theta_b\). It is a bootstrap frequency,
not a posterior probability.

## Example

The data are simulated: four systems with true strengths 0.9, 0.3, -0.3 and -0.9, a tie parameter
\(\nu = 0.5\), every pair judged on each of 40 prompts.

```python
import math
import numpy as np
from statsx.pairwise import Comparisons, fit_bradley_terry

theta = np.array([0.9, 0.3, -0.3, -0.9])
names = ["frontier", "large", "medium", "small"]
nu = 0.5
rng = np.random.default_rng(0)
first, second, outcome, prompt = [], [], [], []
for p in range(40):
    for i in range(4):
        for j in range(i + 1, 4):
            d = theta[i] - theta[j]
            u = rng.random() * (2 * math.cosh(d / 2) + nu)
            win, loss = math.exp(d / 2), math.exp(-d / 2)
            first.append(names[i])
            second.append(names[j])
            outcome.append(1.0 if u < win else 0.0 if u < win + loss else 0.5)
            prompt.append(p)

fit = fit_bradley_terry(Comparisons.from_long(first, second, outcome, item=prompt), seed=1)
for name, value in zip(fit.systems, fit.strength, strict=True):
    low, high = fit.strength_interval(name, 0.9).lower, fit.strength_interval(name, 0.9).upper
    print(f"{name:9s} {value:5.2f}  90% [{low:5.2f}, {high:5.2f}]")
print("tie parameter", round(fit.tie_parameter, 2))
print("P(large above medium) =", round(fit.p_better("large", "medium"), 3))
print("frontier vs small (win, loss, tie):", [round(p, 2) for p in fit.probabilities("frontier", "small")])
```

```text
frontier   1.02  90% [ 0.76,  1.27]
large      0.23  90% [-0.01,  0.47]
medium    -0.44  90% [-0.74, -0.15]
small     -0.80  90% [-1.08, -0.53]
tie parameter 0.72
P(large above medium) = 0.992
frontier vs small (win, loss, tie): [0.69, 0.11, 0.2]
```

## Checked in the tests

- The fitted point is the maximum of the penalized log likelihood: the gradient is 0, and any move
  that keeps the mean at 0 lowers the objective, for both models.
- Two systems give the closed-form fit above (65/100 wins with ties as half a win; for Davidson the
  win and tie probabilities 0.6, 0.3 and 0.1 are recovered).
- With 800 items, the strengths are within 0.08 of the truth and the tie parameter within 0.1.
- A system that always wins gets a finite strength.
- When rows of an item depend on each other, the item-level interval is on average wider than the
  row-level one (1.12 times in the test); with one row per item they are identical.
- The item-level 90% interval held the population value in about 82% of 80 runs with 20 items. The
  percentile bootstrap is a little narrow for few items: use more items before you trust a close call.

## When it fails

- A comparison graph that is not connected: systems in different groups are not on one scale. The
  prior sets them to 0 relative to their own group.
- Judges with their own biases (position, length): the model assumes one preference scale.
- Strengths do not capture intransitive preferences (A beats B, B beats C, C beats A).

## References

- Bradley, R. A., and Terry, M. E. (1952). Rank analysis of incomplete block designs: I. The method of
  paired comparisons. *Biometrika* 39(3/4):324-345. <https://doi.org/10.2307/2334029>
- Davidson, R. R. (1970). On extending the Bradley-Terry model to accommodate ties in paired comparison
  experiments. *Journal of the American Statistical Association* 65(329):317-328.
  <https://doi.org/10.1080/01621459.1970.10481082>
- Efron, B. (1979). Bootstrap methods: another look at the jackknife. *Annals of Statistics* 7(1):1-26.
  <https://doi.org/10.1214/aos/1176344552>
- Field, C. A., and Welsh, A. H. (2007). Bootstrapping clustered data. *Journal of the Royal Statistical
  Society B* 69(3):369-390. <https://doi.org/10.1111/j.1467-9868.2007.00593.x>
- Chiang, W.-L., et al. (2024). Chatbot Arena: an open platform for evaluating LLMs by human preference.
  *ICML*, PMLR 235. <https://proceedings.mlr.press/v235/chiang24b.html>
