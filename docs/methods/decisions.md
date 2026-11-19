# Decisions under uncertainty

Use this when a decision depends on inputs you do not know exactly: to see the spread of a cost, to find
the point where a decision changes, and to see how much more information is worth. The module is
`statsx.decision`.

## The distribution of a cost

`simulate_cost(model, inputs)` draws each uncertain input with a fixed seed, runs your cost model on all
draws at once, and returns a `CostDistribution` (Metropolis and Ulam 1949). An input is a distribution
with a `sample` method (`Beta`, `Gamma`, `Normal`, `LogNormal`) or a plain number, which is a constant.
The draws of the inputs are independent. Write the model with NumPy operations, because it is called once
with an array of draws for each input.

- `mean()` is the expected cost, with the interval of its Monte Carlo error (not a credible interval).
- `interval(level)` is the central interval that holds that share of the cost distribution.
- `quantiles(probs)` and `p_exceeds(limit)` give a quantile, and the probability that the cost is above
  a limit.

## Value of information

`expected_value_of_information(utilities)` takes draws of the utility of each action, shape
(draws, actions), and returns the expected value of perfect information (EVPI):

\[
\mathrm{EVPI} = \mathbb{E}\bigl[\max_a U_a\bigr] - \max_a \mathbb{E}[U_a] .
\]

It is what you would pay, at most, to learn the inputs before you choose (Raiffa and Schlaifer 1961;
Claxton 1999). It is 0 when one action is best in every draw. The result also gives the best action now
and the expected utility of each action.

## Break-even

- `break_even_probability(gain_if_right, loss_if_wrong)` is \(\text{loss} / (\text{gain} + \text{loss})\): acting
  pays when the probability that the state is good is above it.
- `break_even(net_benefit, low, high)` finds, by bisection, the threshold in `[low, high]` where a function
  changes sign: a price, a score cut-off, a probability.

## What it assumes

- The inputs are independent. `simulate_cost` draws each one on its own. Two inputs that move together
  (the number of requests and the tokens per request, in one workload) need a joint model.
- The input distributions are what you believe. The output is no better than they are.
- EVPI is for perfect information. Real information is partial, so the value of a real study is lower.
- The utility is the one you want to maximise, in one unit, for every action.

## What the library checks and does not check

Checked: `n_draws` is at least 2, there is at least one input, the model returns one finite cost per
draw, utilities are finite with at least 2 draws and 2 actions, the gain and the loss are above 0, and
`break_even` sees a sign change.

Not checked:

- That the inputs are independent, or that the cost model is right.
- That the distributions of the inputs are well chosen. A lognormal from an expert interval is only as good
  as the expert (see [priors](bayes.md)).
- That the Monte Carlo error is small. Read the interval of `mean()`.

## Example

The monthly cost of a service is requests times tokens per request times a price. The requests and the
tokens are uncertain. The second part chooses between doing nothing and launching, with a success rate
known only from 7 successes in 40 trials.

```python
import numpy as np
from statsx.conjugate import Beta, beta_binomial
from statsx.decision import (
    break_even,
    break_even_probability,
    expected_value_of_information,
    simulate_cost,
)
from statsx.prior import lognormal_from_interval


def monthly_cost(requests, tokens, price):
    return requests * tokens * price / 1e6


inputs = {
    "requests": lognormal_from_interval(50_000, 200_000),
    "tokens": lognormal_from_interval(500, 2000),
    "price": 3.0,
}
cost = simulate_cost(monthly_cost, inputs, seed=1)
low, high = cost.interval(0.9).lower, cost.interval(0.9).upper
print("expected cost:", round(cost.mean().estimate), "90% interval:", (round(low), round(high)))
print("P(cost above 500):", round(cost.p_exceeds(500.0), 2))

rate = beta_binomial(Beta(1.0, 1.0), 7, 40).sample(10000, seed=2)
utilities = np.column_stack([np.zeros(10000), rate * 100 - 15])  # do nothing, or launch
value = expected_value_of_information(utilities)
print("best action now:", value.best_action, "expected utilities:", np.round(value.expected_utility, 2))
print("EVPI:", round(value.evpi, 2))
print("break-even probability for gain 3, loss 1:", break_even_probability(3.0, 1.0))
print("same by bisection:", round(break_even(lambda p: 3 * p - (1 - p), 0.0, 1.0), 3))
```

```text
expected cost: 358 90% interval: (113, 796)
P(cost above 500): 0.2
best action now: 1 expected utilities: [0.   4.13]
EVPI: 0.77
break-even probability for gain 3, loss 1: 0.25
same by bisection: 0.25
```

The mean cost is 358 and the 90% interval is wide, 113 to 796: the uncertain inputs dominate. Launching has
the higher expected utility. EVPI is 0.77, so learning the true success rate before the choice is worth at
most that much, in the units of the utility.

## Checked in the tests

- The cost distribution has the exact mean for a known model, and a fixed seed gives the same draws.
- It gives quantiles, an interval and a tail probability, and a lognormal input from an expert interval
  can drive the cost.
- The value of perfect information matches the closed form, and is 0 when one action always wins.
- The break-even probability is the loss ratio, and bisection finds where the net benefit changes sign.
- Bad input is refused: a model that does not return one finite cost per draw, and bad utilities.

## References

- Metropolis, N., and Ulam, S. (1949). The Monte Carlo method. *Journal of the American Statistical
  Association* 44(247):335-341. <https://doi.org/10.1080/01621459.1949.10483310>
- Raiffa, H., and Schlaifer, R. (1961). *Applied Statistical Decision Theory*. Harvard University Press.
- Claxton, K. (1999). The irrelevance of inference: a decision-making approach to the stochastic
  evaluation of health care technologies. *Journal of Health Economics* 18(3):341-364.
  <https://doi.org/10.1016/S0167-6296(98)00039-3>
