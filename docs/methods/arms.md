# Spending a budget over arms

Use this when you have several options (arms), each with an unknown success probability, and a
budget of trials to spend. Two goals need two kinds of method:

- Earn as much as possible while you learn: `statsx.bandit.thompson_allocation` and
  `statsx.bandit.bayes_ucb_arm`.
- Find the best arm: `statsx.bandit.successive_halving` for a fixed budget, and
  `statsx.identify.best_arm` to stop when the answer is certain.

## The model

Arm \(i\) gives a 1 with probability \(\mu_i\) and a 0 otherwise. With a \(\mathrm{Beta}(a, b)\)
prior, after \(s_i\) successes in \(n_i\) pulls the posterior is

\[
\mu_i \mid \text{data} \sim \mathrm{Beta}(a + s_i,\; b + n_i - s_i) .
\]

The regret after \(T\) pulls is \(R(T) = \sum_i (\mu^\ast - \mu_i)\,\mathbb{E}[n_i(T)]\), with
\(\mu^\ast = \max_i \mu_i\).

## Thompson sampling

For each pull, draw \(\theta_i \sim \mathrm{Beta}(a + s_i, b + n_i - s_i)\) for every arm and pull
the arm with the largest \(\theta_i\). The chance that arm \(i\) is pulled is its posterior
probability of being the best:

\[
\mathbb{P}(\text{pull } i) = \int_0^1 f_i(x) \prod_{j \ne i} F_j(x)\, dx ,
\]

where \(f_i\) and \(F_i\) are the posterior density and distribution function. Agrawal and Goyal
(2012) prove \(\mathbb{E}[R(T)] = O\!\big( (\sum_{i \ne i^\ast} \Delta_i^{-2})^2 \ln T \big)\), with
\(\Delta_i = \mu^\ast - \mu_i\).

## Bayes-UCB

At round \(t\), pull the arm with the largest posterior quantile

\[
q_i(t) = Q_i\!\left( 1 - \frac{1}{t\,(\log T)^c} \right) ,
\]

where \(Q_i\) is the quantile function of arm \(i\)'s posterior. Kaufmann, Cappe and Garivier
(2012) show logarithmic regret for \(c \ge 5\), and report that \(c = 0\) works best in
experiments. The package uses \(c = 0\) by default, and takes \(c\) and the horizon \(T\) as
arguments.

## Successive halving

With \(n\) arms and a budget of \(B\) pulls, there are \(\lceil \log_2 n \rceil\) rounds. In round
\(r\), each arm in the surviving set \(S_r\) is pulled

\[
\left\lfloor \frac{B}{|S_r|\,\lceil \log_2 n \rceil} \right\rfloor
\]

times, and the \(\lceil |S_r| / 2 \rceil\) arms with the best success rate **in that round** stay.
The last arm is the answer. The package follows Karnin, Koren and Somekh (2013), and uses only the
pulls of the current round to rank the arms, as they do. The posteriors in the result use all
pulls.

## Successive elimination with a stopping guarantee

`best_arm` pulls every arm still in the race, once or in batches. After round \(m\) each arm has
the always-valid interval \(C_{i,m} = [L_{i,m}, U_{i,m}]\) of the
[always-valid inference](always-valid.md) page, at level \(\delta / K\). An arm is removed when

\[
U_{i,m} < \max_{j} L_{j,m} .
\]

The method stops when one arm is left, or when every other upper end is at most
\(\max_j L_{j,m} + \varepsilon\).

**Guarantee.** Let \(E\) be the event that \(\mu_i \in C_{i,m}\) for every arm and every round.
Each arm's interval fails with probability at most \(\delta / K\) over all rounds, so by Boole's
inequality \(\mathbb{P}(E) \ge 1 - \delta\). On \(E\):

1. The best arm \(i^\ast\) is never removed. Its upper end is at least \(\mu^\ast\), and every lower
   end is at most its own mean, which is at most \(\mu^\ast\), so \(U_{i^\ast,m} \ge L_{j,m}\) for
   all \(j\).
2. If the method stops because every other upper end is at most \(L_{\ell,m} + \varepsilon\) for the
   leader \(\ell\), then \(\mu_j \le U_{j,m} \le L_{\ell,m} + \varepsilon \le \mu_\ell + \varepsilon\)
   for every \(j\). The leader is within \(\varepsilon\) of the best.

With \(\varepsilon = 0\) the arm returned is the best arm, with probability at least \(1 - \delta\).
Even-Dar, Mannor and Mansour (2006) give the same rule with a radius of order
\(\sqrt{\log(K t^2 / \delta) / t}\). The package uses the time-uniform radius from the
always-valid page, which avoids the sum over time steps.

If \(\varepsilon = 0\) and two arms are equal, the method never stops by itself. Set `max_pulls`,
and read `confident` in the result: it is `False` when the budget ran out and the guarantee does not
apply.

## Example

```python
import numpy as np
from statsx.conjugate import Beta
from statsx.bandit import thompson_allocation, bayes_ucb_indices, successive_halving
from statsx.identify import best_arm

# Thompson: uniform arm against Beta(2, 1). P(second is larger) = E[second] = 2/3.
alloc = thompson_allocation([Beta(1.0, 1.0), Beta(2.0, 1.0)], 100_000, seed=1)
print([round(s, 3) for s in alloc.shares])

# Bayes-UCB index at t = 4 for Beta(1, 1) is the 0.75 quantile; for Beta(2, 1) it is sqrt(0.75).
print([round(float(v), 4) for v in bayes_ucb_indices([Beta(1.0, 1.0), Beta(2.0, 1.0)], 4)])

rng = np.random.default_rng(0)
p = np.array([0.4] * 7 + [0.8])
result = successive_halving(lambda arm, n: int(rng.binomial(n, p[arm])), 8, 2000)
print(result.best, [len(r.arms) for r in result.rounds], result.pulls_used)

rng = np.random.default_rng(1)
p = [0.2, 0.4, 0.6, 0.9]
found = best_arm(lambda arm, n: int(rng.binomial(n, p[arm])), 4, delta=0.05, batch=5)
print(found.best, found.confident, found.pulls, found.removed_at)
```

```text
[0.332, 0.668]
[0.75, 0.866]
7 [8, 4, 2] 1994
3 True (25, 95, 150, 150) (5, 19, 30, None)
```

The last line reads: the best arm, whether the guarantee applies, the pulls per arm, and the round
in which each arm was removed. Worse arms are removed earlier and get fewer pulls.

## Checked in the tests

- Thompson sampling gives the second arm 2/3 of the pulls against a uniform arm, the exact value.
- The Bayes-UCB index equals the Beta quantile in closed form for \(\mathrm{Beta}(1, 1)\) and
  \(\mathrm{Beta}(2, 1)\).
- With three arms at 0.3, 0.5 and 0.6 and 1000 pulls, mean regret over 20 runs is 14.4 for Thompson
  sampling and 16.0 for Bayes-UCB, against 133 for an even split.
- The halving schedule matches the formula above for 8 and for 5 arms.
- `best_arm` with \(\delta = 0.2\) returned the best arm in all 300 runs (the bound allows up to
  20%). The intervals are conservative, so the observed error is far below \(\delta\).

## When it fails

- The arm outcomes change over time: all methods here assume each arm has one fixed \(\mu_i\).
- Outcomes are not 0 or 1. `best_arm` uses \(\sigma = 1/2\), which is valid for any outcome in
  \([0, 1]\), and conservative.

## References

- Thompson, W. R. (1933). On the likelihood that one unknown probability exceeds another in view of
  the evidence of two samples. *Biometrika* 25(3/4):285-294. <https://doi.org/10.2307/2332286>
- Agrawal, S., and Goyal, N. (2012). Analysis of Thompson sampling for the multi-armed bandit
  problem. *COLT*, PMLR 23:39.1-39.26. <https://proceedings.mlr.press/v23/agrawal12.html>
- Kaufmann, E., Cappe, O., and Garivier, A. (2012). On Bayesian upper confidence bounds for bandit
  problems. *AISTATS*, PMLR 22:592-600. <https://proceedings.mlr.press/v22/kaufmann12.html>
- Karnin, Z., Koren, T., and Somekh, O. (2013). Almost optimal exploration in multi-armed bandits.
  *ICML*, PMLR 28(3):1238-1246. <https://proceedings.mlr.press/v28/karnin13.html>
- Even-Dar, E., Mannor, S., and Mansour, Y. (2006). Action elimination and stopping conditions for
  the multi-armed bandit and reinforcement learning problems. *JMLR* 7:1079-1105.
  <https://jmlr.org/papers/v7/evendar06a.html>
