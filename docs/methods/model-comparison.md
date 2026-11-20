# Compare models on one scale

Use this when you have several models (different sizes, families, or versions) that were tried on
many shared tasks, and you want to say which is more able, by how much, how many tokens each
needs, and what each costs per solved task. The module is `statsx.evals`.

A raw pass rate does not compare models fairly when they ran different tasks, or when the tasks
differ in difficulty. The method below puts the models and the tasks on one scale.

## The data

One row per attempt: the model, the task, and the result in \([0, 1]\) (1 for a pass). Optionally the
tokens or the cost of the attempt. Cells may be missing, and a pair may appear more than once. This
is the `Responses` table.

## Ability and difficulty: the Rasch model

Let \(\theta_m\) be the ability of model \(m\) and \(b_t\) the difficulty of task \(t\), on the same
logit scale. The chance that model \(m\) solves task \(t\) is

\[
\mathbb{P}(y_{mt} = 1) = \sigma(\theta_m - b_t), \qquad \sigma(x) = \frac{1}{1 + e^{-x}} .
\]

A model that is 1 logit more able than another solves a task of any difficulty with odds \(e\) times
higher. Priors are \(\theta_m \sim \mathcal{N}(0, s_\theta^2)\) and \(b_t \sim \mathcal{N}(0, s_b^2)\),
with weak defaults of 2 logits. The log posterior is

\[
\log p(\theta, b \mid y) = \sum_r \Big[ y_r \log \sigma(\eta_r) + (1 - y_r) \log \sigma(-\eta_r) \Big]
- \frac{\lVert\theta\rVert^2}{2 s_\theta^2} - \frac{\lVert b\rVert^2}{2 s_b^2} + \text{const},
\qquad \eta_r = \theta_{m_r} - b_{t_r} .
\]

It is concave. Newton's method finds its mode \(\hat x\), with \(x = (\theta, b)\). With
\(p_r = \sigma(\eta_r)\) and \(w_r = p_r(1 - p_r)\), the negative Hessian is

\[
H = \mathrm{diag}(s^{-2}) + \sum_r w_r\, (e_{m_r} - e_{n_m + t_r})(e_{m_r} - e_{n_m + t_r})^\top ,
\]

and the posterior is approximated by \(\mathcal{N}(\hat x, H^{-1})\) (the Laplace approximation).

**Comparing two models.** The difference \(\theta_a - \theta_b\) is Gaussian under the approximation, with
variance \(H^{-1}_{aa} + H^{-1}_{bb} - 2 H^{-1}_{ab}\). So \(\mathbb{P}(\theta_a > \theta_b)\) and its interval
are closed forms, and `ability_draws` gives joint draws for anything else.

**Linking.** Two models that never saw the same task are comparable when the rows link them through
other models and tasks. Think of a graph with a node per model and per task, and an edge per row.
If the graph has more than one connected component, abilities in different components are not on one
scale, and `fit_rasch` refuses the data (`Responses.n_components` tells you how many groups).

## Tokens: the same tasks, how many times more

Stronger models often need fewer tokens for the same task. Write the log tokens of an attempt as

\[
\log \text{tokens}_{r} = \alpha_{m_r} + c_{t_r} + \varepsilon_r, \qquad \varepsilon_r \sim \mathcal{N}(0, \sigma^2) .
\]

The task effect \(c_t\) absorbs how long a task is, so a model that was given only long tasks is not
penalized. With the reference prior \(p(\alpha, c, \sigma^2) \propto 1/\sigma^2\), a contrast \(\kappa^\top\beta\) of the effects (for
example \(\alpha_a - \alpha_b\)) has a Student t posterior:

\[
\frac{\kappa^\top \beta - \kappa^\top \hat\beta}{s \sqrt{\kappa^\top (X^\top X)^{+} \kappa}} \sim t_{\nu}, \qquad
\nu = n - \operatorname{rank}(X), \quad s^2 = \frac{\mathrm{RSS}}{\nu} .
\]

\(e^{\alpha_a - \alpha_b}\) is the ratio of the tokens of \(a\) to those of \(b\) on one task. On a complete design,
this interval equals the paired t interval of the per-task differences.

### Failed runs and runs that hit the budget

Fit every run that spent tokens, failed runs included. A fit on passed runs only is selected: a
model that fails its long tasks looks cheaper. Such a fit is valid only for the tasks that both
models solved.

A run that hit the token budget shows only that it needed at least the budget. Mark it in
`censored`. The fit is then a censored normal regression on the log scale (Tobin 1958), by maximum
likelihood, with Wald intervals.

```python
import math

import numpy as np
from statsx.evals import fit_tokens

rng = np.random.default_rng(13)
length = rng.normal(8.0, 0.8, 30)
rows = [(m, f"t{t}", math.exp(length[t] + shift + rng.normal(0.0, 0.3)))
        for m, shift in (("a", 0.0), ("b", math.log(1.5))) for t in range(30) for _ in range(2)]
run_model, run_task, need = (list(c) for c in zip(*rows, strict=True))
budget = math.exp(8.6)
spent = np.minimum(need, budget)
hit = np.asarray(need) >= budget
fit = fit_tokens(run_model, run_task, spent, censored=hit)
taken_as_is = fit_tokens(run_model, run_task, spent)
print(f"{hit.mean():.0%} hit the budget, true ratio 1.5")
for name, result in (("censored", fit), ("as is", taken_as_is)):
    r = result.ratio("b", "a", 0.9)
    print(f"{name:8s} 90% [{r.lower:.2f}, {r.upper:.2f}]")
print(len(fit.left_out), "tasks left out, every run at the budget")
```

```text
49% hit the budget, true ratio 1.5
censored 90% [1.44, 1.78]
as is    90% [1.19, 1.35]
11 tasks left out, every run at the budget
```

- A task where every run hit the budget says nothing about the ratio. It is left out and named in
  `left_out`. A model where every run hit the budget is refused.
- The noise and the covariance are scaled by rows over degrees of freedom, so that with no
  censored row the fit is the exact one above.
- In 150 simulations like this one (32% of runs at the budget), the 90% interval held the true
  ratio in 85%. Taking the capped values as they are gave a median ratio of 1.32; dropping the capped
  runs gave 1.41.

## Hardware and infra: remove the infra effect

The same model can use another number of tokens, or pass another share of the tasks, on another
GPU, driver or host. Greedy decoding is not bit-identical across them. If you ignore it, the shift
lands in the model effect and one model looks cheaper or dearer than it is. Pass an `infra` column
(one name per row) and the fits add one effect per infra:

\[
\log \text{tokens}_{r} = \alpha_{m_r} + c_{t_r} + \iota_{i_r} + \varepsilon_r, \qquad
\mathbb{P}(\text{pass}_r) = \sigma(\theta_{m_r} - b_{t_r} + s_{i_r}) .
\]

The first infra that appears is the reference: \(\iota = s = 0\), and abilities are read on it.
`fit_tokens(..., infra=...)` gives `infra_effect(a, b)`, how many times more tokens infra `a` uses than
`b` for the same model and task. `fit_rasch` gives `infra_shift_difference(a, b)`, in logits.

The infra effect is only separate from the model effect when some model ran on more than one infra.
That model is the bridge. Without a bridge, "model A is dearer" and "infra 1 is dearer" fit the data
equally well. Both fits refuse such data with a `ValueError` that lists the models that are not on one
scale. `calibrate_infra(models, infra)` shows the bridges for every pair of infras before you fit.

The shift is the same for every model. A model that reacts to the infra differently from the bridge
(for example a quantized kernel that exists on one GPU only) is not corrected.

```python
import numpy as np
from statsx.evals import calibrate_infra, fit_tokens

# Simulated: Qwen3-0.6B ran on both GPUs, the other two models on one each.
rng = np.random.default_rng(0)
task_effect = rng.normal(5.0, 1.0, 30)
effect = {"qwen3-0.6b": 0.0, "qwen3-1.7b": np.log(1.5), "gemma": 0.3}
cells = [
    ("qwen3-0.6b", "T2000"),
    ("qwen3-0.6b", "RTX3090"),
    ("qwen3-1.7b", "T2000"),
    ("gemma", "RTX3090"),
]
models, infras, tasks, tokens = [], [], [], []
for model, infra in cells:
    for t in range(30):
        shift = np.log(1.2) if infra == "RTX3090" else 0.0  # the 3090 uses 20% more tokens
        models.append(model)
        infras.append(infra)
        tasks.append(f"t{t}")
        tokens.append(np.exp(rng.normal(effect[model] + task_effect[t] + shift, 0.25)))

(bridge,) = calibrate_infra(models, infras)
print("bridge:", bridge.models)

fit = fit_tokens(models, tasks, tokens, infra=infras)
raw = fit_tokens(models, tasks, tokens)
for label, f in (("with infra", fit), ("without   ", raw)):
    r = f.ratio("qwen3-1.7b", "gemma")
    print(label, "qwen3-1.7b / gemma:", round(float(np.sqrt(r.lower * r.upper)), 1))
e = fit.infra_effect("RTX3090", "T2000")
print("RTX3090 / T2000:", round(float(np.sqrt(e.lower * e.upper)), 1))
```

```text
bridge: ('qwen3-0.6b',)
with infra qwen3-1.7b / gemma: 1.1
without    qwen3-1.7b / gemma: 0.9
RTX3090 / T2000: 1.2
```

The true ratio of `qwen3-1.7b` to `gemma` is 1.1. The two models never ran on the same GPU, so the
fit without the infra effect reads 0.9: the 3090 uses 1.2 times the tokens, and the model ratio takes
the blame. The 0.6B model, run on both GPUs, is the bridge that lets the fit separate them.

## Cost per solved task

If attempts are independent with pass rate \(R\), the number of attempts until the first pass is
geometric with mean \(1/R\). With \(C\) the mean cost of one attempt, failures included, the expected
cost to the first pass is

\[
\text{cost per pass} = \frac{C}{R} .
\]

This is the cost-of-pass of Erol et al. (2025). The package draws \(R\) from its Beta posterior and
\(C = \exp(\mu + \sigma^2 / 2)\) from the lognormal model of the costs, using the reference Normal-Inverse-Gamma
posterior of \(\log\text{cost}\). The two draws are independent, which ignores the link between a task's cost and its
result. The draws work in `statsx.posterior.compare` (use their logarithms to compare ratios).

### Attempts grouped by task

Pass `tasks` when the attempts come in tasks or sessions. A few hard tasks often cost most and fail
most, and the posterior above treats every attempt as exchangeable, so its interval is too narrow.
With `tasks` the draws come from a bootstrap that resamples whole tasks and takes total cost over
total passes (Davison and Hinkley 1997, section 3.8).

```python
import numpy as np
from statsx.evals import cost_per_pass

rng = np.random.default_rng(12)
difficulty = rng.normal(0.0, 1.5, 40)
solve = 1.0 / (1.0 + np.exp(-(0.5 - difficulty)))
task_passed = (rng.random((40, 5)) < solve[:, None]).astype(float).ravel()
task_cost = np.exp(1.0 + 0.8 * difficulty[:, None] + rng.normal(0.0, 0.3, (40, 5))).ravel()
task_ids = np.repeat(np.arange(40), 5).tolist()
flat = cost_per_pass(task_passed, task_cost).interval(0.9)
by_task = cost_per_pass(task_passed, task_cost, tasks=task_ids).interval(0.9)
print(f"attempts exchangeable: [{flat.lower:.1f}, {flat.upper:.1f}]")
print(f"tasks resampled:       [{by_task.lower:.1f}, {by_task.upper:.1f}]")
```

```text
attempts exchangeable: [7.1, 10.4]
tasks resampled:       [5.4, 12.9]
```

In 200 simulations like this one (40 tasks, 5 attempts each, true value 9.9), the 90% interval held
the true value in 85% of runs with `tasks` and in 51% without. A resample with no pass gives `inf`,
so the upper end can be unbounded when passes are rare. With few tasks the percentile interval is a
little narrow; prefer 40 tasks or more.

## Cost at the quality of a reference

Use `cost_at_ability` to ask what a model costs per solved task when it must reach the quality of
another model. Token price alone does not answer it: a cheap model that fails often or needs many
tokens may not be cheaper.

1. The Rasch fit gives the abilities and the difficulties. The quality of an ability is the expected
   share of the tasks it solves in one attempt. `reference` is a model name or an ability.
2. A model below the reference tries each task up to \(k\) times and stops at the first pass. Task
   \(t\) is then solved with probability \(1 - (1 - p_{mt})^k\). The number \(k\) is the smallest that reaches
   the reference, mixed with \(k - 1\) so the quality matches exactly. A model above the reference
   is read at one attempt.
3. The cost of an attempt is \(\exp(\alpha_m + c_t + s^2/2)\) from the two-way model of the log costs,
   so retries on hard tasks cost what hard tasks cost. With `infra`, both fits remove the infra effect.

```python
import math

import numpy as np
from statsx.evals import cost_at_ability

rng = np.random.default_rng(14)
hardness = rng.normal(0.0, 1.0, 60)
ability_of = {"weak": -0.5, "strong": 1.0}
price = {"weak": 1.0, "strong": 6.0}
attempt_rows = []
for name, theta in ability_of.items():
    for t in range(60):
        p = 1.0 / (1.0 + math.exp(-(theta - hardness[t])))
        for _ in range(3):
            cost = price[name] * math.exp(0.4 * hardness[t] + rng.normal(0.0, 0.2))
            attempt_rows.append((name, f"t{t}", float(rng.random() < p), cost))
m_col, t_col, p_col, c_col = (list(c) for c in zip(*attempt_rows, strict=True))
matched = cost_at_ability(m_col, t_col, p_col, c_col, reference="strong", n_draws=1000)
print(f"reference quality {matched.reference_quality:.2f}")
for name in matched.models:
    k = matched.attempts[matched.models.index(name)]
    c = matched.interval(name, 0.9)
    print(f"{name:6s} {k:.1f} attempts per task, cost per solved task 90% [{c.lower:.1f}, {c.upper:.1f}]")
print(f"P(weak cheaper at the quality of strong) = {matched.p_cheaper('weak', 'strong'):.2f}")
```

```text
reference quality 0.63
weak   2.9 attempts per task, cost per solved task 90% [3.2, 5.1]
strong 1.0 attempts per task, cost per solved task 90% [10.1, 11.9]
P(weak cheaper at the quality of strong) = 1.00
```

In 30 simulations like this one, the 90% interval held the true cost per solved task in 26 for each
model.

What it assumes, and does not check:

- Attempts at one task are independent given the model and the task, and a verifier tells which
  attempt passed. Retries of a language model on one task are usually more alike than that. The
  attempts a weak model needs, and so its cost, are then a lower bound.
- The quality is the share of these tasks solved. A different task set gives a different answer.
- The Rasch posterior and the cost posterior are drawn independently.
- A reference out of reach within `MAX_ATTEMPTS` (100) attempts gives `inf`.

## Example

The data are simulated: four models with true abilities \(-1, 0, 0.8, 1.6\) on 60 tasks, 30% of the cells
missing, and the weaker models using more tokens.

```python
import math
import numpy as np
from statsx.evals import Responses, cost_per_pass, fit_rasch, fit_tokens
from statsx.posterior import compare

rng = np.random.default_rng(0)
names = ["small", "medium", "large", "frontier"]
ability = [-1.0, 0.0, 0.8, 1.6]
token_scale = [2.0, 1.4, 1.0, 0.8]  # tokens per task, relative to "large"
difficulty = rng.normal(0.0, 1.0, 60)
rows = []
for m, name in enumerate(names):
    for t in range(60):
        if rng.random() < 0.3:
            continue
        solved = rng.random() < 1.0 / (1.0 + math.exp(-(ability[m] - difficulty[t])))
        tokens = 2000.0 * token_scale[m] * math.exp(0.3 * difficulty[t] + rng.normal(0.0, 0.2))
        rows.append((name, f"task{t}", float(solved), tokens))
model_col, task_col, passed, tokens = zip(*rows, strict=True)

fit = fit_rasch(Responses.from_long(model_col, task_col, passed))
for name in names:
    a = fit.ability_interval(name, 0.9)
    print(f"{name:9s} ability {fit.ability[fit.models.index(name)]:5.2f}  90% [{a.lower:5.2f}, {a.upper:5.2f}]")
print("P(frontier > large) =", round(fit.p_better("frontier", "large"), 3))

tok = fit_tokens(model_col, task_col, tokens)
r = tok.ratio("small", "large", 0.9)
print(f"small uses {r.lower:.2f} to {r.upper:.2f} times the tokens of large")

attempts = np.array(model_col)
costs = {n: cost_per_pass(np.array(passed)[attempts == n], np.array(tokens)[attempts == n]) for n in names}
better = compare(np.log(costs["large"].sample(50_000)), np.log(costs["frontier"].sample(50_000, seed=9)),
                 margin=math.log(1.1))
print("P(frontier cheaper per pass than large) =", round(better.p_a_greater, 3))
```

```text
small     ability -1.05  90% [-1.75, -0.35]
medium    ability -0.07  90% [-0.78,  0.64]
large     ability  0.69  90% [-0.05,  1.43]
frontier  ability  2.24  90% [ 1.33,  3.16]
P(frontier > large) = 0.993
small uses 1.89 to 2.21 times the tokens of large
P(frontier cheaper per pass than large) = 1.0
```

## Checked in the tests

- The fitted mode has a zero gradient, and the covariance equals minus the inverse Hessian, both
  against finite differences of the log posterior.
- A 90% interval for the difference of two abilities held the true difference in 87% of 900 simulated
  comparisons (6 models, 40 tasks, priors matching the simulation). The Gaussian approximation is
  slightly narrow: use a level above the one you need when the data are small.
- With 60% of the cells missing the models are still linked; data with two unlinked groups are refused.
- A model that solves every task gets a finite ability.
- The token ratio on a complete design equals the paired t interval exactly. With missing cells and
  different tasks per model, a 90% interval held the true ratio in 91% of 400 simulations.
- The cost-per-pass 90% interval held the true value in 89% of 300 simulations of independent
  attempts. With attempts grouped by task, it held it in 51% of 200; the bootstrap over tasks held it
  in 85%.
- With 3 models on 2 infras and a 20% token shift on one, the 90% interval of the token ratio held the
  true ratio in 88 of 100 runs, and the interval of the infra effect in 91. The Rasch infra shift was
  covered in 90 of 100 runs and the ability gap in 88. Without the infra effect, the ratio is off by
  the infra shift (1.25 instead of 1.5 in the test).
- Models that never share an infra are refused, and the message names them.

## What it assumes

- One ability per model: a model that is better at code and worse at maths is described by an average.
  The task set decides what the average means.
- The Rasch model has equal discrimination for all tasks. A task that every model solves or fails
  carries no information about the abilities.
- `cost_per_pass` without `tasks` treats attempts as independent and exchangeable. Pass `tasks`
  when they come in tasks. Repeated attempts of one task by one model share the task and the model,
  which the model accounts for, but not any other link (the same prompt in two forms).
- The abilities are only comparable for the tasks you ran. A different task set gives a different scale.
- With few cells per model, the Gaussian posterior is rough. Read probabilities near 0 or 1 with care.

## What the library checks and does not check

Checked: the columns are non-empty and of one length, `passed` is between 0 and 1, names are known,
the sds are positive, Newton's method converged, and the data are one linked group. Models or tasks in
groups that no row links, and models that no infra links, are refused.

Not checked: that the Rasch model fits the tasks. The list above is what the fit takes for granted.

## References

- Rasch, G. (1960). *Probabilistic Models for Some Intelligence and Attainment Tests*. Danish Institute
  for Educational Research.
- Polo, F. M., Weber, L., Choshen, L., Sun, Y., and Yurochkin, M. (2024). tinyBenchmarks: evaluating
  LLMs with fewer examples. <https://arxiv.org/abs/2402.14992>
- Erol, M. H., El, B., Suzgun, M., Yuksekgonul, M., and Zou, J. (2025). Cost-of-Pass: an economic framework for evaluating language models.
  <https://arxiv.org/abs/2504.13359>
- Miller, E. (2024). Adding error bars to evals: a statistical approach to language model evaluations.
  <https://arxiv.org/abs/2411.00640>
- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., and Rubin, D. B. (2013).
  *Bayesian Data Analysis*, 3rd ed., chapters 4 and 14. CRC Press. <https://doi.org/10.1201/b16018>
- Davison, A. C., and Hinkley, D. V. (1997). *Bootstrap Methods and their Application*, section 3.8
  (hierarchical data). Cambridge University Press. <https://doi.org/10.1017/CBO9780511802843>
- Tobin, J. (1958). Estimation of relationships for limited dependent variables. *Econometrica*
  26(1):24-36. <https://doi.org/10.2307/1907382>
- Linacre, J. M. (1989). *Many-facet Rasch measurement*. MESA Press, Chicago. The Rasch model with
  facets besides model and task, here the infra.
- Seber, G. A. F., and Lee, A. J. (2003). *Linear Regression Analysis*, 2nd ed., chapter 3 (estimable
  functions). Wiley. <https://doi.org/10.1002/9780471722199>
