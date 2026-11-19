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
- The cost-per-pass 90% interval held the true value in 89% of 300 simulations.
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
- Attempts are independent. Repeated attempts of one task by one model share the task and the model,
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
- Linacre, J. M. (1989). *Many-facet Rasch measurement*. MESA Press, Chicago. The Rasch model with
  facets besides model and task, here the infra.
- Seber, G. A. F., and Lee, A. J. (2003). *Linear Regression Analysis*, 2nd ed., chapter 3 (estimable
  functions). Wiley. <https://doi.org/10.1002/9780471722199>
