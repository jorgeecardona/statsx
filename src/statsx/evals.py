"""Put models on one scale from their results on shared tasks.

Many models are tried on many tasks, and each (model, task) cell holds a pass (1) or fail (0), or a
score in [0, 1]. Cells may be missing, and a pair may be tried more than once. The model asks how
able each model is and how hard each task is:

    P(model m solves task t) = sigmoid(theta_m - b_t)

``theta_m`` is the ability of model ``m`` and ``b_t`` the difficulty of task ``t``, on the same
logit scale. Two models that never saw the same task are still comparable when the cells link them
through other models and tasks. ``Responses.n_components`` counts the groups that are not linked
to each other; abilities from different groups are not on one scale, and ``fit_rasch`` refuses
such data unless you say otherwise.

Two more tools answer the cost side, since a stronger model may also use fewer tokens:

- ``cost_per_pass``: the expected cost to get one pass, ``C / R``, with a posterior. ``C`` is the
  mean cost of one attempt and ``R`` the pass rate. With independent attempts, the expected number
  of attempts until the first pass is ``1 / R``, so ``C / R`` is the expected cost to first pass.
  With ``tasks`` the interval comes from a bootstrap over tasks, not from the posterior.
- ``fit_tokens``: the ratio of tokens between two models on the same tasks, from a two-way model
  ``log tokens = model effect + task effect + noise``. The posterior is the reference one, so each
  ratio has an exact Student t interval.

An ``infra`` column (the GPU, driver, host or run batch of each row) adds one effect per infra:
``log tokens = model + task + infra + noise`` and ``P = sigmoid(theta_m - b_t + s_i)``. The first
infra seen is the reference with effect 0, so abilities are read on it. Models are comparable only
when the data link their infras; both fits refuse data where they do not, and say which models.
``calibrate_infra`` lists the models that ran on both of two infras: the bridges.

The Rasch prior is ``theta_m ~ Normal(0, ability_sd^2)`` and ``b_t ~ Normal(0, difficulty_sd^2)``. The
posterior is approximated by a Gaussian at its mode (the Laplace approximation), found by Newton
steps on the concave log posterior. With few cells, the true posterior is skewed and the Gaussian
is rough: check the sample size before you read a probability near 0 or 1.

References:
    Rasch, G. (1960). *Probabilistic Models for Some Intelligence and Attainment Tests*. Danish
    Institute for Educational Research.

    Polo, F. M., Weber, L., Choshen, L., Sun, Y., Xu, G., & Yurochkin, M. (2024). tinyBenchmarks:
    evaluating LLMs with fewer examples. arXiv:2402.14992. The use of item response models for
    language models, with new models linked to fixed item parameters.

    Erol, M. H., El, B., Suzgun, M., Yuksekgonul, M., and Zou, J. (2025). Cost-of-Pass: an economic
    framework for evaluating language models. arXiv:2504.13359. The cost to first pass as the cost of an attempt over its pass rate.

    Miller, E. (2024). Adding error bars to evals: a statistical approach to language model
    evaluations. arXiv:2411.00640. Paired differences on the same questions.

    Davison, A. C., & Hinkley, D. V. (1997). *Bootstrap Methods and their Application*, section
    3.8 (hierarchical data). Cambridge University Press.
    https://doi.org/10.1017/CBO9780511802843. The bootstrap over tasks in ``cost_per_pass``.

    Tobin, J. (1958). Estimation of relationships for limited dependent variables.
    *Econometrica* 26(1):24-36. https://doi.org/10.2307/1907382. The censored normal regression
    in ``fit_tokens(..., censored=...)``.

    Linacre, J. M. (1989). *Many-facet Rasch measurement*. MESA Press, Chicago. Facets besides
    model and task, used here for the infra.

    Seber, G. A. F., & Lee, A. J. (2003). *Linear Regression Analysis*, 2nd ed., chapter 3
    (estimable functions). Wiley. https://doi.org/10.1002/9780471722199.

    Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013).
    *Bayesian Data Analysis*, 3rd ed., chapter 4 (the normal approximation at the mode). CRC Press.
    https://doi.org/10.1201/b16018.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np
import numpy.typing as npt

from statsx import _accel
from statsx.conjugate import Beta, NormalInverseGamma
from statsx.result import Interval

_NORMAL = NormalDist()
_NEWTON_STEPS = 100
_DECREMENT = 1e-10


@dataclass(frozen=True, slots=True)
class Responses:
    """Results of models on tasks, as one row per attempt.

    Args:
        models: The model names, in the order of their index.
        tasks: The task names, in the order of their index.
        model_index: The model of each row, as an index into ``models``.
        task_index: The task of each row, as an index into ``tasks``.
        passed: The result of each row, in [0, 1].
        infras: The infra names (GPU, driver, host), in the order of their index; empty when the
            rows carry no infra. The first one is the reference.
        infra_index: The infra of each row, as an index into ``infras``; None without infras.
    """

    models: tuple[str, ...]
    tasks: tuple[str, ...]
    model_index: npt.NDArray[np.int64]
    task_index: npt.NDArray[np.int64]
    passed: npt.NDArray[np.float64]
    infras: tuple[str, ...] = ()
    infra_index: npt.NDArray[np.int64] | None = None

    @classmethod
    def from_long(
        cls,
        models: Sequence[str],
        tasks: Sequence[str],
        passed: npt.ArrayLike,
        infra: Sequence[str] | None = None,
    ) -> Responses:
        """Build the table from columns of the same length: model, task, result and, optionally, infra.

        Raises:
            ValueError: If the columns differ in length, are empty, or a result is not in [0, 1].
        """
        y = np.asarray(passed, dtype=np.float64).ravel()
        if not (len(models) == len(tasks) == y.size) or y.size == 0:
            raise ValueError("models, tasks and passed must be non-empty and of the same length")
        if infra is not None and len(infra) != y.size:
            raise ValueError("infra must have one value per row")
        if not np.all((y >= 0.0) & (y <= 1.0)):
            raise ValueError("passed must be between 0 and 1")
        model_names = tuple(dict.fromkeys(models))
        task_names = tuple(dict.fromkeys(tasks))
        m_of = {name: i for i, name in enumerate(model_names)}
        t_of = {name: i for i, name in enumerate(task_names)}
        infra_names = () if infra is None else tuple(dict.fromkeys(infra))
        i_of = {name: i for i, name in enumerate(infra_names)}
        return cls(
            model_names,
            task_names,
            np.array([m_of[m] for m in models], dtype=np.int64),
            np.array([t_of[t] for t in tasks], dtype=np.int64),
            y,
            infra_names,
            None if infra is None else np.array([i_of[i] for i in infra], dtype=np.int64),
        )

    @property
    def n_components(self) -> int:
        """Return the number of groups of models and tasks that are not linked to each other."""
        n_m, n_t = len(self.models), len(self.tasks)
        parent = list(range(n_m + n_t))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for m, t in zip(self.model_index.tolist(), self.task_index.tolist(), strict=True):
            parent[find(m)] = find(n_m + t)
        return len({find(i) for i in range(n_m + n_t)})


@dataclass(frozen=True, slots=True)
class RaschFit:
    """A fitted Rasch model: abilities and difficulties with their Gaussian posterior.

    Args:
        models: The model names.
        tasks: The task names.
        ability: The posterior mode of the ability of each model.
        difficulty: The posterior mode of the difficulty of each task.
        covariance: The posterior covariance of ``(ability, difficulty)`` jointly.
        ability_sd: The prior sd of an ability.
        difficulty_sd: The prior sd of a difficulty.
        n_rows: The number of rows fitted.
        infras: The infra names; empty when the fit has no infra effect. The first is the reference.
        infra_shift: The posterior mode of the shift of each infra, in logits (0 for the reference).
        infra_sd: The prior sd of an infra shift.

    With infras, ``covariance`` covers ``(ability, difficulty, shift of every infra but the
    reference)``, and abilities are read on the reference infra.
    """

    models: tuple[str, ...]
    tasks: tuple[str, ...]
    ability: npt.NDArray[np.float64]
    difficulty: npt.NDArray[np.float64]
    covariance: npt.NDArray[np.float64]
    ability_sd: float
    difficulty_sd: float
    n_rows: int
    infras: tuple[str, ...] = ()
    infra_shift: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    infra_sd: float = 1.0

    def _model(self, name: str) -> int:
        if name not in self.models:
            raise ValueError(f"unknown model {name!r}")
        return self.models.index(name)

    def ability_interval(self, model: str, level: float = 0.95) -> Interval:
        """Return the equal-tailed posterior interval of the ability of ``model``."""
        _check_level(level)
        i = self._model(model)
        half = _NORMAL.inv_cdf(0.5 + level / 2.0) * math.sqrt(self.covariance[i, i])
        return Interval(float(self.ability[i] - half), float(self.ability[i] + half), level)

    def _difference(self, a: str, b: str) -> tuple[float, float]:
        i, j = self._model(a), self._model(b)
        mean = float(self.ability[i] - self.ability[j])
        variance = float(
            self.covariance[i, i] + self.covariance[j, j] - 2.0 * self.covariance[i, j]
        )
        return mean, math.sqrt(max(variance, 0.0))

    def ability_difference(self, a: str, b: str, level: float = 0.95) -> Interval:
        """Return the posterior interval of ``ability(a) - ability(b)``, in logits."""
        _check_level(level)
        mean, sd = self._difference(a, b)
        half = _NORMAL.inv_cdf(0.5 + level / 2.0) * sd
        return Interval(mean - half, mean + half, level)

    def p_better(self, a: str, b: str) -> float:
        """Return the posterior probability that model ``a`` is more able than model ``b``."""
        mean, sd = self._difference(a, b)
        if sd == 0.0:
            return 1.0 if mean > 0.0 else 0.0 if mean < 0.0 else 0.5
        return _NORMAL.cdf(mean / sd)

    def ability_draws(self, n: int, *, seed: int = 0) -> npt.NDArray[np.float64]:
        """Return ``n`` posterior draws of all abilities, one row per draw."""
        k = len(self.models)
        cov = self.covariance[:k, :k]
        rng = np.random.default_rng(seed)
        return rng.multivariate_normal(self.ability, cov, size=n, method="cholesky")

    def _shift_column(self, infra: str) -> int | None:
        """Return the covariance index of the shift of ``infra``; None for the reference."""
        if infra not in self.infras:
            raise ValueError(f"unknown infra {infra!r}")
        k = self.infras.index(infra)
        return None if k == 0 else len(self.models) + len(self.tasks) + k - 1

    def infra_shift_difference(self, a: str, b: str, level: float = 0.95) -> Interval:
        """Return the posterior interval of ``shift(a) - shift(b)`` between two infras, in logits.

        A positive value means a model of the same ability solves more tasks on ``a`` than on ``b``.
        """
        _check_level(level)
        ia, ib = self._shift_column(a), self._shift_column(b)
        cov = self.covariance
        var = 0.0
        if ia is not None:
            var += float(cov[ia, ia])
        if ib is not None:
            var += float(cov[ib, ib])
        if ia is not None and ib is not None:
            var -= 2.0 * float(cov[ia, ib])
        mean = float(
            self.infra_shift[self.infras.index(a)] - self.infra_shift[self.infras.index(b)]
        )
        half = _NORMAL.inv_cdf(0.5 + level / 2.0) * math.sqrt(max(var, 0.0))
        return Interval(mean - half, mean + half, level)

    def p_solve(self, model: str, task: str, infra: str | None = None) -> float:
        """Return the fitted probability that ``model`` solves ``task`` (at the posterior mode).

        With ``infra`` the shift of that infra is added; without it, the reference infra is meant.
        """
        if task not in self.tasks:
            raise ValueError(f"unknown task {task!r}")
        eta = self.ability[self._model(model)] - self.difficulty[self.tasks.index(task)]
        if infra is not None:
            self._shift_column(infra)
            eta += self.infra_shift[self.infras.index(infra)]
        return float(1.0 / (1.0 + math.exp(-eta)))


def _check_level(level: float) -> None:
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")


def _sigmoid(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0))), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class _Term:
    """One indicator column group of the linear predictor: ``sign * x[index]`` on rows in ``mask``."""

    index: npt.NDArray[np.int64]
    sign: float
    mask: npt.NDArray[np.float64]


def _gram(
    terms: Sequence[_Term], weight: npt.NDArray[np.float64], size: int
) -> npt.NDArray[np.float64]:
    """Return ``sum_r weight[r] * z_r z_r'`` for the rows ``z_r`` of the linear predictor.

    Row ``r`` of the predictor is ``sum_k sign_k * x[index_k[r]]`` over the terms whose mask is 1.
    """
    total = np.zeros((size, size))
    for a in terms:
        for b in terms:
            w = weight * a.mask * b.mask * a.sign * b.sign
            flat = np.bincount(a.index * size + b.index, weights=w, minlength=size * size)
            total += flat.reshape(size, size)
    return total


def _terms(data: Responses) -> tuple[list[_Term], int]:
    """Return the predictor terms of ``data`` and the number of parameters.

    Parameters are the abilities, the difficulties, and the shift of every infra but the first.
    """
    n_m, n_t = len(data.models), len(data.tasks)
    ones = np.ones(data.passed.size)
    terms = [_Term(data.model_index, 1.0, ones), _Term(n_m + data.task_index, -1.0, ones)]
    if data.infra_index is None:
        return terms, n_m + n_t
    shifted = (data.infra_index > 0).astype(np.float64)
    column = n_m + n_t + np.maximum(data.infra_index - 1, 0)
    terms.append(_Term(column.astype(np.int64), 1.0, shifted))
    return terms, n_m + n_t + len(data.infras) - 1


def _check_comparable(data: Responses) -> None:
    """Raise ValueError when the data leave some models on a scale of their own.

    The linear predictor is ``model + task + infra``. Two models can be compared only when no
    direction of the parameters that leaves every row unchanged moves one of them and not the
    other. Those directions are the null space of the indicator design.
    """
    if data.infra_index is None:
        return
    n_m, n_t, n_i = len(data.models), len(data.tasks), len(data.infras)
    ones = np.ones(data.passed.size)
    index = [data.model_index, n_m + data.task_index, n_m + n_t + data.infra_index]
    gram = _gram([_Term(i, 1.0, ones) for i in index], ones, n_m + n_t + n_i)
    values, vectors = np.linalg.eigh(gram)
    null = vectors[:n_m, values < 1e-9 * max(float(values.max()), 1.0)]
    groups: dict[tuple[float, ...], list[str]] = {}
    for name, row in zip(data.models, np.round(null, 6).tolist(), strict=True):
        groups.setdefault(tuple(row), []).append(name)
    if len(groups) > 1:
        listed = "; ".join("[" + ", ".join(g) + "]" for g in groups.values())
        raise ValueError(
            "the models cannot be compared because no model links their infras: models in "
            f"different brackets are not on one scale: {listed}"
        )


@dataclass(frozen=True, slots=True)
class InfraBridge:
    """The models that ran on both of two infras.

    Args:
        infra_a: The first infra.
        infra_b: The second infra.
        models: The models with rows on both. Empty when the infras are not bridged directly.
        n_rows: The number of rows of those models on the two infras.
    """

    infra_a: str
    infra_b: str
    models: tuple[str, ...]
    n_rows: int


def calibrate_infra(models: Sequence[str], infra: Sequence[str]) -> tuple[InfraBridge, ...]:
    """Return, for every pair of infras, the models that ran on both: the bridges.

    The infra effect is fitted from the models that ran on more than one infra. A pair with no
    bridge model is linked, if at all, only through a third infra.

    Raises:
        ValueError: If the columns differ in length or are empty.
    """
    if len(models) != len(infra) or len(models) == 0:
        raise ValueError("models and infra must be non-empty and of the same length")
    names = tuple(dict.fromkeys(infra))
    result: list[InfraBridge] = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            on_a = {m for m, k in zip(models, infra, strict=True) if k == a}
            on_b = {m for m, k in zip(models, infra, strict=True) if k == b}
            both = tuple(m for m in dict.fromkeys(models) if m in on_a and m in on_b)
            rows = sum(1 for m, k in zip(models, infra, strict=True) if m in both and k in (a, b))
            result.append(InfraBridge(a, b, both, rows))
    return tuple(result)


def fit_rasch(
    data: Responses,
    *,
    ability_sd: float = 2.0,
    difficulty_sd: float = 2.0,
    infra_sd: float = 1.0,
    allow_disconnected: bool = False,
) -> RaschFit:
    """Fit the Rasch model to ``data`` by Newton steps and a Gaussian posterior at the mode.

    When ``data`` carries an infra column the model is ``sigmoid(theta_m - b_t + s_i)``, the first
    infra is the reference (``s = 0``), and abilities are read on it.

    Args:
        data: The results.
        ability_sd: The prior sd of an ability, in logits. The default is weak.
        difficulty_sd: The prior sd of a difficulty, in logits.
        infra_sd: The prior sd of the shift of an infra, in logits.
        allow_disconnected: Fit even if some models and tasks are not linked to the others. The
            abilities of different groups are then not on one scale.

    Raises:
        ValueError: If a prior sd is not positive, the data are not linked (and
            ``allow_disconnected`` is False), the infras leave models that cannot be compared
            (the message names them), or Newton's method does not converge.
    """
    if not (ability_sd > 0.0 and difficulty_sd > 0.0 and infra_sd > 0.0):
        raise ValueError("ability_sd, difficulty_sd and infra_sd must be positive")
    if data.n_components > 1 and not allow_disconnected:
        raise ValueError(
            f"the data have {data.n_components} groups of models and tasks that are not linked: "
            "their abilities are not on one scale"
        )
    if not allow_disconnected:
        _check_comparable(data)
    n_m, n_t = len(data.models), len(data.tasks)
    terms, size = _terms(data)
    y = data.passed
    prior = np.concatenate(
        [
            np.full(n_m, 1.0 / ability_sd**2),
            np.full(n_t, 1.0 / difficulty_sd**2),
            np.full(size - n_m - n_t, 1.0 / infra_sd**2),
        ]
    )

    def predictor(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        eta = np.zeros(y.size)
        for term in terms:
            eta += term.sign * term.mask * x[term.index]
        return eta

    x = np.zeros(size)
    for _ in range(_NEWTON_STEPS):
        p = _sigmoid(predictor(x))
        residual = y - p
        grad = -prior * x
        for term in terms:
            grad = grad + term.sign * np.bincount(
                term.index, weights=residual * term.mask, minlength=size
            )
        step = np.linalg.solve(np.diag(prior) + _gram(terms, p * (1.0 - p), size), grad)
        x = x + step
        if float(grad @ step) < _DECREMENT:
            break
    else:
        raise ValueError("Newton's method did not converge")
    p = _sigmoid(predictor(x))
    neg_hess = np.diag(prior) + _gram(terms, p * (1.0 - p), size)
    return RaschFit(
        data.models,
        data.tasks,
        x[:n_m].copy(),
        x[n_m : n_m + n_t].copy(),
        np.linalg.inv(neg_hess),
        ability_sd,
        difficulty_sd,
        int(y.size),
        data.infras,
        np.concatenate([np.zeros(min(len(data.infras), 1)), x[n_m + n_t :]]),
        infra_sd,
    )


@dataclass(frozen=True, slots=True)
class CostPerPass:
    """The posterior of the expected cost to get one pass, for one model.

    The pass rate ``R`` has a Beta posterior. The mean cost of an attempt ``C`` comes from a
    lognormal model of the costs: its mean is ``exp(mu + sigma2 / 2)`` with ``(mu, sigma2)`` from
    the reference Normal-Inverse-Gamma posterior of the log costs. The two are drawn
    independently and every attempt counts as exchangeable. That ignores that attempts come in
    tasks, and that a few hard tasks can cost most and fail most: the interval is then too narrow.

    With ``task_cost`` and ``task_passes`` (from ``cost_per_pass(..., tasks=...)``) the draws come
    from a bootstrap over tasks instead: resample the tasks with replacement, keep all attempts of
    a task together, and take total cost over total passes (Davison & Hinkley 1997, section 3.8).

    Args:
        pass_rate: The posterior of the pass rate.
        log_cost: The posterior of the mean and variance of the log cost of an attempt.
        n: The number of attempts.
        task_cost: The total cost of each task, or None for the posterior draws.
        task_passes: The number of passes of each task, or None.
    """

    pass_rate: Beta
    log_cost: NormalInverseGamma
    n: int
    task_cost: npt.NDArray[np.float64] | None = None
    task_passes: npt.NDArray[np.float64] | None = None

    @property
    def n_tasks(self) -> int:
        """Return the number of tasks resampled; 0 when the draws are the posterior ones."""
        return 0 if self.task_cost is None else self.task_cost.size

    def sample(self, n: int, *, seed: int = 0) -> npt.NDArray[np.float64]:
        """Return ``n`` draws of the cost per pass; the same seed gives the same draws.

        With tasks, a draw is total cost over total passes of one resample of the tasks. A
        resample with no pass gives ``inf``: the cost per pass has no upper bound there.
        """
        if self.task_cost is not None and self.task_passes is not None:
            return _task_bootstrap(self.task_cost, self.task_passes, n, seed)
        rate = self.pass_rate.sample(n, seed=seed)
        mu, sigma2 = self.log_cost.sample_joint(n, seed=seed + 2)
        return np.exp(mu + 0.5 * sigma2) / rate

    def interval(self, level: float = 0.95, *, n_draws: int = 100_000, seed: int = 0) -> Interval:
        """Return the equal-tailed credible interval of the cost per pass."""
        _check_level(level)
        draws = self.sample(n_draws, seed=seed)
        tails = [0.5 - level / 2.0, 0.5 + level / 2.0]
        lo, hi = np.quantile(draws, tails, method="inverted_cdf")
        return Interval(float(lo), float(hi), level)


_CHUNK = 4096
"""Bootstrap resamples drawn at once, so memory stays at chunk x tasks."""


def _task_bootstrap(
    cost: npt.NDArray[np.float64], passes: npt.NDArray[np.float64], n: int, seed: int
) -> npt.NDArray[np.float64]:
    """Return ``n`` bootstrap draws of total cost over total passes, resampling tasks."""
    rng = np.random.default_rng(seed)
    k = cost.size
    out = np.empty(n)
    for start in range(0, n, _CHUNK):
        rows = min(_CHUNK, n - start)
        counts = rng.multinomial(k, np.full(k, 1.0 / k), size=rows).astype(np.float64)
        total_cost, total_passes = counts @ cost, counts @ passes
        with np.errstate(divide="ignore"):
            out[start : start + rows] = np.where(
                total_passes > 0.0, total_cost / np.maximum(total_passes, 1e-300), np.inf
            )
    return out


def cost_per_pass(
    passed: npt.ArrayLike,
    costs: npt.ArrayLike,
    *,
    prior: Beta = Beta(1.0, 1.0),  # noqa: B008
    tasks: Sequence[object] | None = None,
) -> CostPerPass:
    """Return the posterior of the expected cost to get one pass, from the attempts of one model.

    Without ``tasks`` every attempt counts as exchangeable. Pass ``tasks`` when attempts come in
    tasks (or sessions): the draws then come from a bootstrap that resamples whole tasks, which
    keeps a hard task's cost and its failures together (#12).

    Args:
        passed: The result of each attempt: 1 for a pass, 0 for a fail (a score in [0, 1] counts as
            that share of a pass).
        costs: The cost of each attempt, above 0, in one unit (tokens or money), failures included.
        prior: The Beta prior of the pass rate.
        tasks: The task (or session) of each attempt. Attempts with the same task are resampled
            together. At least 2 tasks.

    Raises:
        ValueError: If the lengths differ, there are fewer than 2 attempts or 2 tasks, a cost is
            not above 0, the costs are all equal, or a result is outside [0, 1].
    """
    y = np.asarray(passed, dtype=np.float64).ravel()
    c = np.asarray(costs, dtype=np.float64).ravel()
    if y.size != c.size or y.size < 2 or not np.all(np.isfinite(c)) or np.any(c <= 0.0):  # noqa: PLR2004
        raise ValueError(
            "passed and costs must have the same length (at least 2), and costs be above 0"
        )
    if not np.all((y >= 0.0) & (y <= 1.0)):
        raise ValueError("passed must be between 0 and 1")
    successes = float(y.sum())
    pass_rate = Beta(prior.alpha + successes, prior.beta + y.size - successes)
    log_cost = NormalInverseGamma.reference(np.log(c))
    if tasks is None:
        return CostPerPass(pass_rate, log_cost, int(y.size))
    if len(tasks) != y.size:
        raise ValueError("tasks must have the same length as passed")
    names = list(dict.fromkeys(tasks))
    if len(names) < 2:  # noqa: PLR2004
        raise ValueError("tasks must hold at least 2 different tasks to resample")
    index = {name: i for i, name in enumerate(names)}
    task = np.array([index[name] for name in tasks])
    task_cost = np.bincount(task, weights=c, minlength=len(names))
    task_passes = np.bincount(task, weights=y, minlength=len(names))
    return CostPerPass(pass_rate, log_cost, int(y.size), task_cost, task_passes)


@dataclass(frozen=True, slots=True)
class TokenFit:
    """A two-way model of log tokens: ``log tokens = model effect + task effect + noise``.

    A comparison of two models is the difference of their effects, which does not depend on how
    the effects are anchored. It has a Student t posterior with ``df`` degrees of freedom under
    the reference prior.

    With an infra column the model is ``log tokens = model + task + infra + noise``; the first
    infra is the reference with effect 0.

    Args:
        models: The model names.
        tasks: The task names.
        effects: The fitted effect of each model, then of each task, then of each infra but the
            reference (minimum-norm solution).
        gram_pinv: The pseudo-inverse of ``X'X`` for the indicators.
        noise_sd: The residual standard deviation on the log scale.
        df: The degrees of freedom of the noise estimate: rows minus the rank of the design.
        infras: The infra names; empty when the fit has no infra effect.
        n_censored: The number of rows that hit the token budget. With censored rows the fit is
            the maximum likelihood of a censored normal on the log scale, and ``gram_pinv`` times
            ``noise_sd ** 2`` is the inverse of its observed information, so the intervals are
            approximate (Wald, with the t quantile of ``df``).
        left_out: The tasks where every row hit the budget. They are left out of the fit: they
            say only that every model needs more than the budget, nothing about the ratio.
    """

    models: tuple[str, ...]
    tasks: tuple[str, ...]
    effects: npt.NDArray[np.float64]
    gram_pinv: npt.NDArray[np.float64]
    noise_sd: float
    df: int
    infras: tuple[str, ...] = ()
    n_censored: int = 0
    left_out: tuple[str, ...] = ()

    def _summarize(self, contrast: npt.NDArray[np.float64]) -> tuple[float, float]:
        scale = self.noise_sd * math.sqrt(float(contrast @ self.gram_pinv @ contrast))
        return float(contrast @ self.effects), scale

    def _difference(self, a: str, b: str) -> tuple[float, float]:
        for name in (a, b):
            if name not in self.models:
                raise ValueError(f"unknown model {name!r}")
        i, j = self.models.index(a), self.models.index(b)
        contrast = np.zeros(self.effects.size)
        contrast[i], contrast[j] = 1.0, -1.0
        return self._summarize(contrast)

    def _infra_difference(self, a: str, b: str) -> tuple[float, float]:
        for name in (a, b):
            if name not in self.infras:
                raise ValueError(f"unknown infra {name!r}")
        offset = len(self.models) + len(self.tasks)
        contrast = np.zeros(self.effects.size)
        for name, sign in ((a, 1.0), (b, -1.0)):
            k = self.infras.index(name)
            if k > 0:  # the reference infra has effect 0
                contrast[offset + k - 1] += sign
        gram = np.linalg.pinv(self.gram_pinv)
        if float(np.linalg.norm(contrast - self.gram_pinv @ gram @ contrast)) > 1e-8:  # noqa: PLR2004
            raise ValueError(
                f"the effect of infra {a!r} against {b!r} is not identified: "
                "no model links the two infras"
            )
        return self._summarize(contrast)

    def infra_log_effect(self, a: str, b: str, level: float = 0.95) -> Interval:
        """Return the interval of ``log(tokens on infra a / tokens on infra b)`` for one model and task.

        Raises:
            ValueError: If an infra is unknown, or the data do not link the two infras.
        """
        _check_level(level)
        mean, scale = self._infra_difference(a, b)
        q = _accel.t_ppf(np.array([0.5 + level / 2.0]), float(self.df))[0]
        return Interval(mean - q * scale, mean + q * scale, level)

    def infra_effect(self, a: str, b: str, level: float = 0.95) -> Interval:
        """Return the interval of how many times more tokens infra ``a`` uses than infra ``b``."""
        log = self.infra_log_effect(a, b, level)
        return Interval(math.exp(log.lower), math.exp(log.upper), level)

    def log_ratio(self, a: str, b: str, level: float = 0.95) -> Interval:
        """Return the interval of ``log(tokens of a / tokens of b)`` on the same task."""
        _check_level(level)
        mean, scale = self._difference(a, b)
        q = _accel.t_ppf(np.array([0.5 + level / 2.0]), float(self.df))[0]
        return Interval(mean - q * scale, mean + q * scale, level)

    def ratio(self, a: str, b: str, level: float = 0.95) -> Interval:
        """Return the interval of how many times more tokens ``a`` uses than ``b`` on one task."""
        log = self.log_ratio(a, b, level)
        return Interval(math.exp(log.lower), math.exp(log.upper), level)

    def p_fewer(self, a: str, b: str) -> float:
        """Return the posterior probability that ``a`` uses fewer tokens than ``b`` on a task."""
        mean, scale = self._difference(a, b)
        return float(_accel.t_cdf(np.array([-mean / scale]), float(self.df))[0])


def fit_tokens(
    models: Sequence[str],
    tasks: Sequence[str],
    tokens: npt.ArrayLike,
    infra: Sequence[str] | None = None,
    *,
    censored: npt.ArrayLike | None = None,
) -> TokenFit:
    """Fit ``log tokens = model effect + task effect (+ infra effect) + noise``, one row per attempt.

    Cells may be missing and attempts repeated. The tasks are the shared yardstick: the ratio of
    two models is read on the same tasks, so a model that was given only hard tasks is not
    penalized for them. With ``infra`` (the GPU, driver or host of each row) the ratio of two
    models is read after the infra effect is removed. The first infra is the reference.

    Pass every run that spent tokens, failed runs included: a model that fails its long tasks
    looks cheaper when only passed runs are fitted. A fit on passed runs only is valid for the
    tasks that both models solved. A run that hit the token budget did not show how many tokens
    it needed, only that it needed at least the budget: mark it in ``censored``. The fit is then
    a censored normal regression on the log scale (Tobin 1958), by maximum likelihood (#13).

    Args:
        models: The model of each row.
        tasks: The task of each row.
        tokens: The tokens of each row, above 0. For a censored row, the budget it hit.
        infra: The infra of each row, or None.
        censored: True for a row that hit the token budget, or None when none did.

    Raises:
        ValueError: If the columns differ in length, a count is not above 0, the models and tasks
            are not linked, the infras leave models that cannot be compared (the message names
            them), there are no degrees of freedom left for the noise, or a model has only
            censored rows (its effect has no upper bound). A task with only censored rows is
            left out and named in ``TokenFit.left_out``.
    """
    x = np.asarray(tokens, dtype=np.float64).ravel()
    if not (len(models) == len(tasks) == x.size) or x.size == 0:
        raise ValueError("models, tasks and tokens must be non-empty and of the same length")
    cut = _censored_rows(censored, x.size)
    left_out = _all_censored(tasks, cut)
    if left_out:
        keep = [task not in left_out for task in tasks]
        models = [m for m, k in zip(models, keep, strict=True) if k]
        tasks = [t for t, k in zip(tasks, keep, strict=True) if k]
        infra = None if infra is None else [i for i, k in zip(infra, keep, strict=True) if k]
        x, cut = x[np.asarray(keep)], cut[np.asarray(keep)]
        if x.size == 0:
            raise ValueError("every row hit the budget: there is nothing to fit")
    if not np.all(np.isfinite(x)) or np.any(x <= 0.0):
        raise ValueError("tokens must be above 0")
    table = Responses.from_long(models, tasks, np.zeros(x.size), infra)
    if table.n_components > 1:
        raise ValueError("the models and tasks are not linked: ratios would not be comparable")
    _check_comparable(table)
    n_m, n_t = len(table.models), len(table.tasks)
    design = np.zeros((x.size, n_m + n_t + max(len(table.infras) - 1, 0)))
    rows = np.arange(x.size)
    design[rows, table.model_index] = 1.0
    design[rows, n_m + table.task_index] = 1.0
    if table.infra_index is not None:
        later = table.infra_index > 0
        design[rows[later], n_m + n_t + table.infra_index[later] - 1] = 1.0
    y = np.log(x)
    gram_pinv = np.linalg.pinv(design.T @ design)
    effects = gram_pinv @ design.T @ y
    rank = int(np.linalg.matrix_rank(design))
    df = x.size - rank
    if df < 1:
        raise ValueError("no degrees of freedom left for the noise: add attempts or tasks")
    resid = y - design @ effects
    noise_sd = math.sqrt(float(resid @ resid) / df)
    if not cut.any():
        return TokenFit(
            table.models, table.tasks, effects, gram_pinv, noise_sd, df, table.infras, 0, left_out
        )
    _check_seen(table.models, table.model_index, cut, "model")
    effects, covariance, noise_sd = _censored_fit(design, y, cut, effects, noise_sd)
    # Maximum likelihood divides by the rows, not by the degrees of freedom. With one effect per
    # task that makes the noise and the covariance too small; this factor makes both exact when no
    # row is censored.
    covariance *= x.size / df
    noise_sd *= math.sqrt(x.size / df)
    return TokenFit(
        table.models,
        table.tasks,
        effects,
        covariance / noise_sd**2,
        noise_sd,
        df,
        table.infras,
        int(cut.sum()),
        left_out,
    )


def _censored_rows(censored: npt.ArrayLike | None, n: int) -> npt.NDArray[np.bool_]:
    if censored is None:
        return np.zeros(n, dtype=np.bool_)
    cut = np.asarray(censored).ravel()
    if cut.size != n:
        raise ValueError("censored must have one flag per row")
    return cut.astype(np.bool_)


def _all_censored(tasks: Sequence[str], cut: npt.NDArray[np.bool_]) -> tuple[str, ...]:
    """Return the tasks whose every row is censored, in order of first appearance."""
    seen: dict[str, bool] = {}
    for task, flag in zip(tasks, cut.tolist(), strict=True):
        seen[task] = seen.get(task, True) and flag
    return tuple(task for task, all_cut in seen.items() if all_cut)


def _check_seen(
    names: tuple[str, ...], index: npt.NDArray[np.int64], cut: npt.NDArray[np.bool_], kind: str
) -> None:
    """Refuse a model or task with only censored rows: its effect has no upper bound."""
    seen = np.bincount(index[~cut], minlength=len(names))
    blind = [names[i] for i in np.flatnonzero(seen == 0)]
    if blind:
        raise ValueError(
            f"every row of {kind} {blind} hit the budget: its effect has no upper bound"
        )


_erfc = np.vectorize(math.erfc, otypes=[np.float64])
_SQRT2 = math.sqrt(2.0)
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def _upper_tail(z: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], ...]:
    """Return ``log(1 - Phi(z))`` and the inverse Mills ratio ``phi(z) / (1 - Phi(z))``."""
    q = 0.5 * _erfc(z / _SQRT2)
    far = q < 1e-280  # noqa: PLR2004 (past this, erfc underflows; use the asymptotic series)
    zf = np.where(far, z, 1.0)
    series = 1.0 - 1.0 / zf**2 + 3.0 / zf**4
    log_pdf = -0.5 * z**2 - _LOG_SQRT_2PI
    log_q = np.where(far, log_pdf - np.log(zf) + np.log(series), np.log(np.where(far, 1.0, q)))
    mills = np.where(far, zf / series, np.exp(log_pdf - log_q))
    return log_q, mills


def _censored_loglik(
    design: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    cut: npt.NDArray[np.bool_],
    theta: npt.NDArray[np.float64],
) -> float:
    sigma = math.exp(float(theta[-1]))
    z = (y - design @ theta[:-1]) / sigma
    log_q, _ = _upper_tail(z[cut])
    seen = z[~cut]
    return float(-seen.size * theta[-1] - 0.5 * seen @ seen + log_q.sum())


def _censored_fit(
    design: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    cut: npt.NDArray[np.bool_],
    start: npt.NDArray[np.float64],
    start_sd: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
    """Return the effects, their covariance and sigma of the right-censored normal regression.

    Newton steps on ``(effects, log sigma)`` with step halving. The design may be rank deficient;
    the pseudo-inverse keeps the steps in its row space, so estimable contrasts are unique.
    """
    theta = np.append(start, math.log(start_sd))
    current = _censored_loglik(design, y, cut, theta)
    for _ in range(_NEWTON_STEPS):
        gradient, hessian = _censored_derivatives(design, y, cut, theta)
        step = -np.linalg.pinv(hessian) @ gradient
        size = 1.0
        while size > 1e-8:  # noqa: PLR2004
            value = _censored_loglik(design, y, cut, theta + size * step)
            if value >= current:
                break
            size /= 2.0
        else:
            break  # no step uphill: theta is the maximum to rounding
        theta, gain, current = theta + size * step, value - current, value
        if abs(gain) < _DECREMENT and float(np.abs(size * step).max()) < 1e-8:  # noqa: PLR2004
            break
    else:
        raise ValueError("the censored fit did not converge: check the budget flags")
    _, hessian = _censored_derivatives(design, y, cut, theta)
    covariance = np.linalg.pinv(-hessian)
    return theta[:-1], covariance[:-1, :-1], math.exp(float(theta[-1]))


def _censored_derivatives(
    design: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    cut: npt.NDArray[np.bool_],
    theta: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the gradient and Hessian of the censored log likelihood in ``(effects, log sigma)``."""
    sigma = math.exp(float(theta[-1]))
    z = (y - design @ theta[:-1]) / sigma
    _, mills = _upper_tail(z[cut])
    zc = z[cut]
    slope = mills * (mills - zc)  # the derivative of the inverse Mills ratio
    # Per row: d loglik / d eta, d2 / d eta2, d2 / d eta d s, d loglik / d s, d2 / d s2.
    g_eta = np.where(cut, 0.0, z / sigma)
    h_eta = np.where(cut, 0.0, -1.0 / sigma**2)
    h_eta_s = np.where(cut, 0.0, -2.0 * z / sigma)
    g_s = np.where(cut, 0.0, z**2 - 1.0)
    h_s = np.where(cut, 0.0, -2.0 * z**2)
    g_eta[cut] = mills / sigma
    h_eta[cut] = -slope / sigma**2
    h_eta_s[cut] = -(slope * zc + mills) / sigma
    g_s[cut] = mills * zc
    h_s[cut] = -slope * zc**2 - mills * zc
    p = design.shape[1]
    gradient = np.append(design.T @ g_eta, g_s.sum())
    hessian = np.empty((p + 1, p + 1))
    hessian[:p, :p] = design.T @ (design * h_eta[:, None])
    hessian[:p, p] = hessian[p, :p] = design.T @ h_eta_s
    hessian[p, p] = h_s.sum()
    return gradient, hessian
