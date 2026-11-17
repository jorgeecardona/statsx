import numpy as np
import numpy.typing as npt

def version() -> str: ...
def beta_binomial(a: float, b: float, successes: float, trials: float) -> tuple[float, float]: ...
def gamma_poisson(
    shape: float, rate: float, count: float, exposure: float
) -> tuple[float, float]: ...
def normal_normal(
    mean0: float, sd0: float, sigma: float, n: float, xbar: float
) -> tuple[float, float]: ...
def dirichlet_multinomial(alpha: list[float], counts: list[float]) -> list[float]: ...
def bootstrap_indices(
    n: int,
    kind: str,
    n_resamples: int,
    seed: int,
    labels: npt.NDArray[np.int64] | None = None,
    block_length: int = 0,
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64]]: ...
def convergence(draws: npt.NDArray[np.float64]) -> tuple[float, float, float]: ...
def nig_fit(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], coef_var: float, a0: float, b0: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float, float]: ...
def nig_predict(
    mean: npt.NDArray[np.float64],
    cov: npt.NDArray[np.float64],
    an: float,
    bn: float,
    x_new: npt.NDArray[np.float64],
    n_draws: int,
    seed: int,
) -> npt.NDArray[np.float64]: ...
def fit_structural(
    y: npt.NDArray[np.float64],
    design: npt.NDArray[np.float64],
    prior_mean: npt.NDArray[np.float64],
    prior_prec: npt.NDArray[np.float64],
    sigma_scale: float,
    phi_scale: float,
    likelihood: str,
    n_draws: int,
    seed: int,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    tuple[float, float, float, float, float, float, float],
]: ...
def beta_cdf(x: npt.NDArray[np.float64], a: float, b: float) -> list[float]: ...
def beta_ppf(p: npt.NDArray[np.float64], a: float, b: float) -> list[float]: ...
