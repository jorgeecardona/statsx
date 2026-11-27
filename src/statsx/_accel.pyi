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
def effective_sample_size(draws: npt.NDArray[np.float64]) -> float: ...
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
    offset: npt.NDArray[np.float64],
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
def kmeans_fit(
    data: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    k: int,
    max_iter: int,
    tol: float,
    seed: int,
    n_init: int,
    spherical: bool,
    threads: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], float, int, bool]: ...
def kmeans_assign(
    data: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    centroids: npt.NDArray[np.float64],
    threads: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]: ...
def dbscan_fit(
    data: npt.NDArray[np.float64], eps: float, min_samples: int, threads: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]]: ...
def kth_distances(
    data: npt.NDArray[np.float64], k: int, threads: int
) -> npt.NDArray[np.float64]: ...
def hdbscan_fit(
    data: npt.NDArray[np.float64],
    min_samples: int,
    min_cluster_size: int,
    allow_single_cluster: bool,
    cosine: bool,
    threads: int,
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
]: ...
def gmm_fit(
    data: npt.NDArray[np.float64],
    k: int,
    cov: int,
    max_iter: int,
    tol: float,
    reg: float,
    min_count: float,
    seed: int,
    n_init: int,
    threads: int,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    float,
    int,
    bool,
    list[float],
    int,
]: ...
def gmm_score(
    data: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    means: npt.NDArray[np.float64],
    covs: npt.NDArray[np.float64],
    cov: int,
    threads: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
def umap_knn(
    data: npt.NDArray[np.float64], k: int, threads: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]: ...
def umap_knn_between(
    reference: npt.NDArray[np.float64], query: npt.NDArray[np.float64], k: int, threads: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]: ...
def umap_table_d2(
    data: npt.NDArray[np.float64],
    rows: npt.NDArray[np.int64],
    idx: npt.NDArray[np.int64],
    threads: int,
) -> npt.NDArray[np.float64]: ...
def umap_place(
    embedding: npt.NDArray[np.float64],
    idx: npt.NDArray[np.int64],
    dist: npt.NDArray[np.float64],
    a: float,
    b: float,
    n_epochs: int,
    n_neg: int,
    seed: int,
    threads: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
def umap_fuzzy(
    idx: npt.NDArray[np.int64],
    dist: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int64] | None,
    q: float,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    float,
]: ...
def umap_components(
    indptr: npt.NDArray[np.int64], indices: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]: ...
def umap_spectral(
    indptr: npt.NDArray[np.int64],
    indices: npt.NDArray[np.int64],
    data: npt.NDArray[np.float64],
    dim: int,
    max_iter: int,
    tol: float,
    seed: int,
    threads: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], bool, int] | None: ...
def umap_layout(
    y0: npt.NDArray[np.float64],
    indptr: npt.NDArray[np.int64],
    indices: npt.NDArray[np.int64],
    data: npt.NDArray[np.float64],
    a: float,
    b: float,
    n_epochs: int,
    n_neg: int,
    learning_rate: float,
    seed: int,
    threads: int,
) -> npt.NDArray[np.float64]: ...
def umap_cross_entropy(
    y: npt.NDArray[np.float64],
    indptr: npt.NDArray[np.int64],
    indices: npt.NDArray[np.int64],
    data: npt.NDArray[np.float64],
    a: float,
    b: float,
    threads: int,
) -> float: ...
def beta_cdf(x: npt.NDArray[np.float64], a: float, b: float) -> list[float]: ...
def beta_ppf(p: npt.NDArray[np.float64], a: float, b: float) -> list[float]: ...
def sample_beta(a: float, b: float, n: int, seed: int) -> npt.NDArray[np.float64]: ...
def sample_gamma(shape: float, rate: float, n: int, seed: int) -> npt.NDArray[np.float64]: ...
def sample_normal(mean: float, sd: float, n: int, seed: int) -> npt.NDArray[np.float64]: ...
def t_cdf(x: npt.NDArray[np.float64], df: float) -> list[float]: ...
def t_ppf(p: npt.NDArray[np.float64], df: float) -> list[float]: ...
def isotonic_fit(
    y: npt.NDArray[np.float64], w: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]: ...
def platt_fit(
    scores: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> tuple[float, float]: ...
def nct_cdf(x: npt.NDArray[np.float64], df: float, delta: float) -> list[float]: ...
def chi2_cdf(x: npt.NDArray[np.float64], df: float) -> list[float]: ...
def chi2_sf(x: npt.NDArray[np.float64], df: float) -> list[float]: ...
