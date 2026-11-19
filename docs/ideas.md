# Ideas and open questions

This page is the running log of ideas from the maintainer's conversations. Add an entry when an
idea comes up, even if it is not built. Each entry says its status and the need that tracks it.

Status values: `built`, `planned` (has a need), `open` (a question, no decision), `dropped`.

## 2026-11-18

### A computation that runs does not prove its assumption

- Status: planned (principle P-3, need AS-1).
- Idea: an algorithm gives a number for any data once you assume a model. That the number comes out,
  or that the iteration converges, is no evidence that the model is right. This is circular: the
  assumption defines the computation, and the computation cannot confirm the assumption.
- Today: `FitCheck` in `fit_trend` checks the computation (k-hat, ESS, grid edge) and not the model.
  The paired tests carry an `assumption` field in words, and no test of it.
- Direction: every result names its assumptions, and where the data can speak, a check of them
  (posterior predictive check for fits, a residual check for tests). A result that passed its
  computation checks says so in those words, not "valid".

### Embeddings: do two sets of points come from one distribution

- Status: planned (need ED-1, not built).
- Idea: for BERT, nomic or OpenAI embeddings, compare two sets with a split into buckets, or by first
  looking at low-dimensional projections. Intuition: m points per set can be separated in about m+1
  dimensions, so separation in high dimension is cheap and proves little.
- Known: Cover (1965) separability, energy distance, MMD, held-out classifier tests, pooled PCA + KS
  (Rabanser 2019). Exact statements of Cover's theorem and Kim et al. 2021 are not yet verified.
- Rule that follows from the entry above: a classifier that separates the sets only counts on
  held-out points, because in-sample separation is guaranteed when the dimension is large.

### LLM judges and an arena

- Status: partly built. Done: position bias per judge (CO-7), Elo scale with intervals (CO-8), online Elo
  and Glicko-2 (CO-9). To do: do judges rank the systems differently (CO-10).
- Idea: judges with different criteria give different results; an arena scores the systems like chess.
- Assumption to state: Bradley-Terry assumes one strength per system on one scale. If the judges
  disagree, one strength is the wrong model. CO-10 is the check of that assumption.

### From the PyMC and NumPyro research

- Status: open, ranked list. Not started: CUPED (Deng 2013), PPI++ for judge scores (arXiv:2311.01453),
  PSIS-LOO as public API, cluster-robust errors for evals, ratio-metric confidence sequences,
  Diebold-Mariano.
- Open decision: MCMC for non-conjugate models and install without JAX or a C compiler (BA-4, BA-6).
  They wait on whether to write NUTS in Rust.

### GitLab issues #4 to #11 (fit_trend)

- Status: built (see the CHANGELOG). A consumer replacing a hand-written sampler filed eight
  issues: Poisson-like series refused, settable priors, `harmonics="auto"`, one base error, a float
  period, `require=False`, a reusable season, forecast buckets.
- Lesson that matches the first entry: the grid-edge check refused a series because the posterior
  sat at a valid endpoint (no extra dispersion, no drift). A check of the computation read a correct
  answer as a failure. The fix was to name which ends of the grid are valid answers, and to say the
  cause in the message.
