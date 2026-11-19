# Time to finish a task

Use this to estimate how long a task takes before it starts, from its input sizes, and how long is
left once it is running, from its progress. Both come with a credible interval. The modules are
`statsx.eta` and `statsx.progress`.

## Before the task: duration from its parameters

`fit_duration(durations, params)` fits this model to past runs:

\[
\log(\text{duration}) = b_0 + \sum_k b_k \log(1 + x_k) + \varepsilon, \qquad \varepsilon \sim \mathrm{Normal}(0, \sigma^2) .
\]

A coefficient of 1 on \(\log(1 + \text{files})\) means the duration grows in proportion to the number of
files. The prior is the conjugate Normal-Inverse-Gamma, so the fit is closed form and the predictive
distribution is a Student-t on the log scale (Gelman et al. 2013). The defaults of `DurationPrior` are a
weak prior on the coefficients (`coef_sd=10`, in units of the noise sd) and a noise variance prior with
shape 2 and scale 0.5.

- `model.predict(params)` gives the median duration of a new run with an interval.
- `model.coefficients` gives the posterior mean of each coefficient.
- `model.update(duration, params)` returns a new model after one more finished run. The result equals a
  refit on all runs (tested), without the refit.
- `predict` with a parameter left out gives a wider interval from the known ones.

## During the task: time left from progress

A task reports progress: for example files done out of the files to do, so a fraction \(f\) at elapsed
time \(e\). At a constant rate the time left is \(e (1 - f) / f\). Real tasks start slowly, speed up, or
have a slow last part. `fit_progress(durations, checkpoints)` learns from past runs how the true time
left relates to that extrapolation:

\[
\log\frac{\text{left}\cdot f}{e\,(1 - f)} = b_0 + b_1 \log f + \text{noise} .
\]

The fit is the same regression as above. The time left is the constant-rate extrapolation times the
exponential of the predictive draw, so it is always positive. `checkpoints` has one list of
(elapsed, fraction) pairs for each past run. `model.remaining(elapsed, fraction)` returns the time left
with an interval. The interval narrows as the task progresses.

## What it assumes

- The past runs are like the next run: same machine, same code, same load. A change in any of them
  breaks the model.
- On the log scale the duration is linear in the log inputs with normal noise of one size.
- Runs are independent of each other.
- Progress model: how a run progresses is stable across runs. The relation between the extrapolation and
  the true time left is the same shape in every run.
- Inputs are known before the run and are not negative.

## What the library checks and does not check

Checked: durations are finite and above 0, every parameter the model has is given (and no unknown one),
parameters are finite and not negative, each column has one value for each run, there is at least one
past run, `elapsed` is above 0, `fraction` is strictly between 0 and 1, and a checkpoint report that is
not usable is skipped (it needs 0 < elapsed < duration and 0 < fraction < 1).

Not checked:

- That the log-linear form fits your task. A task with a fixed set-up cost and a size-dependent part is
  not a power of the size.
- That a new run is inside the range of the past runs. A prediction outside it is an extrapolation, and
  the interval does not widen for that beyond what the model says.
- That the machine is the same. Durations from two different machines in one fit mix two models.

## Example

The data are simulated. Past runs have a duration of about \(2 (1 + \text{files})^{0.9}\) with log noise of
0.2. The second part is a task whose progress lags at first: the elapsed time is \(t f^{1.3}\) for a run
of total time \(t\), where \(f\) is the fraction done.

```python
import numpy as np
from statsx.eta import fit_duration
from statsx.progress import fit_progress

rng = np.random.default_rng(0)
files = rng.integers(10, 500, 60)
durations = 2.0 * (1 + files) ** 0.9 * np.exp(rng.normal(0.0, 0.2, 60))
model = fit_duration(durations, {"files": files})
print("coefficients:", {k: round(v, 1) for k, v in model.coefficients.items()})
guess = model.predict({"files": 100})
low, high = round(guess.interval.lower), round(guess.interval.upper)
print("100 files:", round(guess.estimate), "interval", (low, high), "level", guess.interval.level)
print("true expected duration:", round(2.0 * 101**0.9))
print("runs after update:", model.update(150.0, {"files": 100}).n)

total = rng.uniform(50, 150, 40)
checkpoints = [[(t * f**1.3, f) for f in (0.2, 0.4, 0.6, 0.8)] for t in total]
progress = fit_progress(total, checkpoints)
fraction = 0.5
elapsed = 100.0 * fraction**1.3  # a new run of total time 100
left = progress.remaining(elapsed, fraction)
print("time left:", round(left.estimate), "interval", (round(left.interval.lower), round(left.interval.upper)))
print("true time left:", round(100.0 - elapsed), " constant-rate guess:", round(elapsed * (1 - fraction) / fraction))
```

```text
coefficients: {'intercept': 0.9, 'files': 0.9}
100 files: 136 interval (92, 201) level 0.9
true expected duration: 127
runs after update: 61
time left: 60 interval (52, 68)
true time left: 59  constant-rate guess: 41
```

The interval of 92 to 201 is wide because 60 runs with noise of 0.2 leave a lot of doubt about a single
run. The time left is 60 with an interval of 52 to 68, and the true time left is 59. The constant-rate
guess, 41, is too low because the task lags at first.

## Checked in the tests

- The fit learns the duration from its parameters, more files predicts longer, and a fit with no
  parameters predicts the typical duration.
- The predictive interval covers new runs.
- Updating with each new run equals refitting from all runs.
- Unknown parameters give a wider interval, and that interval covers new runs.
- The estimate from progress beats the constant-rate guess and covers the truth. Its interval narrows as
  the task progresses. The same seed gives the same interval.
- Unusable reports are skipped and bad input is refused.

## References

- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., and Rubin, D. B. (2013).
  *Bayesian Data Analysis*, 3rd ed., chapter 14. CRC Press. <https://doi.org/10.1201/b16018>
- O'Hagan, A., and Forster, J. (2004). *Kendall's Advanced Theory of Statistics, Vol. 2B: Bayesian
  Inference*, 2nd ed., chapter 11. Arnold.
