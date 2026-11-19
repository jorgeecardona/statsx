# Group sequential bounds

Use this when you plan a fixed number of looks at the data and want to stop early if the effect is
clear, with an overall error rate of \(\alpha\). The function is
`statsx.groupsequential.group_sequential_bounds`.

## The problem

You plan \(K\) looks. At look \(k\) the standardized statistic \(Z_k\) uses all data so far. Testing
each look at \(|Z_k| \ge 1.96\) gives an overall false stop rate above 5%. A design uses a different
critical value \(c_k\) at each look, so that

\[
\mathbb{P}_0\!\left( \exists k \le K : |Z_k| \ge c_k \right) = \alpha .
\]

## The model

Let \(t_k\) be the information time of look \(k\): the share of the final information, with
\(0 < t_1 < \dots < t_K = 1\). Equal spacing gives \(t_k = k/K\). Under no effect, the partial sums
are a Brownian motion \(B\), and

\[
Z_k = \frac{B(t_k)}{\sqrt{t_k}} .
\]

The increments \(B(t_k) - B(t_{k-1})\) are independent \(\mathcal{N}(0, t_k - t_{k-1})\).

## The two boundary families

Write the boundary on the scale of \(B(t_k)\) as \(b_k\), so that \(c_k = b_k / \sqrt{t_k}\).

| Design | Boundary \(b_k\) | Critical value \(c_k\) |
|---|---|---|
| Pocock | \(C\sqrt{t_k}\) | \(C\), the same at every look |
| O'Brien-Fleming | \(C\) | \(C / \sqrt{t_k}\), very strict early |

The constant \(C\) is the only unknown. It is found from the condition on \(\alpha\) above.

## Computing the error rate

Let \(f_k(x)\) be the density of \(B(t_k)\) on the event that no look before or at \(k\) has stopped.
Then

\[
f_1(x) = \frac{1}{\sqrt{t_1}}\,\varphi\!\left(\frac{x}{\sqrt{t_1}}\right), \qquad
f_k(y) = \int_{-b_{k-1}}^{b_{k-1}} f_{k-1}(x)\,
\frac{1}{\sqrt{\Delta_k}}\,\varphi\!\left(\frac{y - x}{\sqrt{\Delta_k}}\right) dx ,
\]

with \(\Delta_k = t_k - t_{k-1}\), \(\varphi\) the standard normal density, and \(f_k\) restricted
to \((-b_k, b_k)\) for the next step. The probability of stopping for the first time at look \(k\)
is

\[
\alpha_k = \int_{-b_{k-1}}^{b_{k-1}} f_{k-1}(x)\,
\mathbb{P}\!\left( |x + \sqrt{\Delta_k}\,N| \ge b_k \right) dx ,
\qquad N \sim \mathcal{N}(0, 1) ,
\]

and the overall error rate is \(\sum_k \alpha_k\). This is the recursive integration of Armitage,
McPherson and Rowe (1969). The package evaluates the integrals on an 801-point trapezoid grid per
look, and solves \(\sum_k \alpha_k(C) = \alpha\) for \(C\) by the secant method. The design reports
\(c_k\), the matching nominal p-value, and \(\alpha_k\) at each look.

For a one-sided design the lower limit of each integral is moved to \(-\infty\) (in practice, nine
standard deviations below), and the tail probability is one-sided.

## Example

```python
from statsx.groupsequential import group_sequential_bounds

for kind in ("pocock", "obrien_fleming"):
    d = group_sequential_bounds(5, kind=kind)
    print(kind, [round(z, 3) for z in d.z_bounds])
    print("  spent", [round(a, 5) for a in d.alpha_spent], round(sum(d.alpha_spent), 6))
```

```text
pocock [2.413, 2.413, 2.413, 2.413, 2.413]
  spent [0.01581, 0.01171, 0.00902, 0.00731, 0.00615] 0.05
obrien_fleming [4.562, 3.226, 2.634, 2.281, 2.04]
  spent [1e-05, 0.00125, 0.00764, 0.01668, 0.02442] 0.05
```

The constants match the published tables to three decimals: Pocock 2.178, 2.289, 2.361 and 2.413
for 2 to 5 looks, and O'Brien-Fleming 4.562, 3.226, 2.634, 2.281 and 2.040 for 5 looks.

## What the choice costs

Pocock spends a large share of \(\alpha\) at the first looks, so early stops are easy and the
final look is stricter than a fixed-sample test. O'Brien-Fleming spends almost nothing early and
ends close to the fixed-sample value, so the final look keeps most of the power. In the example,
the first look of five spends 0.0158 (Pocock) and 0.00001 (O'Brien-Fleming).

## Checked in the tests

- The constants above, against the published tables.
- A simulation of 60000 four-look trials with no effect stopped in 5.0% of them, for both designs.
  Using 1.96 at every look stopped in more than 10%.

## What it assumes

- The test statistic is asymptotically normal, with the usual joint distribution across looks.
- The looks happen at the planned information times.

## What the library checks and does not check

Checked: `n_looks` is a whole number from 1 to 20, `alpha` is in (0, 0.5), `kind` is
`obrien_fleming` or `pocock`, and `times` are increasing, positive and end at 1.

Not checked:

- That the looks really happen at the planned times. A different spending function (Lan and DeMets
  1983) is needed for unplanned looks, and the package does not offer it.
- The estimate at a stop. The design bounds the error under no effect. An effect estimated at an
  early stop is biased upwards.

## References

- Pocock, S. J. (1977). Group sequential methods in the design and analysis of clinical trials.
  *Biometrika* 64(2):191-199. <https://doi.org/10.1093/biomet/64.2.191>
- O'Brien, P. C., and Fleming, T. R. (1979). A multiple testing procedure for clinical trials.
  *Biometrics* 35(3):549-556. <https://doi.org/10.2307/2530245>
- Armitage, P., McPherson, C. K., and Rowe, B. C. (1969). Repeated significance tests on
  accumulating data. *Journal of the Royal Statistical Society A* 132(2):235-244.
  <https://doi.org/10.2307/2343787>
- Jennison, C., and Turnbull, B. W. (2000). *Group Sequential Methods with Applications to
  Clinical Trials*, chapter 2. Chapman and Hall/CRC. <https://doi.org/10.1201/9781584888581>
