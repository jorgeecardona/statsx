"""Errors that mean statsx could not give an answer (GitLab issue #7).

One ``except StatsxError`` catches both kinds, and each kind keeps its builtin base, so a handler
for ``ValueError`` or ``RuntimeError`` written before this module still works.

- ``UnfittableInput``: the input cannot be fitted at all (too few months for the harmonics, a
  negative count, a constant regressor). Retrying with the same input fails again.
- ``FitCheckError`` (in ``statsx.trend``): the fit ran and did not pass its own check. A retry with
  other settings can work.

An argument that is out of range (``n_draws``, ``level``) stays a plain ``ValueError``: it is a
mistake in the call, not a property of the data.
"""

from __future__ import annotations


class StatsxError(Exception):
    """Base of every error that means statsx could not give an answer."""


class UnfittableInput(StatsxError, ValueError):  # noqa: N818 (named for the input, not an event)
    """The input cannot be fitted; the same input fails again, so change the data or the model."""
