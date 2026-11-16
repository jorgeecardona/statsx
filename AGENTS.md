# AGENTS.md

Guidance for AI coding agents working in this repo. Humans: read README.md and CHANGELOG.md.

## Overview

statsx: Statistics for Python with its own Rust core: Bayesian inference, forecasting, causal impact, tests and design, with NumPy as the only dependency.

## Build

- `uv sync` installs dependencies. Toolchain (moon, uv) is pinned in `.prototools`; `proto install` brings it up.
- `make install-hooks` once, to enable the tracked git hooks.

## Test

- `make test` runs the suite.
- `make check` runs lint + typecheck + test (the same stage CI runs via `moon run :ci`).
- Everything in `make check` must pass before you commit.

## Style

- Format and autofix: `make format` (ruff, line length 100).
- Lint: `make lint`. Types: `make typecheck` (pyright strict on `src/`).
- Prefer frozen dataclasses and explicit types. Keep the public API typed (`py.typed`).

## Gotchas

- Version lives in `pyproject.toml` (maturin builds from it; keep `rust/Cargo.toml` in step). The pre-commit hook auto-bumps it when `src/` changes. Do not edit versions by hand elsewhere.
- Record follow-ups in `docs/issues/`, not as scattered TODO comments.
