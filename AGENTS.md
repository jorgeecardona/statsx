# AGENTS.md

Guidance for AI coding agents working in this repo. Humans: read README.md and CHANGELOG.md.

## Overview

statsx: statistics for experiments and forecasts, with a Rust extension and NumPy the only
dependency.

## Build

- `uv sync` installs dependencies. The toolchain (moon, uv) is pinned in `.prototools`;
  `proto install` installs it.
- `make install-hooks` once, to enable the tracked git hooks.

## Test

- `make test` runs the suite.
- `make check` runs lint, format-check, typecheck and test. CI runs `moon run :ci`, which adds
  test-floor, build, docs, version-guard and version-ahead.
- Everything in `make check` must pass before you commit.

## Style

- Format and autofix: `make format` (ruff, line length 100).
- Lint: `make lint`. Types: `make typecheck` (pyright strict on `src/` and `tests/`).
- Prefer frozen dataclasses and explicit types. Keep the public API typed (`py.typed`).

## Gotchas

- Version lives in `pyproject.toml` (maturin builds from it; keep `rust/Cargo.toml` in step). The
  pre-commit hook bumps the patch when a `.py` file under `src/` changes. Do not edit versions by
  hand elsewhere.
- Record follow-ups in `docs/issues/`, not as scattered TODO comments.
