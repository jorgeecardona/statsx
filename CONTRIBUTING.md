# Contributing to statsx

Developer and maintainer notes: toolchain, the dev loop, CI and the release process. Package
users need none of this: the [README](README.md) says what statsx does and how to use it.

## Contributing with an AI agent

Most changes here are written with a coding agent. That is expected and welcome. For a good
result:

- Point the agent at [`AGENTS.md`](AGENTS.md) first. It has the build and test commands, the
  typed-API and style rules, and the repo-specific gotchas. Agents read it when started from the
  repo root.
- Run `make check` (lint, format-check, typecheck, test) and fix what it finds before you open the
  MR.
- Do not bypass the hooks. The pre-commit hook bumps the patch version when a `.py` file under
  `src/` changes, and CI's version-guard checks it: a `--no-verify` commit fails the pipeline.
- Keep the diff small and typed (pyright runs strict), add a `CHANGELOG.md` entry, and do not
  reformat unrelated files.

## Development and CI

The toolchain is pinned with [proto](https://moonrepo.dev/proto) (`.prototools`: `moon` + `uv`)
and every CI-graded task is a [moon](https://moonrepo.dev) target running through `uv run`. After
cloning:

```bash
proto install          # installs the pinned moon + uv
uv sync                # install dependencies into .venv
make install-hooks     # pre-commit auto-bump + pre-push checks (one-time)
```

The dev loop (`make` verbs are aliases of the moon tasks that CI runs):

```bash
make format            # auto-format + autofix lint (ruff, line length 100)
make check             # lint + format-check + typecheck + test
make docs              # build the docs site strictly (as the CI ci job does)
```

CI has three GitLab jobs:

- `ci` runs on every pipeline. `proto install` installs the toolchain, then one `moon run :ci`
  runs lint, format-check, typecheck, test, test-floor (the suite on Python 3.12), build, docs,
  version-guard and version-ahead. On `main` it runs `moon run :release`, which runs the same
  tasks and then publishes.
- `pages` runs on `main` and publishes the docs site that `ci` built.
- `wheels-macos` runs on `main` when `pyproject.toml` changed (manual on other pushes) and uploads
  the macOS wheels.

moon orders the tasks, so run them through `make` or `moon run`.

### Native accelerator (Rust)

The `statsx._accel` extension is built from `rust/Cargo.toml` by
[maturin](https://www.maturin.rs/) and shipped inside the same wheel (abi3, one wheel per
platform for Python 3.12+). For local development:

```bash
maturin develop --release    # from the repo root, builds _accel into .venv
```

## Versioning

The version lives in `pyproject.toml`. maturin builds from it, and `rust/Cargo.toml` has the same
number. Do not edit it by hand:

- The pre-commit hook bumps the patch version when a `.py` file under `src/` changes.
- CI's `version-guard` fails when a `src/` change has no bump. It catches `--no-verify` commits
  and clones with no hooks.
- CI's `version-ahead` fails unless the version is greater than the one on `origin/main`.

For a minor or major release, run `make bump TYPE=minor` (or `TYPE=major`).

## Releasing

Releases are automated. There is no manual tag or publish step:

1. The pre-commit hook bumps the patch version when a `.py` file under `src/` changes (run
   `make install-hooks` once after cloning). Commits that change only docs, tests, examples or
   config do not bump.
2. `version-guard` (part of `:ci`) checks the same rule in CI.
3. `:release` runs in the `ci` job on `main`, after all `:ci` tasks pass. If the package version
   is not on PyPI yet, it publishes with OIDC Trusted Publishing. A version bump that reaches
   `main` is released by that pipeline, with no tag.

### One-time setup

PyPI uses OIDC Trusted Publishing, so no API token is stored. In the PyPI account, open
Publishing and add a pending GitLab publisher (or add it to the project after the first manual
upload):

| Field | Value |
|-------|-------|
| PyPI Project Name | `statsx` |
| Namespace | `cardona-oss` |
| Project name (repo) | `statsx` |
| Top-level pipeline file | `.gitlab-ci.yml` |
| Environment name | `pypi` |

Branch and MR pipelines run `moon run :ci` only. Publishing happens only from `main`, after the
`:ci` tasks pass.

## License

By contributing, you agree that your contributions are licensed under the MIT License, the same
terms as the project (see [LICENSE](LICENSE)).
