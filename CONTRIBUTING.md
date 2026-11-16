# Contributing to statsx

Developer and maintainer notes: toolchain, the dev loop, CI, and the release process. (Package
*users* need none of this — see the [README](README.md) for what statsx does and how
to use it.)

## Contributing with an AI agent

Most changes here will be written with a coding agent — that's expected and welcome. To get a good
result from yours:

- **Point it at [`AGENTS.md`](AGENTS.md) first** — the machine-facing contract for this repo: build
  and test commands, the typed-API and style rules, and the repo-specific gotchas. Agents read it
  automatically when started from the repo root.
- **Ship a green `make check`** (lint + typecheck + test). Have the agent run it and fix what it
  finds *before* opening the MR, not leave it for review.
- **Don't let it bypass the hooks.** The patch version auto-bumps on `src/` changes and CI's
  version-guard enforces it; a `--no-verify` commit fails the pipeline.
- **Keep the diff small and typed** (pyright runs strict), add a `CHANGELOG.md` entry, and don't
  reformat unrelated files.

## Development & CI

The toolchain is pinned with [proto](https://moonrepo.dev/proto) (`.prototools`: `moon` + `uv`)
and every CI-graded task is a [moon](https://moonrepo.dev) target running through `uv run`. After
cloning:

```bash
proto install          # installs the pinned moon + uv
uv sync                # install dependencies into .venv
make install-hooks     # pre-commit auto-bump + pre-push checks (one-time)
```

The dev loop (thin `make` aliases over the moon tasks, so `make` runs exactly what CI runs):

```bash
make format            # auto-format + autofix lint (ruff, line length 100)
make check             # the full test stage: lint + typecheck + test
make docs              # build the docs site strictly (as CI / the pages job do)
```

CI is a single GitLab job: `proto install` brings up the toolchain, then one `moon run :ci`
resolves the whole graph (lint, typecheck, test, build, docs, version-guard) — there are no
per-language jobs or hand-wired stages. moon owns task ordering, so run tasks through `make` or
`moon run`, never by hand-wiring them.

### Native accelerator (Rust)

The `statsx._accel` extension is built from `rust/Cargo.toml` by
[maturin](https://www.maturin.rs/) and shipped inside the same wheel (abi3, one wheel per
platform for Python 3.12+). For local development:

```bash
maturin develop --release    # from the repo root, builds _accel into .venv
```

## Versioning

The version lives in `pyproject.toml` — the single source of truth maturin builds from (keep
`rust/Cargo.toml` in step). Do not edit it by hand: the pre-commit hook auto-bumps the patch version when anything under `src/` changes,
and CI's `version-guard` enforces non-bypassably that a `src/` change carries a bump (catching
`--no-verify` commits and unhooked clones); `version-ahead` additionally enforces that the version
is strictly greater than `origin/main`. For an intentional minor/major release, bump deliberately:
`make bump TYPE=minor`.

## Releasing

Releases are automated — there is no manual tag or publish step:

1. **`pre-commit` auto-bumps** the patch version on any `src/` change (`make install-hooks` once
   after cloning). Doc/test/example/config-only commits don't bump.
2. **`version-guard`** (part of `:ci`) enforces the same rule in CI.
3. **`:release`** runs on a green `main` pipeline (after the full `:ci` graph as deps): if the
   package version isn't on PyPI yet, it publishes via OIDC Trusted Publishing. So merging a
   version bump to `main` releases itself — no second pipeline, no tag required.

### One-time setup

**PyPI — OIDC Trusted Publishing** (no API token is stored anywhere). PyPI account →
Publishing → add a *pending* GitLab publisher (or add it to the project after the first manual
upload):

| Field | Value |
|-------|-------|
| PyPI Project Name | `statsx` |
| Namespace | `cardona-oss` |
| Project name (repo) | `statsx` |
| Top-level pipeline file | `.gitlab-ci.yml` |
| Environment name | `pypi` |

Pushes and MRs run lint + typecheck + test + build + docs only; publishing happens only from a
green `main`.

## License

By contributing, you agree that your contributions are licensed under the MIT License, the same
terms as the project (see [LICENSE](LICENSE)).
