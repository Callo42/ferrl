# Contributing to ferrl

Thanks for contributing! This project keeps a strict, automated quality bar so the
codebase stays maintainable and scalable. Please read this before opening a PR.

## Workflow — pull requests only

- **Never push directly to `main`.** It is protected; all changes land via a pull
  request from a feature branch.
- **Every PR is reviewed at least once before merge**, and **CI must be green**
  before it can be merged. Both are enforced by branch protection.
- Branch naming: `feat/…`, `fix/…`, `docs/…`, `refactor/…`, `test/…`, `ci/…`.
- Keep PRs focused and reviewable.

## Commits

- **[Conventional Commits](https://www.conventionalcommits.org/)** are enforced via
  [cocogitto](https://docs.cocogitto.io/) (`cog check`). Versioning is SemVer; tags
  are `vX.Y.Z`.
- Example: `feat(lora): add low-rank adapter to attention projections`.

## Quality gate

CI runs on every push and PR (CPU; GPU work is manual, never in CI):

| Check | Command |
|---|---|
| Format | `cargo fmt --all --check` |
| Lint | `cargo clippy --all-targets -- -D warnings` |
| Tests + coverage (≥ 90%) | `cargo llvm-cov --fail-under-lines 90` |
| Docs | `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps` |

Run the whole bar locally before pushing:

```sh
just bootstrap   # one-time: toolchain components + pre-commit + cargo-llvm-cov
just gate        # fmt + clippy + check + test + doc
```

The toolchain is pinned in `rust-toolchain.toml`. Lints (`forbid(unsafe_code)`,
`deny(missing_docs)`, a curated clippy set) live in `Cargo.toml [workspace.lints]`.

## Security & privacy

- **No secrets, credentials, or personal data in commits.** Secret-scanning push
  protection is enabled on the repo, and `pre-commit` runs `detect-private-key`.
- Don't commit machine-specific paths, tokens, or private infrastructure details.

## Editing CI

GitHub validates `.github/workflows/*.yml` at startup; a YAML parse error fails the
run in **0 seconds with no annotation**, and `cargo` never reads the workflow — so
validate workflow edits with a real YAML parser, not just a local build.
