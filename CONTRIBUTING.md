# Contributing to ferrl

Thanks for contributing! This project keeps a strict, automated quality bar so the
codebase stays maintainable and scalable. Please read this before opening a PR.

## Workflow — pull requests only

- **Never push directly to `main`.** It is protected; all changes land via a pull
  request from a feature branch.
- **Every PR is reviewed by a human maintainer before merge**, and **CI must be
  green**. Branch protection enforces the PR path and required checks; the
  project review policy is human-enforced because this solo-maintainer repo does
  not encode a required approval count that its owner could not satisfy.
- Branch naming: `feat/…`, `fix/…`, `docs/…`, `refactor/…`, `test/…`, `ci/…`.
- Keep PRs focused and reviewable.

## Commits

- **[Conventional Commits](https://www.conventionalcommits.org/)** are enforced via
  [cocogitto](https://docs.cocogitto.io/) (`cog check`). Versioning is SemVer; tags
  are `vX.Y.Z`.
- Example: `feat(lora): add low-rank adapter to attention projections`.

## Quality gate

CI runs on pushes to `main` and on pull requests targeting `main` (CPU; GPU
work is manual, never in CI):

| Check | Command |
|---|---|
| Format | `cargo fmt --all --check` |
| Lint | `cargo clippy --all-targets -- -D warnings` plus the `gate`-feature TriMul target |
| Tests + coverage (≥ 90%) | `cargo llvm-cov --workspace --fail-under-lines 90`, doctests, verifier-driver syntax, and the `gate`-feature TriMul test |
| Docs | `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps` |
| Supply chain | `cargo deny check` and `cargo audit` |
| MSRV | `cargo +1.87 check --locked --workspace` |
| Commit range (PRs) | `cog check --from-latest-tag` |

Run the core CPU bar locally before pushing:

```sh
just bootstrap   # one-time: toolchain components + pre-commit + cargo-llvm-cov
just gate        # fmt + clippy + check + test + coverage + docs
```

GitHub CI additionally compiles/tests the feature-gated TriMul verifier contract
and runs the supply-chain, MSRV, and PR commit-range jobs listed above. GPU
feature builds and runtime gates remain manual because GitHub runners are CPU-only.

The toolchain is pinned in `rust-toolchain.toml`. Lints (`deny(unsafe_code)` — the
default build is `unsafe`-free; the optional `--features nccl` FFI module is the one
gated exception — `deny(missing_docs)`, a curated clippy set) live in
`Cargo.toml [workspace.lints]`.

## Security & privacy

- **No secrets, credentials, or personal data in commits.** Secret-scanning push
  protection is enabled on the repo, and `pre-commit` runs `detect-private-key`.
- Don't commit machine-specific paths, tokens, or private infrastructure details.

## Editing CI

GitHub validates `.github/workflows/*.yml` at startup; a YAML parse error fails the
run in **0 seconds with no annotation**, and `cargo` never reads the workflow — so
validate workflow edits with a real YAML parser, not just a local build.
