# Contributing to NxPU

Thank you for your interest in contributing to NxPU!

## Prerequisites

- **Rust 1.87+** (edition 2024)
- A working `cargo` toolchain

## Building

```sh
cargo build
```

## Running Tests

```sh
cargo test --workspace
```

## Linting and Formatting

Before submitting a PR, ensure your code passes:

```sh
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
```

## Fuzz Testing

The WGSL parser has fuzz targets under `fuzz/`. Requires nightly Rust:

```sh
cargo install cargo-fuzz
cargo +nightly fuzz run fuzz_parse   # Fuzz the WGSL parser (naga)
cargo +nightly fuzz run fuzz_lower   # Fuzz parse + IR lowering
```

Stop with `Ctrl+C`. Crashes are saved in `fuzz/artifacts/`. If you find one, please open an issue.

## Pull Request Guidelines

1. Fork the repository and create a feature branch from `main`.
2. Write clear, focused commits with imperative-mood messages.
3. Add tests for new functionality.
4. Ensure all existing tests continue to pass.
5. Keep PRs small and reviewable when possible.

## Project Structure

The workspace is organized under `crates/`:

| Crate | Purpose |
|-------|---------|
| `nxpu-ir` | Intermediate representation |
| `nxpu-parser` | WGSL parsing and IR lowering |
| `nxpu-opt` | Optimization passes |
| `nxpu-analysis` | Pattern classification and fusion |
| `nxpu-backend-core` | Backend trait and registry |
| `nxpu-backend-*` | Target-specific code generators |
| `nxpu-cli` | Command-line interface |
| `nxpu-e2e-tests` | End-to-end integration tests |

See `docs/architecture.md` for a detailed overview.

## Adding a New Backend

See `docs/adding-a-backend.md` for a step-by-step guide.

## Code of Conduct

Be respectful and constructive. We follow the [Rust Code of Conduct](https://www.rust-lang.org/policies/code-of-conduct).
