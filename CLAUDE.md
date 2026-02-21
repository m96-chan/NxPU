# NxPU — Project Guide

## What is this?
WGSL → NPU transpiler. Rust workspace layout.

## Build & Test
```sh
cargo build          # Build
cargo test           # Run all tests
cargo clippy         # Lint
cargo fmt --check    # Check formatting
```

## Conventions
- Language: Rust (edition 2024)
- Workspace: crates split under crates/
- Commit messages: English, imperative mood
- Error handling: thiserror + miette
