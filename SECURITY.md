# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

If you discover a security vulnerability in NxPU, please report it responsibly:

1. **Do not** open a public GitHub issue.
2. Email your report to the maintainers via the contact information in the repository.
3. Include a clear description of the vulnerability, steps to reproduce, and potential impact.
4. Allow reasonable time for a fix before public disclosure.

We aim to acknowledge reports within 48 hours and provide an initial assessment within 5 business days.

## Scope

NxPU is a compiler/transpiler that processes WGSL source code. Security concerns include:

- **Denial of service**: Maliciously crafted WGSL input causing excessive resource consumption.
- **Memory safety**: Unsafe memory access during IR processing.
- **Output integrity**: Incorrect code generation that could cause unexpected behavior on target hardware.

## Security Practices

- The project is written in safe Rust with no `unsafe` blocks in core crates.
- All dependencies are tracked via `Cargo.lock`.
- CI runs `cargo clippy` with `-D warnings` to catch common issues.
