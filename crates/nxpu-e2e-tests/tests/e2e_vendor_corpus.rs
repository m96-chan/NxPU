//! Classify every kernel in the vendored corpus.
//!
//! The recognisers in `nxpu-analysis` were written for these kernels, and
//! until this test existed none of them ran against one: the rest of the
//! suite uses inline WGSL, so the arms that handle a real convolution, a real
//! permutation or a real refusal were reached only by hand.
//!
//! Deliberately, this asserts almost nothing about *which* pattern comes back.
//! Pinning 52 answers would have to be rewritten every time classification
//! improves, and the improvements are the point. What it does assert is that
//! classification is total — no panic, no hang — and that it does not depend
//! on the optimization level, which is a property that has been broken twice.
//!
//! The answers themselves are pinned in the focused tests next to this one.

use std::path::{Path, PathBuf};

use nxpu_analysis::analyze;
use nxpu_opt::{OptLevel, PassManager};

fn corpus() -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("vendor/web-xpu-ops/ops");
    let Ok(ops) = std::fs::read_dir(&root) else {
        return Vec::new();
    };
    let mut found = Vec::new();
    for op in ops.filter_map(Result::ok) {
        let wgsl = op.path().join("wgsl");
        let Ok(entries) = std::fs::read_dir(&wgsl) else {
            continue;
        };
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "wgsl") {
                found.push(path);
            }
        }
    }
    found.sort();
    found
}

/// Classify at one level, folding every failure into a comparable string.
///
/// A source that cannot be classified has to fail the same way at every
/// level, so the errors are compared rather than skipped.
fn classify_at(source: &str, level: OptLevel) -> String {
    let mut module = match nxpu_parser::parse(source) {
        Ok(m) => m,
        Err(e) => return format!("parse error: {e}"),
    };
    PassManager::for_level(level).run(&mut module);
    if module.entry_points.is_empty() {
        return "no entry points".into();
    }
    match analyze::classify_entry_point(&module, 0) {
        Ok(pattern) => format!("{pattern:?}"),
        Err(e) => format!("error: {e}"),
    }
}

#[test]
fn every_vendored_kernel_classifies_the_same_at_every_optimization_level() {
    let corpus = corpus();
    if corpus.is_empty() {
        // The submodule is not checked out. Local clones without it should not
        // fail; CI fetches it, which is where this test earns its keep.
        eprintln!("vendor/web-xpu-ops is empty — run `git submodule update --init`");
        return;
    }

    let mut disagreed = Vec::new();
    for path in &corpus {
        let source = std::fs::read_to_string(path).expect("read kernel");
        let o0 = classify_at(&source, OptLevel::O0);
        let o1 = classify_at(&source, OptLevel::O1);
        let o2 = classify_at(&source, OptLevel::O2);
        if o0 != o1 || o1 != o2 {
            disagreed.push(format!(
                "{}\n      O0: {o0}\n      O1: {o1}\n      O2: {o2}",
                path.display()
            ));
        }
    }
    assert!(
        disagreed.is_empty(),
        "classification depends on the optimization level for {} kernel(s):\n  {}",
        disagreed.len(),
        disagreed.join("\n  ")
    );
}

#[test]
fn the_corpus_is_where_it_is_expected_to_be() {
    // A silent empty corpus would make the test above pass while checking
    // nothing, which is the failure mode this whole branch exists to remove.
    // In CI the submodule is checked out, so the corpus must be found.
    if std::env::var_os("CI").is_none() {
        return;
    }
    assert!(
        corpus().len() >= 40,
        "expected the vendored corpus, found {} kernels — is the submodule checked out?",
        corpus().len()
    );
}
