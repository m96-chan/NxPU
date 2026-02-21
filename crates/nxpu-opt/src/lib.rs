//! IR optimization passes for NxPU.
//!
//! Provides a [`Pass`] trait, a [`PassManager`] with fixed-point iteration,
//! and built-in optimization passes (constant folding, FMA fusion, dead code
//! elimination).

mod const_fold;
mod dce;
mod fma_fusion;
pub mod layout;
pub mod quantize;
pub mod shape;

pub use const_fold::ConstantFolding;
pub use dce::DeadCodeElimination;
pub use fma_fusion::FmaFusion;
pub use layout::LayoutTransform;
pub use quantize::{
    CalibrationData, F32ToBf16, F32ToF16, F32ToInt8, MixedPrecisionPass, MixedPrecisionPolicy,
    QuantizationParams,
};
pub use shape::ShapeInference;

use std::fmt::Debug;

use nxpu_ir::Module;

/// An optimization pass that transforms an IR module.
pub trait Pass: Debug {
    /// Human-readable name of the pass.
    fn name(&self) -> &str;

    /// Run the pass on a module. Returns `true` if anything was modified.
    fn run(&self, module: &mut Module) -> bool;
}

/// Optimization level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptLevel {
    /// No optimizations.
    O0,
    /// Basic optimizations (constant folding, FMA fusion, DCE).
    O1,
    /// Aggressive optimizations (same as O1 for now).
    O2,
}

/// Maximum number of fixed-point iterations before giving up.
const MAX_ITERATIONS: usize = 10;

/// Runs passes in sequence with fixed-point iteration.
pub struct PassManager {
    passes: Vec<Box<dyn Pass>>,
}

impl Default for PassManager {
    fn default() -> Self {
        Self::new()
    }
}

impl PassManager {
    /// Creates an empty pass manager with no passes.
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    /// Creates a pass manager with passes appropriate for the given level.
    pub fn for_level(level: OptLevel) -> Self {
        let mut pm = Self::new();
        match level {
            OptLevel::O0 => {}
            OptLevel::O1 | OptLevel::O2 => {
                pm.add_pass(Box::new(ConstantFolding));
                pm.add_pass(Box::new(FmaFusion));
                pm.add_pass(Box::new(DeadCodeElimination));
            }
        }
        pm
    }

    /// Adds a pass to the pipeline.
    pub fn add_pass(&mut self, pass: Box<dyn Pass>) {
        self.passes.push(pass);
    }

    /// Runs all passes until a fixed point is reached or the iteration limit.
    pub fn run(&self, module: &mut Module) {
        for _ in 0..MAX_ITERATIONS {
            let mut changed = false;
            for pass in &self.passes {
                changed |= pass.run(module);
            }
            if !changed {
                break;
            }
        }
    }
}

/// Convenience function: runs O1 optimization passes on a module.
pub fn optimize(module: &mut Module) {
    PassManager::for_level(OptLevel::O1).run(module);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimize_empty_module() {
        let mut module = Module::default();
        optimize(&mut module);
        // Should not panic; module remains empty.
        assert!(module.entry_points.is_empty());
    }

    #[test]
    fn pass_manager_o0_is_noop() {
        let pm = PassManager::for_level(OptLevel::O0);
        let mut module = Module::default();
        pm.run(&mut module);
        assert!(module.entry_points.is_empty());
    }

    #[test]
    fn pass_manager_o1_runs() {
        let pm = PassManager::for_level(OptLevel::O1);
        let mut module = Module::default();
        pm.run(&mut module);
        // No crash on empty module.
    }
}
