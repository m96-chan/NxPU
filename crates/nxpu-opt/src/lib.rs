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
mod validation;

pub use const_fold::ConstantFolding;
pub use dce::DeadCodeElimination;
pub use fma_fusion::FmaFusion;
pub use layout::LayoutTransform;
pub use quantize::{
    CalibrationData, F32ToBf16, F32ToF16, F32ToInt8, MixedPrecisionPass, MixedPrecisionPolicy,
    QuantizationParams,
};
pub use shape::ShapeInference;
pub use validation::{IrValidation, ValidationWarning, collect_warnings};

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
    /// Passes run once before the fixed-point loop (e.g. validation).
    pre_passes: Vec<Box<dyn Pass>>,
    /// Passes run in the fixed-point loop.
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
        Self {
            pre_passes: Vec::new(),
            passes: Vec::new(),
        }
    }

    /// Creates a pass manager with passes appropriate for the given level.
    pub fn for_level(level: OptLevel) -> Self {
        let mut pm = Self::new();
        match level {
            OptLevel::O0 => {}
            OptLevel::O1 | OptLevel::O2 => {
                pm.add_pre_pass(Box::new(IrValidation));
                pm.add_pass(Box::new(ConstantFolding));
                pm.add_pass(Box::new(FmaFusion));
                pm.add_pass(Box::new(DeadCodeElimination));
            }
        }
        pm
    }

    /// Adds a pass to run once before the fixed-point loop.
    pub fn add_pre_pass(&mut self, pass: Box<dyn Pass>) {
        self.pre_passes.push(pass);
    }

    /// Adds a pass to the fixed-point pipeline.
    pub fn add_pass(&mut self, pass: Box<dyn Pass>) {
        self.passes.push(pass);
    }

    /// Runs pre-passes once, then iterates the main passes until a fixed point
    /// is reached or the iteration limit.
    pub fn run(&self, module: &mut Module) {
        for pass in &self.pre_passes {
            pass.run(module);
            log::debug!("pre-pass '{}' completed", pass.name());
        }

        for iteration in 0..MAX_ITERATIONS {
            let mut changed = false;
            for pass in &self.passes {
                let pass_changed = pass.run(module);
                log::debug!(
                    "pass '{}' iteration {}: changed={}",
                    pass.name(),
                    iteration,
                    pass_changed
                );
                changed |= pass_changed;
            }
            if !changed {
                log::debug!("fixed point reached after {} iteration(s)", iteration + 1);
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
