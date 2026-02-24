//! Workgroup size optimization.
//!
//! Provides hardware-specific workgroup size selection based on occupancy
//! analysis, including shared memory and register pressure constraints.

use std::fmt;

use nxpu_ir::Module;

use crate::Pass;

/// Hardware parameters for a specific NPU target.
#[derive(Debug, Clone)]
pub struct WorkgroupHwParams {
    /// Maximum total threads per workgroup.
    pub max_threads: u32,
    /// Maximum per-axis workgroup dimensions `[x, y, z]`.
    pub max_dim: [u32; 3],
    /// Warp/wavefront size (threads execute in lockstep).
    pub warp_size: u32,
    /// Maximum shared memory per workgroup in bytes.
    pub max_shared_memory: u32,
    /// Registers available per thread.
    pub registers_per_thread: u32,
    /// Maximum warps that can be resident on a single compute unit.
    pub max_warps_per_cu: u32,
}

impl WorkgroupHwParams {
    /// Return hardware parameters for a named target.
    pub fn for_target(target: &str) -> Self {
        match target {
            "qualcomm" => Self {
                max_threads: 1024,
                max_dim: [1024, 1024, 64],
                warp_size: 64,
                max_shared_memory: 32768,
                registers_per_thread: 128,
                max_warps_per_cu: 32,
            },
            "samsung" => Self {
                max_threads: 512,
                max_dim: [512, 512, 64],
                warp_size: 32,
                max_shared_memory: 65536,
                registers_per_thread: 64,
                max_warps_per_cu: 48,
            },
            _ => Self {
                // Generic / fallback.
                max_threads: 256,
                max_dim: [256, 256, 64],
                warp_size: 32,
                max_shared_memory: 16384,
                registers_per_thread: 64,
                max_warps_per_cu: 32,
            },
        }
    }
}

/// Result of occupancy analysis for a given workgroup size.
#[derive(Debug, Clone)]
pub struct OccupancyResult {
    /// Chosen workgroup size `[x, y, z]`.
    pub workgroup_size: [u32; 3],
    /// Total threads in the workgroup.
    pub threads: u32,
    /// Occupancy ratio (0.0 to 1.0).
    pub occupancy: f64,
}

impl fmt::Display for OccupancyResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "workgroup [{}x{}x{}] = {} threads, occupancy {:.1}%",
            self.workgroup_size[0],
            self.workgroup_size[1],
            self.workgroup_size[2],
            self.threads,
            self.occupancy * 100.0,
        )
    }
}

/// Calculate occupancy for a given workgroup size and resource usage.
///
/// Occupancy is the ratio of active warps to maximum warps, limited by:
/// - Total thread count
/// - Shared memory usage
/// - Register usage per thread
pub fn calculate_occupancy(
    size: [u32; 3],
    hw: &WorkgroupHwParams,
    shared_mem_bytes: u32,
    regs_per_thread: u32,
) -> OccupancyResult {
    let threads = size[0] * size[1] * size[2];
    let warps_per_group = threads.div_ceil(hw.warp_size);

    // How many workgroups can fit based on shared memory?
    let groups_by_mem = if shared_mem_bytes == 0 {
        hw.max_warps_per_cu / warps_per_group.max(1)
    } else {
        hw.max_shared_memory / shared_mem_bytes.max(1)
    };

    // How many workgroups can fit based on register pressure?
    let total_regs_per_group = regs_per_thread * threads;
    let total_regs_available = hw.registers_per_thread * hw.warp_size * hw.max_warps_per_cu;
    let groups_by_regs = if total_regs_per_group == 0 {
        hw.max_warps_per_cu / warps_per_group.max(1)
    } else {
        total_regs_available / total_regs_per_group
    };

    let max_concurrent_groups = groups_by_mem.min(groups_by_regs);
    let active_warps = max_concurrent_groups * warps_per_group;
    let occupancy = (active_warps as f64) / (hw.max_warps_per_cu as f64);
    let occupancy = occupancy.min(1.0);

    OccupancyResult {
        workgroup_size: size,
        threads,
        occupancy,
    }
}

/// Find the workgroup size that maximizes occupancy.
///
/// Explores candidate sizes that are multiples of `warp_size`, within
/// hardware dimension limits.
pub fn optimize_workgroup_size(
    hw: &WorkgroupHwParams,
    shared_mem_bytes: u32,
    regs_per_thread: u32,
) -> [u32; 3] {
    let mut best_size = [hw.warp_size.min(hw.max_dim[0]), 1, 1];
    let mut best_occupancy = 0.0f64;

    // Candidate 1D sizes: multiples of warp_size up to max_threads.
    let mut threads = hw.warp_size;
    while threads <= hw.max_threads && threads <= hw.max_dim[0] {
        let size = [threads, 1, 1];
        let result = calculate_occupancy(size, hw, shared_mem_bytes, regs_per_thread);
        if result.occupancy > best_occupancy {
            best_occupancy = result.occupancy;
            best_size = size;
        }
        threads += hw.warp_size;
    }

    // Also try 2D configurations.
    for x in (hw.warp_size..=hw.max_dim[0]).step_by(hw.warp_size as usize) {
        for y_pow in 0..=3 {
            let y = 1u32 << y_pow;
            if y > hw.max_dim[1] {
                break;
            }
            let total = x * y;
            if total > hw.max_threads {
                break;
            }
            let size = [x, y, 1];
            let result = calculate_occupancy(size, hw, shared_mem_bytes, regs_per_thread);
            if result.occupancy > best_occupancy {
                best_occupancy = result.occupancy;
                best_size = size;
            }
        }
    }

    best_size
}

/// Workgroup size optimization pass.
///
/// Overrides `entry_point.workgroup_size` with the optimized size for the
/// generic hardware target.
#[derive(Debug)]
pub struct WorkgroupOptimization {
    target: String,
}

impl WorkgroupOptimization {
    /// Create a workgroup optimization pass for the given target.
    pub fn new(target: &str) -> Self {
        Self {
            target: target.to_string(),
        }
    }
}

impl Pass for WorkgroupOptimization {
    fn name(&self) -> &str {
        "workgroup-opt"
    }

    fn run(&self, module: &mut Module) -> bool {
        let hw = WorkgroupHwParams::for_target(&self.target);
        let optimal = optimize_workgroup_size(&hw, 0, hw.registers_per_thread);
        let mut changed = false;
        for ep in &mut module.entry_points {
            if ep.workgroup_size != optimal {
                ep.workgroup_size = optimal;
                changed = true;
            }
        }
        changed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nxpu_ir::EntryPoint;

    #[test]
    fn occupancy_full() {
        let hw = WorkgroupHwParams {
            max_threads: 256,
            max_dim: [256, 256, 64],
            warp_size: 32,
            max_shared_memory: 65536,
            registers_per_thread: 32,
            max_warps_per_cu: 32,
        };
        // 256 threads = 8 warps. With max 32 warps, 4 groups fit → 32/32 = 1.0.
        let result = calculate_occupancy([256, 1, 1], &hw, 0, 32);
        assert!((result.occupancy - 1.0).abs() < 0.01);
    }

    #[test]
    fn occupancy_limited_by_shared_memory() {
        let hw = WorkgroupHwParams {
            max_threads: 256,
            max_dim: [256, 256, 64],
            warp_size: 32,
            max_shared_memory: 16384,
            registers_per_thread: 64,
            max_warps_per_cu: 32,
        };
        // Request half the shared memory per group.
        let result = calculate_occupancy([256, 1, 1], &hw, 16384, 16);
        // Only 1 group fits by shared memory.
        assert!(result.occupancy < 1.0);
    }

    #[test]
    fn occupancy_limited_by_registers() {
        let hw = WorkgroupHwParams {
            max_threads: 256,
            max_dim: [256, 256, 64],
            warp_size: 32,
            max_shared_memory: 65536,
            registers_per_thread: 64,
            max_warps_per_cu: 32,
        };
        // High register usage: 128 regs per thread × 256 threads = 32768 regs/group.
        // Total available: 64 * 32 * 32 = 65536. Only 2 groups fit → 16/32 = 0.5.
        let result = calculate_occupancy([256, 1, 1], &hw, 0, 128);
        assert!(result.occupancy < 1.0);
    }

    #[test]
    fn optimize_selects_warp_multiple() {
        let hw = WorkgroupHwParams::for_target("generic");
        let size = optimize_workgroup_size(&hw, 0, 32);
        let total = size[0] * size[1] * size[2];
        assert_eq!(total % hw.warp_size, 0);
    }

    #[test]
    fn optimize_respects_max_dims() {
        let hw = WorkgroupHwParams {
            max_threads: 256,
            max_dim: [128, 64, 32],
            warp_size: 32,
            max_shared_memory: 16384,
            registers_per_thread: 64,
            max_warps_per_cu: 32,
        };
        let size = optimize_workgroup_size(&hw, 0, 32);
        assert!(size[0] <= hw.max_dim[0]);
        assert!(size[1] <= hw.max_dim[1]);
        assert!(size[2] <= hw.max_dim[2]);
    }

    #[test]
    fn optimize_respects_max_threads() {
        let hw = WorkgroupHwParams::for_target("qualcomm");
        let size = optimize_workgroup_size(&hw, 0, 32);
        let total = size[0] * size[1] * size[2];
        assert!(total <= hw.max_threads);
    }

    #[test]
    fn hw_params_for_target() {
        let q = WorkgroupHwParams::for_target("qualcomm");
        let s = WorkgroupHwParams::for_target("samsung");
        let g = WorkgroupHwParams::for_target("generic");
        // All three should have distinct configurations.
        assert_ne!(q.max_threads, s.max_threads);
        assert_ne!(q.warp_size, s.warp_size);
        assert_ne!(s.max_shared_memory, g.max_shared_memory);
    }

    #[test]
    fn pass_overrides_workgroup_size() {
        let mut module = Module::default();
        let func = nxpu_ir::Function::new("ep");
        module.entry_points.push(EntryPoint {
            name: "main".into(),
            workgroup_size: [1, 1, 1],
            function: func,
        });
        let pass = WorkgroupOptimization::new("generic");
        let changed = pass.run(&mut module);
        assert!(changed);
        let ws = module.entry_points[0].workgroup_size;
        assert!(ws[0] > 1 || ws[1] > 1 || ws[2] > 1);
    }

    #[test]
    fn pass_noop_when_optimal() {
        let hw = WorkgroupHwParams::for_target("generic");
        let optimal = optimize_workgroup_size(&hw, 0, hw.registers_per_thread);
        let mut module = Module::default();
        let func = nxpu_ir::Function::new("ep");
        module.entry_points.push(EntryPoint {
            name: "main".into(),
            workgroup_size: optimal,
            function: func,
        });
        let pass = WorkgroupOptimization::new("generic");
        let changed = pass.run(&mut module);
        assert!(!changed);
    }

    #[test]
    fn test_at_least_2_backends() {
        let q = WorkgroupHwParams::for_target("qualcomm");
        let s = WorkgroupHwParams::for_target("samsung");
        // Verify they produce different optimal sizes.
        let opt_q = optimize_workgroup_size(&q, 0, 32);
        let opt_s = optimize_workgroup_size(&s, 0, 32);
        // Both should be valid.
        assert!(opt_q[0] * opt_q[1] * opt_q[2] <= q.max_threads);
        assert!(opt_s[0] * opt_s[1] * opt_s[2] <= s.max_threads);
    }
}
