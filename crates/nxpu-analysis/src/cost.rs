//! Cost model and latency prediction for kernel operations.
//!
//! Provides per-op FLOP/memory estimation, roofline analysis,
//! and hardware profile–based latency prediction.

use std::collections::HashMap;
use std::fmt;

use crate::KernelPattern;

/// Per-operation cost in terms of compute and memory traffic.
#[derive(Debug, Clone, PartialEq)]
pub struct OpCost {
    /// Total floating-point operations.
    pub flops: u64,
    /// Bytes read from memory.
    pub bytes_read: u64,
    /// Bytes written to memory.
    pub bytes_written: u64,
}

impl OpCost {
    /// Arithmetic intensity: FLOPs per byte of memory traffic.
    pub fn arithmetic_intensity(&self) -> f64 {
        let total_bytes = self.bytes_read + self.bytes_written;
        if total_bytes == 0 {
            return 0.0;
        }
        self.flops as f64 / total_bytes as f64
    }
}

impl fmt::Display for OpCost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "OpCost {{ flops: {}, bytes_read: {}, bytes_written: {} }}",
            self.flops, self.bytes_read, self.bytes_written
        )
    }
}

/// Whether an operation is bottlenecked by compute or memory bandwidth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bottleneck {
    /// FLOPs exceed what can be hidden by memory latency.
    ComputeBound,
    /// Memory traffic exceeds what can be hidden by compute.
    MemoryBound,
}

impl fmt::Display for Bottleneck {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::ComputeBound => "compute-bound",
            Self::MemoryBound => "memory-bound",
        })
    }
}

/// Hardware profile for roofline analysis.
#[derive(Debug, Clone)]
pub struct HardwareProfile {
    /// Peak throughput in GFLOPS.
    pub peak_gflops: f64,
    /// Memory bandwidth in GB/s.
    pub memory_bandwidth_gbs: f64,
    /// Human-readable profile name.
    pub name: String,
}

impl HardwareProfile {
    /// Ridge point: arithmetic intensity where compute and memory are balanced.
    pub fn ridge_point(&self) -> f64 {
        if self.memory_bandwidth_gbs == 0.0 {
            return 0.0;
        }
        self.peak_gflops / self.memory_bandwidth_gbs
    }

    /// Predict execution latency in seconds using the roofline model.
    ///
    /// `latency = max(flops / peak_flops_per_sec, bytes / bandwidth_per_sec)`
    pub fn predict_latency_secs(&self, cost: &OpCost) -> f64 {
        let compute_secs = cost.flops as f64 / (self.peak_gflops * 1e9);
        let total_bytes = (cost.bytes_read + cost.bytes_written) as f64;
        let memory_secs = total_bytes / (self.memory_bandwidth_gbs * 1e9);
        compute_secs.max(memory_secs)
    }

    /// Determine the bottleneck for a given operation cost.
    pub fn bottleneck(&self, cost: &OpCost) -> Bottleneck {
        let ai = cost.arithmetic_intensity();
        if ai >= self.ridge_point() {
            Bottleneck::ComputeBound
        } else {
            Bottleneck::MemoryBound
        }
    }
}

/// Estimate the compute and memory cost of a classified kernel pattern.
///
/// Element size defaults to 4 bytes (f32) unless the pattern provides
/// enough information to determine otherwise.
pub fn estimate_kernel_cost(pattern: &KernelPattern) -> OpCost {
    const ELEM_SIZE: u64 = 4; // f32

    match pattern {
        KernelPattern::MatMul { shape, .. } => {
            // Symbolic dims — use dummy size for cost estimation.
            // For actual numeric shapes, callers can provide concrete dims.
            // FLOPs = 2 * M * N * K (multiply-accumulate)
            let (m, n, k) = parse_matmul_dims(shape);
            let flops = 2 * m * n * k;
            let bytes_read = (m * k + k * n) * ELEM_SIZE;
            let bytes_written = m * n * ELEM_SIZE;
            OpCost {
                flops,
                bytes_read,
                bytes_written,
            }
        }
        KernelPattern::Conv2D { shape, .. } => {
            let (batch, cin, cout, oh, ow, kh, kw, groups) = parse_conv2d_dims(shape);
            // FLOPs = 2 * batch * cout * oh * ow * (cin/groups) * kh * kw
            let flops = 2 * batch * cout * oh * ow * (cin / groups) * kh * kw;
            // Input read: batch * cin * (oh + kh - 1) * (ow + kw - 1) * elem
            let ih = oh + kh - 1;
            let iw = ow + kw - 1;
            let bytes_read = (batch * cin * ih * iw + cout * (cin / groups) * kh * kw) * ELEM_SIZE;
            let bytes_written = batch * cout * oh * ow * ELEM_SIZE;
            OpCost {
                flops,
                bytes_read,
                bytes_written,
            }
        }
        KernelPattern::ElementWise { op, .. } => {
            let n = parse_dim_name_default();
            let flops = n;
            // Read 2 inputs, write 1 output
            let bytes_read = 2 * n * ELEM_SIZE;
            let bytes_written = n * ELEM_SIZE;
            let _ = op;
            OpCost {
                flops,
                bytes_read,
                bytes_written,
            }
        }
        KernelPattern::Activation { .. } => {
            let n = parse_dim_name_default();
            let flops = n;
            // Read 1 input, write 1 output
            let bytes_read = n * ELEM_SIZE;
            let bytes_written = n * ELEM_SIZE;
            OpCost {
                flops,
                bytes_read,
                bytes_written,
            }
        }
        KernelPattern::Pool { shape, .. } => {
            let (batch, cout, oh, ow) = (1u64, 1, 64, 64);
            let kh = shape.kernel_h as u64;
            let kw = shape.kernel_w as u64;
            let flops = batch * cout * oh * ow * kh * kw;
            let bytes_read = batch * cout * (oh + kh - 1) * (ow + kw - 1) * ELEM_SIZE;
            let bytes_written = batch * cout * oh * ow * ELEM_SIZE;
            OpCost {
                flops,
                bytes_read,
                bytes_written,
            }
        }
        KernelPattern::Reduce { .. } => {
            let n = parse_dim_name_default();
            let flops = n;
            let bytes_read = n * ELEM_SIZE;
            let bytes_written = ELEM_SIZE; // scalar output
            OpCost {
                flops,
                bytes_read,
                bytes_written,
            }
        }
        _ => OpCost {
            flops: 0,
            bytes_read: 0,
            bytes_written: 0,
        },
    }
}

/// Estimate cost with explicit numeric dimensions for a MatMul pattern.
pub fn estimate_matmul_cost(m: u64, n: u64, k: u64) -> OpCost {
    const ELEM_SIZE: u64 = 4;
    OpCost {
        flops: 2 * m * n * k,
        bytes_read: (m * k + k * n) * ELEM_SIZE,
        bytes_written: m * n * ELEM_SIZE,
    }
}

/// Estimate cost with explicit numeric dimensions for a Conv2D pattern.
#[allow(clippy::too_many_arguments)]
pub fn estimate_conv2d_cost(
    batch: u64,
    cin: u64,
    cout: u64,
    oh: u64,
    ow: u64,
    kh: u64,
    kw: u64,
    groups: u64,
) -> OpCost {
    const ELEM_SIZE: u64 = 4;
    let g = if groups == 0 { 1 } else { groups };
    let flops = 2 * batch * cout * oh * ow * (cin / g) * kh * kw;
    let ih = oh + kh - 1;
    let iw = ow + kw - 1;
    let bytes_read = (batch * cin * ih * iw + cout * (cin / g) * kh * kw) * ELEM_SIZE;
    let bytes_written = batch * cout * oh * ow * ELEM_SIZE;
    OpCost {
        flops,
        bytes_read,
        bytes_written,
    }
}

/// Estimate cost for an element-wise operation with known element count.
pub fn estimate_elementwise_cost(n: u64) -> OpCost {
    const ELEM_SIZE: u64 = 4;
    OpCost {
        flops: n,
        bytes_read: 2 * n * ELEM_SIZE,
        bytes_written: n * ELEM_SIZE,
    }
}

/// Estimate cost for an activation operation with known element count.
pub fn estimate_activation_cost(n: u64) -> OpCost {
    const ELEM_SIZE: u64 = 4;
    OpCost {
        flops: n,
        bytes_read: n * ELEM_SIZE,
        bytes_written: n * ELEM_SIZE,
    }
}

/// Return default hardware profiles for known backends.
pub fn default_profiles() -> HashMap<&'static str, HardwareProfile> {
    let mut map = HashMap::new();
    map.insert(
        "onnx",
        HardwareProfile {
            peak_gflops: 100.0,
            memory_bandwidth_gbs: 50.0,
            name: "ONNX Runtime (Generic CPU)".into(),
        },
    );
    map.insert(
        "tflite",
        HardwareProfile {
            peak_gflops: 50.0,
            memory_bandwidth_gbs: 25.0,
            name: "TFLite (Mobile CPU)".into(),
        },
    );
    map.insert(
        "arm-ethos",
        HardwareProfile {
            peak_gflops: 4.0,
            memory_bandwidth_gbs: 16.0,
            name: "Arm Ethos-U65".into(),
        },
    );
    map
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn parse_matmul_dims(shape: &crate::MatMulShape) -> (u64, u64, u64) {
    let m = parse_dim(&shape.m);
    let n = parse_dim(&shape.n);
    let k = parse_dim(&shape.k);
    (m, n, k)
}

fn parse_conv2d_dims(shape: &crate::Conv2DShape) -> (u64, u64, u64, u64, u64, u64, u64, u64) {
    let batch = parse_dim(&shape.batch);
    let cin = parse_dim(&shape.channels_in);
    let cout = parse_dim(&shape.channels_out);
    let height = parse_dim(&shape.height);
    let width = parse_dim(&shape.width);
    let kh = shape.kernel_h_val.max(1) as u64;
    let kw = shape.kernel_w_val.max(1) as u64;
    let groups = shape.groups.max(1) as u64;
    // Approximate output spatial dims accounting for stride and padding.
    let sh = shape.stride_h.max(1) as u64;
    let sw = shape.stride_w.max(1) as u64;
    let ph = shape.pad_h.max(0) as u64;
    let pw = shape.pad_w.max(0) as u64;
    let oh = (height + 2 * ph - kh) / sh + 1;
    let ow = (width + 2 * pw - kw) / sw + 1;
    (batch, cin, cout, oh, ow, kh, kw, groups)
}

/// Try to parse a numeric dimension string; fall back to a default.
fn parse_dim(s: &str) -> u64 {
    s.parse::<u64>().unwrap_or(256)
}

/// Default element count for symbolic dimension names.
fn parse_dim_name_default() -> u64 {
    256
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_flop_count() {
        let cost = estimate_matmul_cost(128, 128, 64);
        assert_eq!(cost.flops, 2 * 128 * 128 * 64);
    }

    #[test]
    fn conv2d_flop_count() {
        // batch=1, cin=3, cout=16, oh=32, ow=32, kh=3, kw=3, groups=1
        let cost = estimate_conv2d_cost(1, 3, 16, 32, 32, 3, 3, 1);
        assert_eq!(cost.flops, 2 * 16 * 32 * 32 * 3 * 3 * 3);
    }

    #[test]
    fn elementwise_cost() {
        let cost = estimate_elementwise_cost(256);
        assert_eq!(cost.flops, 256);
        assert_eq!(cost.bytes_read, 2 * 256 * 4);
        assert_eq!(cost.bytes_written, 256 * 4);
    }

    #[test]
    fn roofline_compute_bound() {
        // Large MatMul → high arithmetic intensity → compute-bound.
        let cost = estimate_matmul_cost(1024, 1024, 1024);
        let profile = HardwareProfile {
            peak_gflops: 100.0,
            memory_bandwidth_gbs: 200.0, // very high BW
            name: "test".into(),
        };
        assert_eq!(profile.bottleneck(&cost), Bottleneck::ComputeBound);
    }

    #[test]
    fn roofline_memory_bound() {
        // Small element-wise → low arithmetic intensity → memory-bound.
        let cost = estimate_elementwise_cost(64);
        let profile = HardwareProfile {
            peak_gflops: 1000.0, // very high compute
            memory_bandwidth_gbs: 10.0,
            name: "test".into(),
        };
        assert_eq!(profile.bottleneck(&cost), Bottleneck::MemoryBound);
    }

    #[test]
    fn predict_latency_value() {
        let cost = OpCost {
            flops: 1_000_000_000, // 1 GFLOP
            bytes_read: 0,
            bytes_written: 0,
        };
        let profile = HardwareProfile {
            peak_gflops: 100.0,
            memory_bandwidth_gbs: 50.0,
            name: "test".into(),
        };
        let latency = profile.predict_latency_secs(&cost);
        // 1e9 / (100 * 1e9) = 0.01 seconds
        assert!((latency - 0.01).abs() < 1e-9);
    }

    #[test]
    fn default_profiles_has_3_entries() {
        let profiles = default_profiles();
        assert!(profiles.len() >= 3);
        assert!(profiles.contains_key("onnx"));
        assert!(profiles.contains_key("tflite"));
        assert!(profiles.contains_key("arm-ethos"));
    }

    #[test]
    fn bottleneck_at_ridge_point() {
        let profile = HardwareProfile {
            peak_gflops: 100.0,
            memory_bandwidth_gbs: 50.0,
            name: "test".into(),
        };
        let ridge = profile.ridge_point(); // 2.0
        // At exactly the ridge point, AI >= ridge → ComputeBound.
        let total_bytes = 1000u64;
        let flops = (ridge * total_bytes as f64) as u64;
        let cost = OpCost {
            flops,
            bytes_read: total_bytes,
            bytes_written: 0,
        };
        assert_eq!(profile.bottleneck(&cost), Bottleneck::ComputeBound);
    }

    #[test]
    fn arithmetic_intensity() {
        let cost = OpCost {
            flops: 1000,
            bytes_read: 400,
            bytes_written: 100,
        };
        let ai = cost.arithmetic_intensity();
        assert!((ai - 2.0).abs() < 1e-9);
    }

    #[test]
    fn cost_model_used_by_schedule() {
        // Verify that OpCost can produce cost values consumed by critical_path_with_costs.
        use crate::DataflowGraph;
        use nxpu_ir::{Expression, Function, Literal, Range, Statement};

        let mut func = Function::new("test");
        let lit = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));
        func.body
            .push(Statement::Emit(Range::from_index_range(0..1)));
        func.body.push(Statement::Return { value: Some(lit) });

        let dfg = DataflowGraph::build(&func);
        // Compute cost-model-derived costs (one per DFG node).
        let cost = estimate_elementwise_cost(256);
        let costs: Vec<usize> = (0..dfg.nodes().len())
            .map(|_| cost.flops as usize)
            .collect();
        let cp = dfg.critical_path_with_costs(&costs);
        assert!(cp.critical_path_length > 0);
    }
}
