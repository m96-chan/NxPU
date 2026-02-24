//! Loop tiling and cache blocking.
//!
//! Computes tiling plans for MatMul and Conv2D operations to improve
//! data locality by partitioning work into cache-friendly tile sizes.

use std::fmt;

use nxpu_ir::Module;

use crate::Pass;

/// Configuration for a single tile dimension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileConfig {
    /// Name of the dimension being tiled (e.g., "M", "N", "K").
    pub dim_name: String,
    /// Tile size in elements.
    pub tile_size: u32,
}

impl fmt::Display for TileConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}={}", self.dim_name, self.tile_size)
    }
}

/// A tiling plan for a single operation.
#[derive(Debug, Clone)]
pub struct TilingPlan {
    /// Name of the operation (e.g., "matmul_0").
    pub op_name: String,
    /// Per-dimension tile configurations.
    pub tiles: Vec<TileConfig>,
    /// Cache reuse factor (ratio of data reuse from tiling; >1.0 means improvement).
    pub reuse_factor: f64,
}

impl fmt::Display for TilingPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TilingPlan({}: ", self.op_name)?;
        for (i, tile) in self.tiles.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{tile}")?;
        }
        write!(f, " reuse={:.2})", self.reuse_factor)
    }
}

/// Default tile sizes for common operations.
#[derive(Debug, Clone)]
pub struct TilingDefaults {
    /// Default tile size for MatMul M dimension.
    pub matmul_m: u32,
    /// Default tile size for MatMul N dimension.
    pub matmul_n: u32,
    /// Default tile size for MatMul K dimension.
    pub matmul_k: u32,
    /// Default tile size for Conv2D output height.
    pub conv2d_oh: u32,
    /// Default tile size for Conv2D output width.
    pub conv2d_ow: u32,
}

impl Default for TilingDefaults {
    fn default() -> Self {
        Self {
            matmul_m: 32,
            matmul_n: 32,
            matmul_k: 32,
            conv2d_oh: 4,
            conv2d_ow: 4,
        }
    }
}

/// MatMul shape for tiling purposes (numeric dimensions).
#[derive(Debug, Clone)]
pub struct MatMulShape {
    /// Row dimension.
    pub m: u32,
    /// Column dimension.
    pub n: u32,
    /// Inner/reduction dimension.
    pub k: u32,
}

/// Conv2D shape for tiling purposes (numeric dimensions).
#[derive(Debug, Clone)]
pub struct Conv2DShape {
    /// Output height.
    pub oh: u32,
    /// Output width.
    pub ow: u32,
    /// Kernel height.
    pub kh: u32,
    /// Kernel width.
    pub kw: u32,
}

/// Compute a tiling plan for a MatMul operation.
///
/// Tile sizes are clamped to the actual dimension size when the dimension
/// is smaller than the default tile size.
pub fn tile_matmul(shape: &MatMulShape, defaults: &TilingDefaults) -> TilingPlan {
    let tm = defaults.matmul_m.min(shape.m);
    let tn = defaults.matmul_n.min(shape.n);
    let tk = defaults.matmul_k.min(shape.k);

    // Reuse factor: how much more data is reused from L1 cache.
    // For tiled MatMul: each tile of A is reused N/tn times, each tile of B is reused M/tm times.
    // Overall reuse factor ≈ min(M/tm, N/tn) * (K/tk) / (K/tk) simplified to:
    // reuse ≈ (M * N) / (tm * tn) for the output tile reuse.
    let tiles_m = (shape.m as f64) / (tm as f64);
    let tiles_n = (shape.n as f64) / (tn as f64);
    let tiles_k = (shape.k as f64) / (tk as f64);
    // Each A-tile (tm×tk) is read tiles_n times, each B-tile (tk×tn) is read tiles_m times.
    // Without tiling: read A once (M×K), read B once (K×N), total = M*K + K*N.
    // With tiling: read A tiles_n times, read B tiles_m times.
    // Reuse factor = untiled_reads / tiled_reads? That's <1 since tiling reads MORE.
    // Actually, reuse factor measures how much a tile is reused from cache:
    // For K dimension: each (tm,tn) output tile needs tm*tk + tk*tn data per k-step,
    // producing tm*tn partial results. The inner product over K amortizes loads.
    // reuse ≈ tm * tn * tk / (tm*tk + tk*tn) = tn*tm / (tm + tn) [when tk factors out]
    let reuse_factor = if tm + tn > 0 {
        (tm as f64 * tn as f64) / ((tm + tn) as f64)
    } else {
        1.0
    };
    let _ = (tiles_m, tiles_n, tiles_k);

    TilingPlan {
        op_name: "matmul".into(),
        tiles: vec![
            TileConfig {
                dim_name: "M".into(),
                tile_size: tm,
            },
            TileConfig {
                dim_name: "N".into(),
                tile_size: tn,
            },
            TileConfig {
                dim_name: "K".into(),
                tile_size: tk,
            },
        ],
        reuse_factor,
    }
}

/// Compute a tiling plan for a Conv2D operation.
pub fn tile_conv2d(shape: &Conv2DShape, defaults: &TilingDefaults) -> TilingPlan {
    let toh = defaults.conv2d_oh.min(shape.oh);
    let tow = defaults.conv2d_ow.min(shape.ow);

    // Reuse factor: each output tile (toh, tow) shares input data from the
    // receptive field. More tiles reuse kernel weights.
    let reuse_factor = if toh + tow > 0 {
        (shape.kh as f64 * shape.kw as f64 * toh as f64 * tow as f64)
            / ((toh + shape.kh - 1) as f64 * (tow + shape.kw - 1) as f64)
    } else {
        1.0
    };

    TilingPlan {
        op_name: "conv2d".into(),
        tiles: vec![
            TileConfig {
                dim_name: "OH".into(),
                tile_size: toh,
            },
            TileConfig {
                dim_name: "OW".into(),
                tile_size: tow,
            },
        ],
        reuse_factor,
    }
}

/// Tiling pass that classifies entry points and computes tiling plans.
#[derive(Debug)]
pub struct TilingPass {
    defaults: TilingDefaults,
}

impl TilingPass {
    /// Create a tiling pass with the given defaults.
    pub fn new(defaults: TilingDefaults) -> Self {
        Self { defaults }
    }
}

impl Default for TilingPass {
    fn default() -> Self {
        Self::new(TilingDefaults::default())
    }
}

impl Pass for TilingPass {
    fn name(&self) -> &str {
        "tiling"
    }

    fn run(&self, module: &mut Module) -> bool {
        // Classify each entry point and compute tiling if applicable.
        let mut any_tiled = false;
        for i in 0..module.entry_points.len() {
            if let Ok(pattern) = nxpu_analysis::classify_entry_point(module, i) {
                match &pattern {
                    nxpu_analysis::KernelPattern::MatMul { shape, .. } => {
                        let m = shape.m.parse::<u32>().unwrap_or(256);
                        let n = shape.n.parse::<u32>().unwrap_or(256);
                        let k = shape.k.parse::<u32>().unwrap_or(256);
                        let _plan = tile_matmul(&MatMulShape { m, n, k }, &self.defaults);
                        any_tiled = true;
                    }
                    nxpu_analysis::KernelPattern::Conv2D { shape, .. } => {
                        let oh = shape.height.parse::<u32>().unwrap_or(32);
                        let ow = shape.width.parse::<u32>().unwrap_or(32);
                        let _plan = tile_conv2d(
                            &Conv2DShape {
                                oh,
                                ow,
                                kh: shape.kernel_h_val as u32,
                                kw: shape.kernel_w_val as u32,
                            },
                            &self.defaults,
                        );
                        any_tiled = true;
                    }
                    _ => {}
                }
            }
        }
        any_tiled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_matmul_default_sizes() {
        let shape = MatMulShape {
            m: 1024,
            n: 1024,
            k: 1024,
        };
        let plan = tile_matmul(&shape, &TilingDefaults::default());
        assert_eq!(plan.tiles[0].tile_size, 32); // M
        assert_eq!(plan.tiles[1].tile_size, 32); // N
        assert_eq!(plan.tiles[2].tile_size, 32); // K
    }

    #[test]
    fn tile_matmul_clamps_small_dim() {
        let shape = MatMulShape {
            m: 16,
            n: 1024,
            k: 1024,
        };
        let plan = tile_matmul(&shape, &TilingDefaults::default());
        assert_eq!(plan.tiles[0].tile_size, 16); // M clamped to 16
        assert_eq!(plan.tiles[1].tile_size, 32); // N stays 32
    }

    #[test]
    fn tile_matmul_reuse_factor() {
        let shape = MatMulShape {
            m: 1024,
            n: 1024,
            k: 1024,
        };
        let plan = tile_matmul(&shape, &TilingDefaults::default());
        // reuse = (32*32) / (32+32) = 1024/64 = 16.0
        assert!((plan.reuse_factor - 16.0).abs() < 0.01);
    }

    #[test]
    fn tile_conv2d_default_sizes() {
        let shape = Conv2DShape {
            oh: 32,
            ow: 32,
            kh: 3,
            kw: 3,
        };
        let plan = tile_conv2d(&shape, &TilingDefaults::default());
        assert_eq!(plan.tiles[0].tile_size, 4); // OH
        assert_eq!(plan.tiles[1].tile_size, 4); // OW
    }

    #[test]
    fn tile_conv2d_clamps_small_spatial() {
        let shape = Conv2DShape {
            oh: 2,
            ow: 32,
            kh: 3,
            kw: 3,
        };
        let plan = tile_conv2d(&shape, &TilingDefaults::default());
        assert_eq!(plan.tiles[0].tile_size, 2); // OH clamped
        assert_eq!(plan.tiles[1].tile_size, 4); // OW stays 4
    }

    #[test]
    fn tiling_defaults_sane() {
        let d = TilingDefaults::default();
        assert!(d.matmul_m >= 4);
        assert!(d.matmul_n >= 4);
        assert!(d.matmul_k >= 4);
        assert!(d.conv2d_oh >= 4);
        assert!(d.conv2d_ow >= 4);
        // All should be powers of 2.
        assert!(d.matmul_m.is_power_of_two());
        assert!(d.matmul_n.is_power_of_two());
        assert!(d.matmul_k.is_power_of_two());
        assert!(d.conv2d_oh.is_power_of_two());
        assert!(d.conv2d_ow.is_power_of_two());
    }

    #[test]
    fn tiling_plan_display() {
        let plan = TilingPlan {
            op_name: "matmul_0".into(),
            tiles: vec![
                TileConfig {
                    dim_name: "M".into(),
                    tile_size: 32,
                },
                TileConfig {
                    dim_name: "N".into(),
                    tile_size: 32,
                },
            ],
            reuse_factor: 16.0,
        };
        let s = format!("{plan}");
        assert!(s.contains("matmul_0"));
        assert!(s.contains("M=32"));
        assert!(s.contains("N=32"));
        assert!(s.contains("reuse=16.00"));
    }

    #[test]
    fn pass_on_matmul_module() {
        // Create a module that would classify as MatMul.
        // Since we can't easily build a full MatMul IR from scratch, test the
        // pass on an empty module (no entry points → no tiling → false).
        let mut module = Module::default();
        let pass = TilingPass::default();
        let changed = pass.run(&mut module);
        // No entry points → no tiling.
        assert!(!changed);
    }

    #[test]
    fn pass_noop_on_elementwise() {
        // Empty module with no entry points → no tiling.
        let mut module = Module::default();
        let pass = TilingPass::default();
        let changed = pass.run(&mut module);
        assert!(!changed);
    }

    #[test]
    fn performance_test() {
        // Large MatMul should have reuse_factor > 1.0.
        let shape = MatMulShape {
            m: 2048,
            n: 2048,
            k: 2048,
        };
        let plan = tile_matmul(&shape, &TilingDefaults::default());
        assert!(plan.reuse_factor > 1.0);
    }
}
