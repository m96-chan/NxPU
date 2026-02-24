//! Calibration pipeline for quantization.
//!
//! Provides infrastructure for loading calibration datasets, collecting
//! activation histograms, and computing optimal quantization parameters
//! using various calibration methods (MinMax, Percentile, KL-divergence).

use std::path::{Path, PathBuf};

use crate::quantize::{PerChannelQuantParams, QuantizationParams};

/// Errors that can occur during calibration.
#[derive(Debug, thiserror::Error)]
pub enum CalibrationError {
    /// Failed to read a calibration data file.
    #[error("failed to read calibration file {path}: {source}")]
    IoError {
        path: PathBuf,
        source: std::io::Error,
    },
    /// Calibration data file has invalid size (not a multiple of 4 bytes for f32).
    #[error("calibration file {path} has invalid size {size} (must be a multiple of 4 bytes)")]
    InvalidFileSize { path: PathBuf, size: u64 },
    /// No calibration samples found in the directory.
    #[error("no .bin calibration files found in {0}")]
    NoSamples(PathBuf),
    /// Empty histogram (no values collected).
    #[error("empty histogram: no values were collected")]
    EmptyHistogram,
}

/// Calibration method for determining quantization parameters.
#[derive(Clone, Debug, PartialEq)]
pub enum CalibrationMethod {
    /// Use the observed min/max values directly.
    MinMax,
    /// Clip to the given percentile (e.g. 99.99) to reduce outlier impact.
    Percentile(f32),
    /// Find the optimal clipping threshold that minimizes KL divergence
    /// between the original and quantized distributions (TensorRT-style).
    KlDivergence,
}

impl CalibrationMethod {
    /// Parse a calibration method from a CLI string.
    pub fn from_str_name(s: &str) -> Option<Self> {
        match s {
            "minmax" => Some(Self::MinMax),
            "percentile" => Some(Self::Percentile(99.99)),
            "kl-divergence" | "kl" | "entropy" => Some(Self::KlDivergence),
            _ => None,
        }
    }
}

/// A set of calibration samples loaded from binary files.
#[derive(Clone, Debug)]
pub struct CalibrationDataset {
    /// Each sample is a flat vector of f32 values.
    pub samples: Vec<Vec<f32>>,
    /// The directory these samples were loaded from.
    pub source_dir: PathBuf,
}

impl CalibrationDataset {
    /// Load calibration tensors from a directory of `.bin` files.
    ///
    /// Each `.bin` file must contain raw little-endian f32 values.
    /// Files are sorted by name for deterministic ordering.
    pub fn load_from_dir(dir: &Path) -> Result<Self, CalibrationError> {
        let mut bin_files: Vec<PathBuf> = Vec::new();

        let entries = std::fs::read_dir(dir).map_err(|e| CalibrationError::IoError {
            path: dir.to_path_buf(),
            source: e,
        })?;

        for entry in entries {
            let entry = entry.map_err(|e| CalibrationError::IoError {
                path: dir.to_path_buf(),
                source: e,
            })?;
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "bin") {
                bin_files.push(path);
            }
        }

        bin_files.sort();

        if bin_files.is_empty() {
            return Err(CalibrationError::NoSamples(dir.to_path_buf()));
        }

        let mut samples = Vec::with_capacity(bin_files.len());
        for path in &bin_files {
            let data = std::fs::read(path).map_err(|e| CalibrationError::IoError {
                path: path.clone(),
                source: e,
            })?;

            if data.len() % 4 != 0 {
                return Err(CalibrationError::InvalidFileSize {
                    path: path.clone(),
                    size: data.len() as u64,
                });
            }

            let floats: Vec<f32> = data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();

            samples.push(floats);
        }

        Ok(Self {
            samples,
            source_dir: dir.to_path_buf(),
        })
    }

    /// Create a dataset from in-memory samples (useful for testing).
    pub fn from_samples(samples: Vec<Vec<f32>>) -> Self {
        Self {
            samples,
            source_dir: PathBuf::new(),
        }
    }

    /// Returns the number of calibration samples.
    pub fn num_samples(&self) -> usize {
        self.samples.len()
    }
}

/// Histogram of activation values for a single tensor.
#[derive(Clone, Debug)]
pub struct TensorHistogram {
    /// Observed minimum value.
    pub min: f32,
    /// Observed maximum value.
    pub max: f32,
    /// Bin counts.
    pub bins: Vec<u32>,
    /// Bin edges (len = bins.len() + 1).
    pub bin_edges: Vec<f32>,
    /// Total number of values collected.
    pub total_count: u64,
}

/// Default number of histogram bins.
const DEFAULT_NUM_BINS: usize = 2048;

impl TensorHistogram {
    /// Create a new histogram from a collection of f32 values.
    ///
    /// The histogram covers the range [min, max] of the input values
    /// with `num_bins` equal-width bins.
    pub fn from_values(values: &[f32], num_bins: usize) -> Result<Self, CalibrationError> {
        if values.is_empty() {
            return Err(CalibrationError::EmptyHistogram);
        }

        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for &v in values {
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }

        // Handle constant tensors.
        if (max - min).abs() < f32::EPSILON {
            return Ok(Self {
                min,
                max,
                bins: vec![values.len() as u32],
                bin_edges: vec![min, max + f32::EPSILON],
                total_count: values.len() as u64,
            });
        }

        let mut bins = vec![0u32; num_bins];
        let mut bin_edges = Vec::with_capacity(num_bins + 1);
        let bin_width = (max - min) / num_bins as f32;

        for i in 0..=num_bins {
            bin_edges.push(min + bin_width * i as f32);
        }

        for &v in values {
            let idx = ((v - min) / bin_width) as usize;
            let idx = idx.min(num_bins - 1);
            bins[idx] += 1;
        }

        Ok(Self {
            min,
            max,
            bins,
            bin_edges,
            total_count: values.len() as u64,
        })
    }

    /// Merge another histogram's values into this one.
    ///
    /// If the ranges differ, the histogram is rebuilt with the union range.
    pub fn merge(&mut self, other: &TensorHistogram) {
        let new_min = self.min.min(other.min);
        let new_max = self.max.max(other.max);
        let num_bins = self.bins.len();

        if (new_min - self.min).abs() < f32::EPSILON
            && (new_max - self.max).abs() < f32::EPSILON
            && other.bins.len() == num_bins
        {
            // Same range, just add counts.
            for (a, b) in self.bins.iter_mut().zip(other.bins.iter()) {
                *a += b;
            }
            self.total_count += other.total_count;
            return;
        }

        // Rebuild with new range. Redistribute existing counts.
        let new_width = (new_max - new_min) / num_bins as f32;
        let mut new_bins = vec![0u32; num_bins];
        let mut new_edges = Vec::with_capacity(num_bins + 1);
        for i in 0..=num_bins {
            new_edges.push(new_min + new_width * i as f32);
        }

        // Redistribute self's bins.
        redistribute_bins(&self.bins, &self.bin_edges, &mut new_bins, &new_edges);
        // Redistribute other's bins.
        redistribute_bins(&other.bins, &other.bin_edges, &mut new_bins, &new_edges);

        self.min = new_min;
        self.max = new_max;
        self.bins = new_bins;
        self.bin_edges = new_edges;
        self.total_count += other.total_count;
    }

    /// Compute the value at a given percentile (0.0 to 100.0).
    pub fn percentile(&self, pct: f32) -> f32 {
        let target_count = (pct / 100.0 * self.total_count as f32) as u64;
        let mut cumulative = 0u64;

        for (i, &count) in self.bins.iter().enumerate() {
            cumulative += count as u64;
            if cumulative >= target_count {
                // Interpolate within this bin.
                let bin_start = self.bin_edges[i];
                let bin_end = self.bin_edges[i + 1];
                if count == 0 {
                    return bin_start;
                }
                let prev_cumulative = cumulative - count as u64;
                let frac = (target_count - prev_cumulative) as f32 / count as f32;
                return bin_start + frac * (bin_end - bin_start);
            }
        }

        self.max
    }
}

/// Redistribute counts from old bins to new bins using proportional overlap.
fn redistribute_bins(old_bins: &[u32], old_edges: &[f32], new_bins: &mut [u32], new_edges: &[f32]) {
    let new_num = new_bins.len();
    for (i, &count) in old_bins.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let old_lo = old_edges[i];
        let old_hi = old_edges[i + 1];
        let old_width = old_hi - old_lo;
        if old_width <= 0.0 {
            continue;
        }

        for j in 0..new_num {
            let new_lo = new_edges[j];
            let new_hi = new_edges[j + 1];

            // Compute overlap.
            let overlap_lo = old_lo.max(new_lo);
            let overlap_hi = old_hi.min(new_hi);
            if overlap_lo >= overlap_hi {
                continue;
            }
            let overlap_frac = (overlap_hi - overlap_lo) / old_width;
            new_bins[j] += (count as f32 * overlap_frac).round() as u32;
        }
    }
}

/// Histogram collector that aggregates activation values across calibration samples.
#[derive(Clone, Debug)]
pub struct HistogramCollector {
    /// Number of bins for histograms.
    pub num_bins: usize,
    /// Collected histograms, keyed by tensor name.
    pub histograms: Vec<(String, TensorHistogram)>,
}

impl Default for HistogramCollector {
    fn default() -> Self {
        Self {
            num_bins: DEFAULT_NUM_BINS,
            histograms: Vec::new(),
        }
    }
}

impl HistogramCollector {
    /// Create a collector with a custom number of bins.
    pub fn with_bins(num_bins: usize) -> Self {
        Self {
            num_bins,
            histograms: Vec::new(),
        }
    }

    /// Add values for a named tensor. If a histogram already exists for this
    /// tensor, the new values are merged into it.
    pub fn add_values(&mut self, name: &str, values: &[f32]) -> Result<(), CalibrationError> {
        let new_hist = TensorHistogram::from_values(values, self.num_bins)?;

        if let Some((_n, existing)) = self.histograms.iter_mut().find(|(n, _)| n == name) {
            existing.merge(&new_hist);
        } else {
            self.histograms.push((name.to_string(), new_hist));
        }

        Ok(())
    }

    /// Collect histograms from a calibration dataset.
    ///
    /// Each sample is treated as a single tensor named `tensor_name`.
    pub fn collect_from_dataset(
        &mut self,
        dataset: &CalibrationDataset,
        tensor_name: &str,
    ) -> Result<(), CalibrationError> {
        for sample in &dataset.samples {
            self.add_values(tensor_name, sample)?;
        }
        Ok(())
    }

    /// Get the histogram for a named tensor.
    pub fn get(&self, name: &str) -> Option<&TensorHistogram> {
        self.histograms
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, h)| h)
    }
}

// ---- Calibration methods ----

/// Compute quantization parameters using MinMax calibration.
///
/// For asymmetric (uint8): scale = (max - min) / 255, zero_point = round(-min / scale)
/// For symmetric (int8): scale = max(|min|, |max|) / 127, zero_point = 0
pub fn calibrate_minmax(hist: &TensorHistogram, symmetric: bool) -> QuantizationParams {
    if symmetric {
        calibrate_symmetric(hist.min, hist.max)
    } else {
        QuantizationParams::from_range(hist.min, hist.max)
    }
}

/// Compute symmetric quantization parameters from min/max.
fn calibrate_symmetric(min: f32, max: f32) -> QuantizationParams {
    let abs_max = min.abs().max(max.abs());
    if abs_max < f32::EPSILON {
        return QuantizationParams {
            scale: 1.0,
            zero_point: 0,
        };
    }
    let scale = abs_max / 127.0;
    QuantizationParams {
        scale,
        zero_point: 0,
    }
}

/// Compute quantization parameters using Percentile calibration.
///
/// Clips to the given percentile (e.g. 99.99) instead of absolute min/max,
/// reducing the impact of outliers.
pub fn calibrate_percentile(
    hist: &TensorHistogram,
    percentile: f32,
    symmetric: bool,
) -> QuantizationParams {
    let low_pct = 100.0 - percentile;
    let clipped_min = hist.percentile(low_pct);
    let clipped_max = hist.percentile(percentile);

    if symmetric {
        calibrate_symmetric(clipped_min, clipped_max)
    } else {
        QuantizationParams::from_range(clipped_min, clipped_max)
    }
}

/// Compute quantization parameters using KL-divergence (entropy) calibration.
///
/// Finds the optimal threshold that minimizes the KL divergence between the
/// original float distribution and the quantized distribution. This is the
/// standard TensorRT-style calibration approach.
pub fn calibrate_kl_divergence(hist: &TensorHistogram) -> QuantizationParams {
    let num_bins = hist.bins.len();
    if num_bins < 128 {
        // Not enough bins for KL calibration; fall back to MinMax.
        return calibrate_minmax(hist, true);
    }

    let total: u64 = hist.bins.iter().map(|&c| c as u64).sum();
    if total == 0 {
        return QuantizationParams {
            scale: 1.0,
            zero_point: 0,
        };
    }

    // Normalize histogram to probability distribution.
    let reference: Vec<f64> = hist.bins.iter().map(|&c| c as f64 / total as f64).collect();

    let target_bins: usize = 128; // int8 quantization levels
    let mut best_divergence = f64::INFINITY;
    let mut best_threshold_bin = num_bins;

    // Try different thresholds starting from 128 bins up to the full histogram.
    let start_bin = target_bins;
    for threshold_bin in start_bin..=num_bins {
        // Create a truncated distribution clipped at threshold_bin.
        let mut truncated = reference[..threshold_bin].to_vec();

        // Collect outlier mass into the last bin.
        if threshold_bin < num_bins {
            let outlier_mass: f64 = reference[threshold_bin..].iter().sum();
            if let Some(last) = truncated.last_mut() {
                *last += outlier_mass;
            }
        }

        let truncated_sum: f64 = truncated.iter().sum();
        if truncated_sum < 1e-12 {
            continue;
        }

        // Quantize: map threshold_bin bins down to target_bins.
        let bins_per_quant = threshold_bin as f64 / target_bins as f64;
        let mut quantized = vec![0.0f64; target_bins];

        for (i, &val) in truncated.iter().enumerate() {
            let q_idx = (i as f64 / bins_per_quant) as usize;
            let q_idx = q_idx.min(target_bins - 1);
            quantized[q_idx] += val;
        }

        // Expand quantized distribution back to threshold_bin bins.
        let mut expanded = vec![0.0f64; threshold_bin];
        for (q_idx, &q_val) in quantized.iter().enumerate() {
            let start = (q_idx as f64 * bins_per_quant) as usize;
            let end = (((q_idx + 1) as f64 * bins_per_quant) as usize).min(threshold_bin);
            let num_expanded = end - start;
            if num_expanded == 0 {
                continue;
            }
            // Count non-zero bins in this range.
            let nonzero_count = truncated[start..end].iter().filter(|&&v| v > 1e-12).count();
            if nonzero_count == 0 {
                continue;
            }
            let avg = q_val / nonzero_count as f64;
            for j in start..end {
                if truncated[j] > 1e-12 {
                    expanded[j] = avg;
                }
            }
        }

        // Compute KL divergence: sum p * log(p / q).
        let divergence = kl_divergence(&truncated, &expanded);
        if divergence < best_divergence {
            best_divergence = divergence;
            best_threshold_bin = threshold_bin;
        }
    }

    // Convert best threshold bin to a clipping range.
    let bin_width = (hist.max - hist.min) / num_bins as f32;
    let threshold = hist.min + bin_width * best_threshold_bin as f32;
    let abs_threshold = hist.min.abs().max(threshold.abs());

    if abs_threshold < f32::EPSILON {
        return QuantizationParams {
            scale: 1.0,
            zero_point: 0,
        };
    }

    let scale = abs_threshold / 127.0;
    QuantizationParams {
        scale,
        zero_point: 0,
    }
}

/// Compute KL divergence between distributions p and q.
/// Both must be the same length. Skips bins where p is near zero.
fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    let mut divergence = 0.0f64;
    for (i, &pi) in p.iter().enumerate() {
        if pi < 1e-12 {
            continue;
        }
        let qi = q[i];
        if qi < 1e-12 {
            continue;
        }
        divergence += pi * (pi / qi).ln();
    }
    divergence
}

/// Compute quantization parameters for a histogram using the specified method.
pub fn calibrate(
    hist: &TensorHistogram,
    method: &CalibrationMethod,
    symmetric: bool,
) -> QuantizationParams {
    match method {
        CalibrationMethod::MinMax => calibrate_minmax(hist, symmetric),
        CalibrationMethod::Percentile(pct) => calibrate_percentile(hist, *pct, symmetric),
        CalibrationMethod::KlDivergence => calibrate_kl_divergence(hist),
    }
}

// ---- Per-channel weight quantization ----

/// Compute per-channel quantization parameters for a weight tensor.
///
/// For Conv2D weights in OIHW layout, quantizes along the output channel
/// axis (axis 0). Each output channel gets its own scale/zero_point.
///
/// # Arguments
/// * `weights` - The flat weight tensor values.
/// * `shape` - The shape of the weight tensor [O, I, H, W] or [O, I].
/// * `channel_axis` - The axis along which to compute per-channel params (typically 0).
pub fn per_channel_quantize(
    weights: &[f32],
    shape: &[usize],
    channel_axis: usize,
) -> PerChannelQuantParams {
    assert!(
        channel_axis < shape.len(),
        "channel_axis {} out of bounds for shape with {} dims",
        channel_axis,
        shape.len()
    );

    let num_channels = shape[channel_axis];
    let total_elements: usize = shape.iter().product();
    assert_eq!(
        weights.len(),
        total_elements,
        "weights length {} doesn't match shape product {}",
        weights.len(),
        total_elements
    );

    // Compute stride for the channel axis.
    let inner_size: usize = shape[channel_axis + 1..].iter().product();

    let mut scales = Vec::with_capacity(num_channels);
    let mut zero_points = Vec::with_capacity(num_channels);

    for ch in 0..num_channels {
        // Collect all elements belonging to this channel.
        let mut ch_min = f32::INFINITY;
        let mut ch_max = f32::NEG_INFINITY;

        // Iterate over all elements for this channel.
        let outer_size: usize = shape[..channel_axis].iter().product();
        let outer_stride: usize = shape[channel_axis..].iter().product();

        for outer in 0..outer_size {
            let base = outer * outer_stride + ch * inner_size;
            for inner in 0..inner_size {
                let val = weights[base + inner];
                if val < ch_min {
                    ch_min = val;
                }
                if val > ch_max {
                    ch_max = val;
                }
            }
        }

        // Symmetric quantization for weights.
        let abs_max = ch_min.abs().max(ch_max.abs());
        let scale = if abs_max < f32::EPSILON {
            1.0
        } else {
            abs_max / 127.0
        };

        scales.push(scale);
        zero_points.push(0);
    }

    PerChannelQuantParams {
        scales,
        zero_points,
        channel_axis: channel_axis as u32,
    }
}

/// Results from running the calibration pipeline on a module.
#[derive(Clone, Debug, Default)]
pub struct CalibrationResult {
    /// Per-tensor quantization parameters, keyed by tensor name.
    pub tensor_params: Vec<(String, QuantizationParams)>,
    /// Per-channel weight quantization parameters, keyed by weight name.
    pub weight_params: Vec<(String, PerChannelQuantParams)>,
    /// The calibration method that was used.
    pub method: Option<CalibrationMethod>,
}

impl CalibrationResult {
    /// Look up per-tensor parameters by name.
    pub fn find_tensor(&self, name: &str) -> Option<&QuantizationParams> {
        self.tensor_params
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, p)| p)
    }

    /// Look up per-channel parameters by name.
    pub fn find_weight(&self, name: &str) -> Option<&PerChannelQuantParams> {
        self.weight_params
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, p)| p)
    }

    /// Print a summary of calibration results to the log.
    pub fn log_summary(&self) {
        log::info!(
            "Calibration results ({} tensors):",
            self.tensor_params.len()
        );
        for (name, params) in &self.tensor_params {
            log::info!(
                "  {}: scale={:.6}, zero_point={}",
                name,
                params.scale,
                params.zero_point
            );
        }
        if !self.weight_params.is_empty() {
            log::info!(
                "Per-channel weight params ({} tensors):",
                self.weight_params.len()
            );
            for (name, params) in &self.weight_params {
                log::info!(
                    "  {}: {} channels, axis={}",
                    name,
                    params.num_channels(),
                    params.channel_axis
                );
            }
        }
    }
}

/// Run the full calibration pipeline on a calibration dataset.
///
/// Collects histograms from the dataset and computes quantization parameters
/// using the specified method.
pub fn run_calibration(
    dataset: &CalibrationDataset,
    method: &CalibrationMethod,
    symmetric: bool,
) -> Result<CalibrationResult, CalibrationError> {
    let mut collector = HistogramCollector::default();

    // Each sample is treated as a single activation tensor.
    collector.collect_from_dataset(dataset, "input")?;

    let mut tensor_params = Vec::new();
    for (name, hist) in &collector.histograms {
        let params = calibrate(hist, method, symmetric);
        tensor_params.push((name.clone(), params));
    }

    Ok(CalibrationResult {
        tensor_params,
        weight_params: Vec::new(),
        method: Some(method.clone()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- TensorHistogram tests ----

    #[test]
    fn histogram_from_uniform_values() {
        let values: Vec<f32> = (0..1000).map(|i| i as f32 / 999.0).collect();
        let hist = TensorHistogram::from_values(&values, 100).unwrap();
        assert!((hist.min - 0.0).abs() < 1e-6);
        assert!((hist.max - 1.0).abs() < 1e-4);
        assert_eq!(hist.bins.len(), 100);
        assert_eq!(hist.total_count, 1000);
    }

    #[test]
    fn histogram_from_constant_values() {
        let values = vec![5.0; 100];
        let hist = TensorHistogram::from_values(&values, 100).unwrap();
        assert!((hist.min - 5.0).abs() < 1e-6);
        assert_eq!(hist.bins.len(), 1);
        assert_eq!(hist.total_count, 100);
    }

    #[test]
    fn histogram_empty_values_returns_error() {
        let result = TensorHistogram::from_values(&[], 100);
        assert!(result.is_err());
    }

    #[test]
    fn histogram_percentile() {
        let values: Vec<f32> = (0..10000).map(|i| i as f32).collect();
        let hist = TensorHistogram::from_values(&values, 1000).unwrap();

        let p50 = hist.percentile(50.0);
        assert!((p50 - 5000.0).abs() < 100.0, "p50 = {p50}");

        let p99 = hist.percentile(99.0);
        assert!((p99 - 9900.0).abs() < 100.0, "p99 = {p99}");
    }

    #[test]
    fn histogram_merge_same_range() {
        let values1: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let values2: Vec<f32> = (0..100).map(|i| i as f32).collect();

        let mut hist1 = TensorHistogram::from_values(&values1, 50).unwrap();
        let hist2 = TensorHistogram::from_values(&values2, 50).unwrap();

        hist1.merge(&hist2);
        assert_eq!(hist1.total_count, 200);
    }

    #[test]
    fn histogram_merge_different_ranges() {
        let values1: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let values2: Vec<f32> = (100..200).map(|i| i as f32).collect();

        let mut hist1 = TensorHistogram::from_values(&values1, 50).unwrap();
        let hist2 = TensorHistogram::from_values(&values2, 50).unwrap();

        hist1.merge(&hist2);
        assert!((hist1.min - 0.0).abs() < 1e-6);
        assert!((hist1.max - 199.0).abs() < 1e-1);
        assert_eq!(hist1.total_count, 200);
    }

    // ---- MinMax calibration tests ----

    #[test]
    fn minmax_asymmetric() {
        let values: Vec<f32> = vec![-1.0, 0.0, 1.0, 2.0, 3.0];
        let hist = TensorHistogram::from_values(&values, 100).unwrap();
        let params = calibrate_minmax(&hist, false);

        // scale = (3.0 - (-1.0)) / 255 = 4.0 / 255
        assert!((params.scale - 4.0 / 255.0).abs() < 1e-5);
    }

    #[test]
    fn minmax_symmetric() {
        let values: Vec<f32> = vec![-3.0, -1.0, 0.0, 1.0, 2.0];
        let hist = TensorHistogram::from_values(&values, 100).unwrap();
        let params = calibrate_minmax(&hist, true);

        // abs_max = 3.0, scale = 3.0 / 127
        assert!((params.scale - 3.0 / 127.0).abs() < 1e-5);
        assert_eq!(params.zero_point, 0);
    }

    #[test]
    fn minmax_all_positive() {
        let values: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let hist = TensorHistogram::from_values(&values, 100).unwrap();
        let params = calibrate_minmax(&hist, false);

        assert!((params.scale - 6.0 / 255.0).abs() < 1e-5);
        assert_eq!(params.zero_point, 0);
    }

    // ---- Percentile calibration tests ----

    #[test]
    fn percentile_clips_outliers() {
        // Most values in [0, 1], with one extreme outlier at 100.
        let mut values: Vec<f32> = (0..999).map(|i| i as f32 / 999.0).collect();
        values.push(100.0);

        let hist = TensorHistogram::from_values(&values, 2048).unwrap();

        let minmax_params = calibrate_minmax(&hist, false);
        let pct_params = calibrate_percentile(&hist, 99.0, false);

        // Percentile scale should be much smaller since it ignores the outlier.
        assert!(
            pct_params.scale < minmax_params.scale,
            "percentile scale {} should be less than minmax scale {}",
            pct_params.scale,
            minmax_params.scale
        );
    }

    // ---- KL-Divergence calibration tests ----

    #[test]
    fn kl_divergence_on_normal_distribution() {
        // Generate a normal-like distribution using the Box-Muller approximation.
        let mut values = Vec::with_capacity(10000);
        for i in 0..10000 {
            // Simple deterministic "normal-ish" distribution
            let x = (i as f32 - 5000.0) / 1000.0;
            let density = (-x * x / 2.0).exp();
            // Add proportional number of samples
            let count = (density * 10.0) as usize;
            for _ in 0..count.max(1) {
                values.push(x);
            }
        }

        let hist = TensorHistogram::from_values(&values, 2048).unwrap();
        let params = calibrate_kl_divergence(&hist);

        // Should produce reasonable parameters.
        assert!(params.scale > 0.0);
        assert_eq!(params.zero_point, 0); // KL is always symmetric.
    }

    #[test]
    fn kl_divergence_small_histogram_fallback() {
        let values: Vec<f32> = (0..50).map(|i| i as f32).collect();
        let hist = TensorHistogram::from_values(&values, 64).unwrap();
        let params = calibrate_kl_divergence(&hist);

        // Should fallback to minmax symmetric.
        assert!(params.scale > 0.0);
        assert_eq!(params.zero_point, 0);
    }

    // ---- Per-channel weight quantization tests ----

    #[test]
    fn per_channel_conv2d_weights() {
        // OIHW layout: 3 output channels, 1 input channel, 3x3 kernel
        let shape = [3, 1, 3, 3];
        #[rustfmt::skip]
        let weights = vec![
            // Channel 0: values in [-1, 1]
            -1.0, -0.5, 0.0, 0.5, 1.0, -0.5, 0.0, 0.5, -1.0,
            // Channel 1: values in [-2, 2]
            -2.0, -1.0, 0.0, 1.0, 2.0, -1.0, 0.0, 1.0, -2.0,
            // Channel 2: values in [-0.5, 0.5]
            -0.5, -0.25, 0.0, 0.25, 0.5, -0.25, 0.0, 0.25, -0.5,
        ];

        let params = per_channel_quantize(&weights, &shape, 0);
        assert_eq!(params.num_channels(), 3);
        assert_eq!(params.channel_axis, 0);

        // Channel 0: abs_max = 1.0, scale = 1.0 / 127
        assert!((params.scales[0] - 1.0 / 127.0).abs() < 1e-5);
        // Channel 1: abs_max = 2.0, scale = 2.0 / 127
        assert!((params.scales[1] - 2.0 / 127.0).abs() < 1e-5);
        // Channel 2: abs_max = 0.5, scale = 0.5 / 127
        assert!((params.scales[2] - 0.5 / 127.0).abs() < 1e-5);

        // All zero points should be 0 (symmetric).
        assert!(params.zero_points.iter().all(|&zp| zp == 0));
    }

    #[test]
    fn per_channel_matmul_weights() {
        // Shape: [4, 8] — 4 output features, 8 input features
        let shape = [4, 8];
        let mut weights = vec![0.0f32; 32];
        // Channel 0: max = 3.0
        weights[0] = 3.0;
        weights[7] = -3.0;
        // Channel 1: max = 1.0
        weights[8] = 1.0;
        // Channel 2: all zeros
        // Channel 3: max = 0.5
        weights[24] = 0.5;

        let params = per_channel_quantize(&weights, &shape, 0);
        assert_eq!(params.num_channels(), 4);

        assert!((params.scales[0] - 3.0 / 127.0).abs() < 1e-5);
        assert!((params.scales[1] - 1.0 / 127.0).abs() < 1e-5);
        assert!((params.scales[2] - 1.0).abs() < 1e-5); // degenerate: all zeros
        assert!((params.scales[3] - 0.5 / 127.0).abs() < 1e-5);
    }

    // ---- CalibrationDataset tests ----

    #[test]
    fn dataset_from_samples() {
        let samples = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let dataset = CalibrationDataset::from_samples(samples);
        assert_eq!(dataset.num_samples(), 2);
    }

    #[test]
    fn dataset_load_from_dir() {
        // Create temp dir with sample .bin files.
        let dir = std::env::temp_dir().join("nxpu_calibrate_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(dir.join("sample_000.bin"), &bytes).unwrap();
        std::fs::write(dir.join("sample_001.bin"), &bytes).unwrap();

        let dataset = CalibrationDataset::load_from_dir(&dir).unwrap();
        assert_eq!(dataset.num_samples(), 2);
        assert_eq!(dataset.samples[0], values);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dataset_load_empty_dir() {
        let dir = std::env::temp_dir().join("nxpu_calibrate_empty_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let result = CalibrationDataset::load_from_dir(&dir);
        assert!(result.is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dataset_load_invalid_size() {
        let dir = std::env::temp_dir().join("nxpu_calibrate_invalid_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // Write 5 bytes (not a multiple of 4).
        std::fs::write(dir.join("bad.bin"), &[0u8; 5]).unwrap();

        let result = CalibrationDataset::load_from_dir(&dir);
        assert!(result.is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- HistogramCollector tests ----

    #[test]
    fn collector_accumulates_histograms() {
        let mut collector = HistogramCollector::default();
        collector.add_values("tensor_a", &[1.0, 2.0, 3.0]).unwrap();
        collector.add_values("tensor_b", &[4.0, 5.0, 6.0]).unwrap();
        collector.add_values("tensor_a", &[7.0, 8.0, 9.0]).unwrap();

        assert!(collector.get("tensor_a").is_some());
        assert!(collector.get("tensor_b").is_some());
        assert_eq!(collector.get("tensor_a").unwrap().total_count, 6);
    }

    #[test]
    fn collector_from_dataset() {
        let dataset =
            CalibrationDataset::from_samples(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

        let mut collector = HistogramCollector::default();
        collector
            .collect_from_dataset(&dataset, "activations")
            .unwrap();

        let hist = collector.get("activations").unwrap();
        assert_eq!(hist.total_count, 6);
    }

    // ---- CalibrationMethod tests ----

    #[test]
    fn calibration_method_from_str() {
        assert_eq!(
            CalibrationMethod::from_str_name("minmax"),
            Some(CalibrationMethod::MinMax)
        );
        assert_eq!(
            CalibrationMethod::from_str_name("percentile"),
            Some(CalibrationMethod::Percentile(99.99))
        );
        assert_eq!(
            CalibrationMethod::from_str_name("kl-divergence"),
            Some(CalibrationMethod::KlDivergence)
        );
        assert_eq!(
            CalibrationMethod::from_str_name("kl"),
            Some(CalibrationMethod::KlDivergence)
        );
        assert_eq!(
            CalibrationMethod::from_str_name("entropy"),
            Some(CalibrationMethod::KlDivergence)
        );
        assert_eq!(CalibrationMethod::from_str_name("invalid"), None);
    }

    // ---- Full pipeline test ----

    #[test]
    fn run_calibration_pipeline() {
        let dataset = CalibrationDataset::from_samples(vec![
            vec![-1.0, 0.0, 1.0, 2.0],
            vec![-0.5, 0.5, 1.5, 2.5],
            vec![-2.0, -0.5, 0.0, 3.0],
        ]);

        let result = run_calibration(&dataset, &CalibrationMethod::MinMax, false).unwrap();
        assert_eq!(result.tensor_params.len(), 1);
        assert_eq!(result.tensor_params[0].0, "input");
        assert!(result.tensor_params[0].1.scale > 0.0);
    }

    // ---- CalibrationResult tests ----

    #[test]
    fn calibration_result_lookup() {
        let result = CalibrationResult {
            tensor_params: vec![
                (
                    "a".into(),
                    QuantizationParams {
                        scale: 0.1,
                        zero_point: 5,
                    },
                ),
                (
                    "b".into(),
                    QuantizationParams {
                        scale: 0.2,
                        zero_point: 10,
                    },
                ),
            ],
            weight_params: vec![(
                "w".into(),
                PerChannelQuantParams {
                    scales: vec![0.01, 0.02],
                    zero_points: vec![0, 0],
                    channel_axis: 0,
                },
            )],
            method: Some(CalibrationMethod::MinMax),
        };

        assert!(result.find_tensor("a").is_some());
        assert!((result.find_tensor("a").unwrap().scale - 0.1).abs() < 1e-6);
        assert!(result.find_tensor("c").is_none());
        assert!(result.find_weight("w").is_some());
        assert_eq!(result.find_weight("w").unwrap().num_channels(), 2);
    }

    // ---- Calibrate dispatch tests ----

    #[test]
    fn calibrate_dispatch_minmax() {
        let values: Vec<f32> = vec![-1.0, 0.0, 1.0];
        let hist = TensorHistogram::from_values(&values, 100).unwrap();
        let params = calibrate(&hist, &CalibrationMethod::MinMax, true);
        assert!((params.scale - 1.0 / 127.0).abs() < 1e-5);
    }

    #[test]
    fn calibrate_dispatch_percentile() {
        let values: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let hist = TensorHistogram::from_values(&values, 2048).unwrap();
        let params = calibrate(&hist, &CalibrationMethod::Percentile(99.0), false);
        assert!(params.scale > 0.0);
    }

    #[test]
    fn calibrate_dispatch_kl() {
        let values: Vec<f32> = (0..1000).map(|i| (i as f32 - 500.0) / 100.0).collect();
        let hist = TensorHistogram::from_values(&values, 2048).unwrap();
        let params = calibrate(&hist, &CalibrationMethod::KlDivergence, true);
        assert!(params.scale > 0.0);
        assert_eq!(params.zero_point, 0);
    }
}
