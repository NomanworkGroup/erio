//! GGUF-quantized `EmbeddingGemma` model internals.
//!
//! Implements the full forward pass: token embedding → transformer layers
//! (bidirectional attention) → mean pooling → dense projections → L2 normalization.

use std::path::Path;

use candle_core::quantized::gguf_file;
use candle_core::quantized::{QMatMul, QTensor};
use candle_core::{DType, Device, IndexOp, Module, Result as CandleResult, Tensor, D};
use candle_nn::Linear;

use crate::error::EmbeddingError;

// ---------------------------------------------------------------------------
// Utility functions (testable without model)
// ---------------------------------------------------------------------------

/// Mean-pools token embeddings over the sequence dimension, respecting an attention mask.
///
/// Masked positions (0 in `attention_mask`) are excluded from the average.
///
/// - `embeddings`: shape `[batch, seq_len, hidden]`
/// - `attention_mask`: shape `[batch, seq_len]` with 1 for real tokens, 0 for padding
pub fn mean_pool(embeddings: &Tensor, attention_mask: &Tensor) -> CandleResult<Tensor> {
    // Expand mask to [batch, seq_len, 1] for broadcasting
    let mask = attention_mask
        .unsqueeze(D::Minus1)?
        .to_dtype(embeddings.dtype())?;
    let masked = embeddings.broadcast_mul(&mask)?;
    let summed = masked.sum(1)?; // [batch, hidden]
    let counts = mask.sum(1)?; // [batch, 1]
                               // Clamp to avoid division by zero
    let counts = counts.clamp(1e-9, f64::MAX)?;
    summed.broadcast_div(&counts)
}

/// L2-normalizes a tensor along the last dimension.
///
/// Each vector is scaled to unit length: `x / ||x||_2`.
pub fn l2_normalize(tensor: &Tensor) -> CandleResult<Tensor> {
    let norm_sq = tensor.sqr()?.sum_keepdim(D::Minus1)?;
    let norm = norm_sq.sqrt()?.clamp(1e-12, f64::MAX)?;
    tensor.broadcast_div(&norm)
}

// ---------------------------------------------------------------------------
// RMS Normalization
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn from_qtensor(qtensor: &QTensor, eps: f64) -> CandleResult<Self> {
        let weight = qtensor.dequantize(&Device::Cpu)?;
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        #[allow(clippy::cast_possible_truncation)]
        let eps = self.eps as f32;
        candle_nn::ops::rms_norm(x, &self.weight, eps)
    }
}

// ---------------------------------------------------------------------------
// Rotary Position Embedding
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    fn new(head_dim: usize, max_seq_len: usize, rope_theta: f64) -> CandleResult<Self> {
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / head_dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq =
            Tensor::from_vec(inv_freq, (1, inv_freq_len), &Device::Cpu)?.to_dtype(DType::F32)?;
        #[allow(clippy::cast_possible_truncation)]
        let max_seq_u32 = max_seq_len as u32;
        let t = Tensor::arange(0u32, max_seq_u32, &Device::Cpu)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        let (_b, _h, seq_len, _d) = q.dims4()?;
        let cos = self.cos.i(..seq_len)?;
        let sin = self.sin.i(..seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

// ---------------------------------------------------------------------------
// Transformer Layer
// ---------------------------------------------------------------------------

struct LayerWeights {
    attn_q: QMatMul,
    attn_k: QMatMul,
    attn_v: QMatMul,
    attn_o: QMatMul,
    attn_q_norm: RmsNorm,
    attn_k_norm: RmsNorm,
    attn_norm: RmsNorm,
    post_attn_norm: RmsNorm,
    ffn_norm: RmsNorm,
    post_ffn_norm: RmsNorm,
    ffn_gate: QMatMul,
    ffn_up: QMatMul,
    ffn_down: QMatMul,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rotary: RotaryEmbedding,
}

impl LayerWeights {
    /// Forward pass with **bidirectional** attention (no causal mask).
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let (b_sz, seq_len, _hidden) = x.dims3()?;
        let residual = x;

        // Pre-attention norm
        let x = self.attn_norm.forward(x)?;

        // Q, K, V projections
        let q = self.attn_q.forward(&x)?;
        let k = self.attn_k.forward(&x)?;
        let v = self.attn_v.forward(&x)?;

        // Reshape to multi-head: [batch, heads, seq, head_dim]
        let q = q
            .reshape((b_sz, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Q/K normalization (Gemma3-specific)
        let q = self.attn_q_norm.forward(&q.contiguous()?)?;
        let k = self.attn_k_norm.forward(&k.contiguous()?)?;

        // Rotary position embeddings
        let (q, k) = self.rotary.apply(&q, &k)?;

        // Repeat KV heads for GQA
        let repeat = self.n_heads / self.n_kv_heads;
        let k = if repeat > 1 {
            let k = k.unsqueeze(2)?;
            k.expand((b_sz, self.n_kv_heads, repeat, seq_len, self.head_dim))?
                .reshape((b_sz, self.n_heads, seq_len, self.head_dim))?
        } else {
            k
        };
        let v = if repeat > 1 {
            let v = v.unsqueeze(2)?;
            v.expand((b_sz, self.n_kv_heads, repeat, seq_len, self.head_dim))?
                .reshape((b_sz, self.n_heads, seq_len, self.head_dim))?
        } else {
            v
        };

        // Scaled dot-product attention — NO CAUSAL MASK (bidirectional)
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_out = attn_weights.matmul(&v)?;

        // Reshape back: [batch, seq, hidden]
        let q_dim = self.n_heads * self.head_dim;
        let attn_out = attn_out.transpose(1, 2)?.reshape((b_sz, seq_len, q_dim))?;
        let attn_out = self.attn_o.forward(&attn_out)?;

        // Post-attention norm + residual
        let x = (residual + self.post_attn_norm.forward(&attn_out)?)?;
        let residual = &x;

        // FFN: pre-norm → gate + up → SiLU → down → post-norm + residual
        let ff_in = self.ffn_norm.forward(&x)?;
        let gate = self.ffn_gate.forward(&ff_in)?;
        let up = self.ffn_up.forward(&ff_in)?;
        let ff_out = (candle_nn::Activation::Gelu.forward(&gate)? * up)?;
        let ff_out = self.ffn_down.forward(&ff_out)?;
        let out = (residual + self.post_ffn_norm.forward(&ff_out)?)?;

        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Full Embedding Model
// ---------------------------------------------------------------------------

/// The full `EmbeddingGemma` model loaded from GGUF + Dense layer safetensors.
pub struct EmbeddingGemmaModel {
    token_embd: Tensor,
    layers: Vec<LayerWeights>,
    output_norm: RmsNorm,
    dense1: Linear,
    dense2: Linear,
    tokenizer: tokenizers::Tokenizer,
}

impl EmbeddingGemmaModel {
    /// Loads the model from a GGUF file and supplementary safetensors.
    pub fn load(
        gguf_path: &Path,
        dense1_path: &Path,
        dense2_path: &Path,
        tokenizer_path: &Path,
    ) -> Result<Self, EmbeddingError> {
        let device = Device::Cpu;

        // Load tokenizer
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| EmbeddingError::ModelLoad(format!("tokenizer load failed: {e}")))?;

        // Load GGUF
        let mut file = std::fs::File::open(gguf_path)
            .map_err(|e| EmbeddingError::ModelLoad(format!("failed to open GGUF: {e}")))?;
        let ct = gguf_file::Content::read(&mut file)
            .map_err(|e| EmbeddingError::ModelLoad(format!("failed to read GGUF: {e}")))?;

        // Read model config from GGUF metadata
        let arch = match ct.metadata.get("general.architecture") {
            Some(gguf_file::Value::String(s)) => s.clone(),
            _ => "gemma3".to_string(),
        };
        let get_meta_u32 = |key: &str| -> Result<u32, EmbeddingError> {
            let full_key = format!("{arch}.{key}");
            match ct.metadata.get(&full_key) {
                Some(gguf_file::Value::U32(v)) => Ok(*v),
                #[allow(clippy::cast_possible_truncation)]
                Some(gguf_file::Value::U64(v)) => Ok(*v as u32),
                _ => Err(EmbeddingError::ModelLoad(format!(
                    "missing or invalid GGUF metadata: {arch}.{key}"
                ))),
            }
        };
        let get_meta_f32 = |key: &str| -> Result<f32, EmbeddingError> {
            let full_key = format!("{arch}.{key}");
            match ct.metadata.get(&full_key) {
                Some(gguf_file::Value::F32(v)) => Ok(*v),
                _ => Err(EmbeddingError::ModelLoad(format!(
                    "missing or invalid GGUF metadata: {arch}.{key}"
                ))),
            }
        };

        #[allow(clippy::cast_possible_truncation)]
        let n_layers = get_meta_u32("block_count")? as usize;
        #[allow(clippy::cast_possible_truncation)]
        let n_heads = get_meta_u32("attention.head_count")? as usize;
        #[allow(clippy::cast_possible_truncation)]
        let n_kv_heads = get_meta_u32("attention.head_count_kv")? as usize;
        #[allow(clippy::cast_possible_truncation)]
        let head_dim = get_meta_u32("attention.key_length")? as usize;
        let rms_eps =
            f64::from(get_meta_f32("attention.layer_norm_rms_epsilon").unwrap_or(1e-6_f32));
        let rope_theta = f64::from(get_meta_f32("rope.freq_base").unwrap_or(10000.0_f32));
        let max_seq_len = 2048_usize;

        // Token embeddings (dequantize for lookup)
        let token_embd = ct
            .tensor(&mut file, "token_embd.weight", &device)
            .map_err(|e| EmbeddingError::ModelLoad(format!("token_embd: {e}")))?
            .dequantize(&device)
            .map_err(|e| EmbeddingError::ModelLoad(format!("token_embd dequant: {e}")))?;

        // Build rotary embedding (shared across layers with same rope_theta)
        let rotary = RotaryEmbedding::new(head_dim, max_seq_len, rope_theta)
            .map_err(|e| EmbeddingError::ModelLoad(format!("rotary: {e}")))?;

        // Build transformer layers
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let prefix = format!("blk.{i}");
            let layer = Self::load_layer(
                &ct, &mut file, &device, &prefix, rms_eps, n_heads, n_kv_heads, head_dim, &rotary,
            )?;
            layers.push(layer);
        }

        // Output norm
        let output_norm_tensor = ct
            .tensor(&mut file, "output_norm.weight", &device)
            .map_err(|e| EmbeddingError::ModelLoad(format!("output_norm: {e}")))?;
        let output_norm = RmsNorm::from_qtensor(&output_norm_tensor, rms_eps)
            .map_err(|e| EmbeddingError::ModelLoad(format!("output_norm rmsnorm: {e}")))?;

        // Dense layers from safetensors
        let dense1 = Self::load_dense(dense1_path, &device)?;
        let dense2 = Self::load_dense(dense2_path, &device)?;

        Ok(Self {
            token_embd,
            layers,
            output_norm,
            dense1,
            dense2,
            tokenizer,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn load_layer(
        ct: &gguf_file::Content,
        file: &mut std::fs::File,
        device: &Device,
        prefix: &str,
        rms_eps: f64,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        rotary: &RotaryEmbedding,
    ) -> Result<LayerWeights, EmbeddingError> {
        macro_rules! qt {
            ($name:expr) => {{
                let full = format!("{}.{}", prefix, $name);
                ct.tensor(file, &full, device)
                    .map_err(|e| EmbeddingError::ModelLoad(format!("{full}: {e}")))?
            }};
        }
        macro_rules! qm {
            ($name:expr) => {{
                let t = qt!($name);
                let full = format!("{}.{}", prefix, $name);
                QMatMul::from_qtensor(t)
                    .map_err(|e| EmbeddingError::ModelLoad(format!("{full} qmatmul: {e}")))?
            }};
        }
        macro_rules! rn {
            ($name:expr) => {{
                let t = qt!($name);
                let full = format!("{}.{}", prefix, $name);
                RmsNorm::from_qtensor(&t, rms_eps)
                    .map_err(|e| EmbeddingError::ModelLoad(format!("{full} rmsnorm: {e}")))?
            }};
        }

        Ok(LayerWeights {
            attn_q: qm!("attn_q.weight"),
            attn_k: qm!("attn_k.weight"),
            attn_v: qm!("attn_v.weight"),
            attn_o: qm!("attn_output.weight"),
            attn_q_norm: rn!("attn_q_norm.weight"),
            attn_k_norm: rn!("attn_k_norm.weight"),
            attn_norm: rn!("attn_norm.weight"),
            post_attn_norm: rn!("post_attention_norm.weight"),
            ffn_norm: rn!("ffn_norm.weight"),
            post_ffn_norm: rn!("post_ffw_norm.weight"),
            ffn_gate: qm!("ffn_gate.weight"),
            ffn_up: qm!("ffn_up.weight"),
            ffn_down: qm!("ffn_down.weight"),
            n_heads,
            n_kv_heads,
            head_dim,
            rotary: rotary.clone(),
        })
    }

    fn load_dense(path: &Path, device: &Device) -> Result<Linear, EmbeddingError> {
        let tensors = candle_core::safetensors::load(path, device).map_err(|e| {
            EmbeddingError::ModelLoad(format!("dense safetensors load {}: {e}", path.display()))
        })?;
        let weight = tensors
            .get("linear.weight")
            .or_else(|| tensors.get("weight"))
            .or_else(|| tensors.get("0.weight"))
            .ok_or_else(|| {
                let keys: Vec<_> = tensors.keys().collect();
                EmbeddingError::ModelLoad(format!(
                    "no weight tensor found in {}, available keys: {keys:?}",
                    path.display()
                ))
            })?
            .clone();
        let bias = tensors
            .get("linear.bias")
            .or_else(|| tensors.get("bias"))
            .or_else(|| tensors.get("0.bias"))
            .cloned();
        Ok(Linear::new(weight, bias))
    }

    /// Tokenizes text and runs the full embedding pipeline.
    ///
    /// Returns a 768-dimensional L2-normalized embedding vector.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| EmbeddingError::Tokenization(e.to_string()))?;
        let token_ids = encoding.get_ids();
        let attention_mask_data: Vec<f32> = encoding
            .get_attention_mask()
            .iter()
            .map(|&v| if v == 0 { 0.0_f32 } else { 1.0_f32 })
            .collect();

        let device = Device::Cpu;

        // [1, seq_len]
        let input_ids = Tensor::new(token_ids, &device)
            .map_err(|e| EmbeddingError::Inference(format!("input tensor: {e}")))?
            .unsqueeze(0)
            .map_err(|e| EmbeddingError::Inference(format!("unsqueeze: {e}")))?;
        let attention_mask = Tensor::new(&attention_mask_data[..], &device)
            .map_err(|e| EmbeddingError::Inference(format!("mask tensor: {e}")))?
            .unsqueeze(0)
            .map_err(|e| EmbeddingError::Inference(format!("mask unsqueeze: {e}")))?;

        // Token embeddings: [1, seq_len, hidden]
        let mut hidden = self
            .token_embd
            .index_select(
                &input_ids
                    .squeeze(0)
                    .map_err(|e| EmbeddingError::Inference(format!("squeeze: {e}")))?,
                0,
            )
            .map_err(|e| EmbeddingError::Inference(format!("embedding lookup: {e}")))?
            .unsqueeze(0)
            .map_err(|e| EmbeddingError::Inference(format!("embd unsqueeze: {e}")))?;

        // Gemma scales embeddings by sqrt(hidden_dim)
        let hidden_dim = hidden
            .dim(D::Minus1)
            .map_err(|e| EmbeddingError::Inference(format!("hidden dim: {e}")))?;
        #[allow(clippy::cast_precision_loss)]
        let scale = (hidden_dim as f64).sqrt();
        hidden = hidden
            .affine(scale, 0.0)
            .map_err(|e| EmbeddingError::Inference(format!("embd scale: {e}")))?;

        // Transformer layers (bidirectional — no causal mask)
        for layer in &self.layers {
            hidden = layer
                .forward(&hidden)
                .map_err(|e| EmbeddingError::Inference(format!("layer forward: {e}")))?;
        }

        // Final norm
        hidden = self
            .output_norm
            .forward(&hidden)
            .map_err(|e| EmbeddingError::Inference(format!("output norm: {e}")))?;

        // Mean pool
        let pooled = mean_pool(&hidden, &attention_mask)
            .map_err(|e| EmbeddingError::Inference(format!("mean pool: {e}")))?;

        // Dense projections
        let projected = self
            .dense1
            .forward(&pooled)
            .map_err(|e| EmbeddingError::Inference(format!("dense1: {e}")))?;
        let projected = self
            .dense2
            .forward(&projected)
            .map_err(|e| EmbeddingError::Inference(format!("dense2: {e}")))?;

        // L2 normalize
        let normalized = l2_normalize(&projected)
            .map_err(|e| EmbeddingError::Inference(format!("l2 normalize: {e}")))?;

        // [1, 768] → Vec<f32>
        let result: Vec<f32> = normalized
            .squeeze(0)
            .map_err(|e| EmbeddingError::Inference(format!("result squeeze: {e}")))?
            .to_vec1()
            .map_err(|e| EmbeddingError::Inference(format!("to_vec1: {e}")))?;

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_pool_averages_over_sequence() {
        let device = Device::Cpu;
        // [1, 3, 2] — 3 tokens, 2 hidden dims
        let embeddings =
            Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], (1, 3, 2), &device).unwrap();
        // All tokens are real
        let mask = Tensor::from_vec(vec![1.0_f32, 1.0, 1.0], (1, 3), &device).unwrap();

        let pooled = mean_pool(&embeddings, &mask).unwrap();
        let result: Vec<f32> = pooled.squeeze(0).unwrap().to_vec1().unwrap();

        // (1+3+5)/3 = 3.0, (2+4+6)/3 = 4.0
        assert!((result[0] - 3.0).abs() < 1e-5);
        assert!((result[1] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn mean_pool_respects_attention_mask() {
        let device = Device::Cpu;
        // [1, 3, 2] — 3 tokens, 2 hidden dims
        let embeddings =
            Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 99.0, 99.0], (1, 3, 2), &device).unwrap();
        // Last token is padding
        let mask = Tensor::from_vec(vec![1.0_f32, 1.0, 0.0], (1, 3), &device).unwrap();

        let pooled = mean_pool(&embeddings, &mask).unwrap();
        let result: Vec<f32> = pooled.squeeze(0).unwrap().to_vec1().unwrap();

        // Only first two tokens: (1+3)/2 = 2.0, (2+4)/2 = 3.0
        assert!((result[0] - 2.0).abs() < 1e-5);
        assert!((result[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn l2_normalize_produces_unit_vector() {
        let device = Device::Cpu;
        let tensor = Tensor::from_vec(vec![3.0_f32, 4.0], (1, 2), &device).unwrap();

        let normalized = l2_normalize(&tensor).unwrap();
        let result: Vec<f32> = normalized.squeeze(0).unwrap().to_vec1().unwrap();

        // ||[3,4]|| = 5, so normalized = [0.6, 0.8]
        assert!((result[0] - 0.6).abs() < 1e-5);
        assert!((result[1] - 0.8).abs() < 1e-5);

        // Verify unit length
        let magnitude: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((magnitude - 1.0).abs() < 1e-5);
    }

    #[test]
    fn l2_normalize_handles_batch() {
        let device = Device::Cpu;
        let tensor = Tensor::from_vec(vec![3.0_f32, 4.0, 0.0, 5.0], (2, 2), &device).unwrap();

        let normalized = l2_normalize(&tensor).unwrap();
        let result: Vec<Vec<f32>> = normalized.to_vec2().unwrap();

        // First: [0.6, 0.8]
        assert!((result[0][0] - 0.6).abs() < 1e-5);
        assert!((result[0][1] - 0.8).abs() < 1e-5);
        // Second: [0.0, 1.0]
        assert!(result[1][0].abs() < 1e-5);
        assert!((result[1][1] - 1.0).abs() < 1e-5);
    }
}
