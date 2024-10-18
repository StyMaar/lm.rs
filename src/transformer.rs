use crate::functional::matmul;
use crate::functional::matmul_q4;
use crate::functional::matmul_q8;
use crate::functional::rmsnorm;
use crate::functional::slice_to_u32;
use crate::functional::softmax;
use crate::functional::u8_to_f32_slice;
use crate::functional::u8_to_i8_slice;

use crate::functional::SliceOrVec;
use crate::gpu::{WgpuContext, Tensor};
use crate::quantization::*;

use memmap2::Mmap;
use rayon::prelude::*;
use std::mem::size_of;

fn init_param<'a>(gpu_context: WgpuContext<'a>, offset: &mut usize, n: u32, size_each: u32) -> Tensor<'a> {

    todo!()
    // let ptr: &[f32] =
    //     u8_to_f32_slice(&data[*offset..(*offset + ((n * size_each) as usize * size_of::<f32>()))]);

    // *offset += (n * size_each) as usize * size_of::<f32>();

    // ptr
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ModelType {
    GEMMA,
    LLAMA,
}

#[repr(C, packed)]
#[derive(Debug, Copy, Clone)]
pub struct TransformerArgs {
    dim: u32,
    hidden_dim: u32,
    n_layers: u32,
    n_heads: u32,
    head_size: u32,
    n_kv_heads: u32,
    pub vocab_size: u32,
    seq_len: u32,
    rms_norm_eps: f32,
    rope_theta: f32,
    q_type: QuantType,
    pub model_type: ModelType,
    group_size: u32,
}

pub struct TransformerWeights<'a> {
    token_embedding_table: Tensor<'a>,

    // Attention
    wq: Tensor<'a>,
    wk: Tensor<'a>,
    wv: Tensor<'a>,
    wo: Tensor<'a>,

    w_rms_att: Tensor<'a>,

    // FFN
    w1: Tensor<'a>,
    w2: Tensor<'a>,
    w3: Tensor<'a>,

    w_rms_post_att: Tensor<'a>,

    w_rms_final: Tensor<'a>,

    w_cls: Tensor<'a>,
}

pub struct TransformerState {
    x: Vec<f32>,
    xb: Vec<f32>,
    xb2: Vec<f32>,
    xb3: Vec<f32>,
    hb: Vec<f32>,
    hb2: Vec<f32>,
    q: Vec<f32>,
    logits: Vec<f32>,

    // kv cache
    key_cache: Vec<f32>,
    value_cache: Vec<f32>,
}

pub struct Transformer<'a> {
    pub args: TransformerArgs,
    weights: TransformerWeights<'a>,
    state: TransformerState,
}

impl<'a> Transformer<'a> {
    pub fn new(data: &'a Mmap) -> Transformer<'a> {
        assert_eq!(
            data[0..4],
            [0x6c, 0x6d, 0x72, 0x73],
            "Model not in lm.rs format."
        );

        let lmrs_version = slice_to_u32(&data[4..8]);

        println!("LMRS version: {}", lmrs_version);

        let (head, body, _) = unsafe { data[8..54].align_to::<TransformerArgs>() };

        assert!(head.is_empty(), "Data was not aligned");

        let cfg = &body[0];

        println!("Model type: {:?}\n", cfg.model_type);

        let head_size = cfg.head_size;

        let mut offset: usize = 256;

        let kv_dim = cfg.head_size * cfg.n_kv_heads;

        let gpu_context = WgpuContext::new(&data);

        let emb_tab = init_param(data, &mut offset, 1, cfg.vocab_size * cfg.dim);
        let rms_att = init_param(data, &mut offset, cfg.n_layers, cfg.dim);
        let wq = init_param(
            data,
            &mut offset,
            cfg.n_layers,
            cfg.dim * cfg.n_heads * head_size,
        );
        let wk = init_param(
            data,
            &mut offset,
            cfg.n_layers,
            cfg.dim * cfg.n_kv_heads * head_size,
        );
        let wv = init_param(
            data,
            &mut offset,
            cfg.n_layers,
            cfg.dim * cfg.n_kv_heads * head_size,
        );
        let wo = init_param(
            data,
            &mut offset,
            cfg.n_layers,
            cfg.dim * cfg.n_heads * head_size,
        );
        let rms_post_att = init_param(data, &mut offset, cfg.n_layers, cfg.dim);

        let w1 = init_param(data, &mut offset, cfg.n_layers, cfg.dim * cfg.hidden_dim);
        let w2 = init_param(data, &mut offset, cfg.n_layers, cfg.dim * cfg.hidden_dim);
        let w3 = init_param(data, &mut offset, cfg.n_layers, cfg.dim * cfg.hidden_dim);

        let rms_final = init_param(data, &mut offset, 1, cfg.dim);

        let weights = TransformerWeights {
            token_embedding_table: SliceOrVec::Slice(emb_tab),
            wq,
            wk,
            wv,
            wo,
            w_rms_att: rms_att,
            w1,
            w2,
            w3,
            w_rms_post_att: rms_post_att,
            w_rms_final: rms_final,
            w_cls: emb_tab,
        };

        let state = TransformerState {
            x: vec![0.0; cfg.dim as usize],
            xb: vec![0.0; cfg.dim as usize],
            xb2: vec![0.0; cfg.dim as usize],
            xb3: vec![0.0; (cfg.head_size * cfg.n_heads) as usize],
            hb: vec![0.0; cfg.hidden_dim as usize],
            hb2: vec![0.0; cfg.hidden_dim as usize],
            q: vec![0.0; (cfg.head_size * cfg.n_heads) as usize],
            key_cache: vec![0.0; (cfg.n_layers * cfg.seq_len * kv_dim) as usize],
            value_cache: vec![0.0; (cfg.n_layers * cfg.seq_len * kv_dim) as usize],
            logits: vec![0.0; cfg.vocab_size as usize],
        };

        return Transformer {
            args: *cfg,
            weights,
            state,
        };
    }

    pub fn forward(&mut self, token: u32, pos: u32) -> &mut [f32] {
        let p = self.args;
        let w = &self.weights;
        let s = &mut self.state;
        let x = &mut s.x;
        let dim = p.dim;
        let head_size = p.head_size;
        let att_dim = p.n_heads * head_size;
        let kv_dim = head_size * p.n_kv_heads;
        let kv_mul = p.n_heads / p.n_kv_heads;
        let hidden_dim = p.hidden_dim;
        let gs = p.group_size;

        x.copy_from_slice(
            &w.token_embedding_table[(token * dim) as usize..(token * dim + dim) as usize],
        );

        for l in 0..p.n_layers {
            rmsnorm(
                &mut s.xb,
                x,
                &w.w_rms_att[(l * dim) as usize..(l * dim + dim) as usize],
                dim as usize,
                p.rms_norm_eps,
                p.model_type == ModelType::GEMMA,
            );

            let loff = l * p.seq_len * kv_dim;
            let k = &mut s.key_cache
                [(loff + pos * kv_dim) as usize..(loff + pos * kv_dim + kv_dim) as usize];
            let v = &mut s.value_cache
                [(loff + pos * kv_dim) as usize..(loff + pos * kv_dim + kv_dim) as usize];

            matmul(
                &mut s.q,
                &s.xb,
                &w.wq[(l * dim * att_dim) as usize..(l * dim * att_dim + dim * att_dim) as usize],
            );
            matmul(
                k,
                &s.xb,
                &w.wk[(l * dim * kv_dim) as usize..(l * dim * kv_dim + dim * kv_dim) as usize],
            );
            matmul(
                v,
                &s.xb,
                &w.wv[(l * dim * kv_dim) as usize..(l * dim * kv_dim + dim * kv_dim) as usize],
            );

            for i in 0..p.n_heads {
                for j in 0..(head_size / 2) {
                    let head_dim: u32 = j * 2;
                    let mut freq: f32 = 1.0 / p.rope_theta.powf(head_dim as f32 / head_size as f32);

                    if p.model_type == ModelType::LLAMA {
                        let wavelen = (2.0 * std::f32::consts::PI) / freq;

                        // Should be on args
                        let factor = 32.0;
                        let low_freq_factor = 1.0;
                        let high_freq_factor = 4.0;
                        let old_context_len = 8192.0;

                        let low_freq_wavelen = old_context_len / low_freq_factor;
                        let high_freq_wavelen = old_context_len / high_freq_factor;

                        if wavelen > low_freq_wavelen {
                            freq /= factor;
                        } else if wavelen <= low_freq_wavelen && wavelen >= high_freq_wavelen {
                            let smooth_factor = (old_context_len / wavelen - low_freq_factor)
                                / (high_freq_factor - low_freq_factor);

                            freq = (1.0 - smooth_factor) * freq / factor + smooth_factor * freq
                        }
                    }

                    let val: f32 = pos as f32 * freq;
                    let fcr = val.cos();
                    let fci = val.sin();
                    let rotn: u32 = if (i * head_size) + j + head_size / 2 < kv_dim {
                        2
                    } else {
                        1
                    };

                    for v in 0..rotn {
                        let vec: &mut [f32] = if v == 0 { &mut s.q } else { k };
                        let v0: f32 = vec[((i * head_size) + j) as usize];
                        let v1: f32 = vec[(((i * head_size) + j) + (head_size / 2)) as usize];

                        vec[((i * head_size) + j) as usize] = v0 * fcr - v1 * fci;
                        vec[(((i * head_size) + j) + (head_size / 2)) as usize] =
                            v0 * fci + v1 * fcr;
                    }
                }
            }

            s.xb3
                .par_chunks_mut(head_size as usize)
                .enumerate()
                .for_each(|(h, xb)| {
                    let q = &s.q[(h as u32 * head_size) as usize
                        ..(h as u32 * head_size + head_size) as usize];

                    let att = &mut vec![0.0; p.seq_len as usize];

                    for t in 0..pos + 1 {
                        let k = &s.key_cache[(loff + t * kv_dim + (h as u32 / kv_mul) * head_size)
                            as usize
                            ..(loff + t * kv_dim + (h as u32 / kv_mul) * head_size + head_size)
                                as usize];

                        let mut score: f32 = 0.0;

                        for i in 0..head_size {
                            score += q[i as usize] * k[i as usize];
                        }

                        score /= (head_size as f32).sqrt();

                        att[t as usize] = score;
                    }

                    softmax(&mut att[..(pos + 1) as usize]);

                    xb.fill(0.0);

                    for t in 0..pos + 1 {
                        let v = &s.value_cache[(loff + t * kv_dim + (h as u32 / kv_mul) * head_size)
                            as usize
                            ..(loff + t * kv_dim + (h as u32 / kv_mul) * head_size + head_size)
                                as usize];
                        let a = att[t as usize];

                        for i in 0..head_size {
                            xb[i as usize] += a * v[i as usize];
                        }
                    }
                });

            matmul(
                &mut s.xb2,
                &s.xb3,
                &w.wo[(l * dim * att_dim) as usize..(l * dim * att_dim + dim * att_dim) as usize],
            );

            for i in 0..dim {
                x[i as usize] += s.xb2[i as usize];
            }

            rmsnorm(
                &mut s.xb,
                x,
                &w.w_rms_post_att[(l * dim) as usize..(l * dim + dim) as usize],
                dim as usize,
                p.rms_norm_eps,
                p.model_type == ModelType::GEMMA,
            );

            // GeGLU is w2(GELU(w1(x)) * w3(x))
            // w1 -> gate_proj weights
            // w2 -> down_proj weights
            // w3 -> up_proj weights
            // GELU using tanh as the approximation

            matmul(
                &mut s.hb,
                &s.xb,
                &w.w1[(l * dim * hidden_dim) as usize
                    ..(l * dim * hidden_dim + dim * hidden_dim) as usize],
            );
            matmul(
                &mut s.hb2,
                &s.xb,
                &w.w3[(l * dim * hidden_dim) as usize
                    ..(l * dim * hidden_dim + dim * hidden_dim) as usize],
            );

            for i in 0..hidden_dim {
                let mut val = s.hb[i as usize];

                val *= 1.0 / (1.0 + (-val).exp());

                val *= s.hb2[i as usize];

                s.hb[i as usize] = val;
            }

            matmul(
                &mut s.xb,
                &s.hb,
                &w.w2[(l * dim * hidden_dim) as usize
                    ..(l * dim * hidden_dim + dim * hidden_dim) as usize],
            );

            for i in 0..dim {
                x[i as usize] += s.xb[i as usize];
            }
        }

        s.xb.copy_from_slice(x);

        rmsnorm(
            x,
            &s.xb,
            w.w_rms_final,
            dim as usize,
            p.rms_norm_eps,
            p.model_type == ModelType::GEMMA,
        );

        matmul(&mut s.logits, x, w.w_cls);

        &mut s.logits
    }
}
