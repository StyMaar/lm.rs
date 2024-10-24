use crate::functional::rmsnorm;
use crate::functional::slice_to_u32;
use crate::functional::softmax;
use crate::functional::u8_to_f32_slice;
use crate::functional::u8_to_i8_slice;

use crate::functional::SliceOrVec;
use crate::gpu::{matmul, Cache, Vector, Weights, WgpuContext, WgpuContextBuilder};
use crate::quantization::*;

use memmap2::Mmap;
use rayon::prelude::*;
use std::mem::size_of;

fn init_param<'a>(
    gpu_context: &'a WgpuContext<'a>,
    offset: &mut usize,
    n: u32,
    size_each: u32,
) -> Weights<'a> {
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
    token_embedding_table: Weights<'a>,

    // Attention
    wq: Weights<'a>,
    wk: Weights<'a>,
    wv: Weights<'a>,
    wo: Weights<'a>,

    w_rms_att: Weights<'a>,

    // FFN
    w1: Weights<'a>,
    w2: Weights<'a>,
    w3: Weights<'a>,

    w_rms_post_att: Weights<'a>,

    w_rms_final: Weights<'a>,

    w_cls: Weights<'a>,
}

pub struct TransformerState<'a> {
    x: Vector<'a>,
    xb: Vector<'a>,
    xb2: Vector<'a>,
    xb3: Vector<'a>,
    hb: Vector<'a>,
    hb2: Vector<'a>,
    q: Vector<'a>,
    logits: Vector<'a>,

    // kv cache
    key_cache: Cache<'a>,
    value_cache: Cache<'a>,
}

pub struct Transformer<'a> {
    pub args: TransformerArgs,
    weights: TransformerWeights<'a>,
    state: TransformerState<'a>,
}

impl<'a> Transformer<'a> {
    // pour une question de lifetime du WgpuContext qu'on créé à l'intérieur de la fonction et qu'on ne peut pas `move` on passe une option `None`et on `mem::replace` dedans. Comme ça le WgpuContext ne bouge pas et a la bonne place
    pub fn new(data: &'a [u8], gpu_context: &'a mut Option<WgpuContext<'a>>) -> Transformer<'a> {
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

        let mut gpu_context_builder = WgpuContextBuilder::new();

        let x = gpu_context_builder.make_state(cfg.dim as usize);
        let xb = gpu_context_builder.make_state(cfg.dim as usize);
        let xb2 = gpu_context_builder.make_state(cfg.dim as usize);
        let xb3 = gpu_context_builder.make_state((cfg.head_size * cfg.n_heads) as usize);
        let hb = gpu_context_builder.make_state(cfg.hidden_dim as usize);
        let hb2 = gpu_context_builder.make_state(cfg.hidden_dim as usize);
        let q = gpu_context_builder.make_state((cfg.head_size * cfg.n_heads) as usize);
        let key_cache =
            gpu_context_builder.make_state((cfg.n_layers * cfg.seq_len * kv_dim) as usize);
        let value_cache =
            gpu_context_builder.make_state((cfg.n_layers * cfg.seq_len * kv_dim) as usize);
        let logits = gpu_context_builder.make_state(cfg.vocab_size as usize);

        gpu_context.replace(pollster::block_on(gpu_context_builder.finalize(data)));

        let gpu_context = gpu_context
            .as_ref()
            .expect("GPU context has been initialized");

        //        let gpu_context = gpu_context.as_mut().expect("The GPU context has been initialized");

        let emb_tab = init_param(gpu_context, &mut offset, 1, cfg.vocab_size * cfg.dim);
        let rms_att = init_param(gpu_context, &mut offset, cfg.n_layers, cfg.dim);
        let wq = init_param(
            gpu_context,
            &mut offset,
            cfg.n_layers,
            cfg.dim * cfg.n_heads * head_size,
        );
        let wk = init_param(
            gpu_context,
            &mut offset,
            cfg.n_layers,
            cfg.dim * cfg.n_kv_heads * head_size,
        );
        let wv = init_param(
            gpu_context,
            &mut offset,
            cfg.n_layers,
            cfg.dim * cfg.n_kv_heads * head_size,
        );
        let wo = init_param(
            gpu_context,
            &mut offset,
            cfg.n_layers,
            cfg.dim * cfg.n_heads * head_size,
        );
        let rms_post_att = init_param(gpu_context, &mut offset, cfg.n_layers, cfg.dim);

        let w1 = init_param(
            gpu_context,
            &mut offset,
            cfg.n_layers,
            cfg.dim * cfg.hidden_dim,
        );
        let w2 = init_param(
            gpu_context,
            &mut offset,
            cfg.n_layers,
            cfg.dim * cfg.hidden_dim,
        );
        let w3 = init_param(
            gpu_context,
            &mut offset,
            cfg.n_layers,
            cfg.dim * cfg.hidden_dim,
        );

        let rms_final = init_param(gpu_context, &mut offset, 1, cfg.dim);

        let weights = TransformerWeights {
            token_embedding_table: emb_tab.clone(),
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
            x: Vector::new(gpu_context, x),
            xb: Vector::new(gpu_context, xb),
            xb2: Vector::new(gpu_context, xb2),
            xb3: Vector::new(gpu_context, xb3),
            hb: Vector::new(gpu_context, hb),
            hb2: Vector::new(gpu_context, hb2),
            q: Vector::new(gpu_context, q),
            key_cache: Cache::new(gpu_context, key_cache),
            value_cache: Cache::new(gpu_context, value_cache),
            logits: Vector::new(gpu_context, logits),
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
        let x = s.x.data_mut();
        let dim = p.dim;
        let head_size = p.head_size;
        let att_dim = p.n_heads * head_size;
        let kv_dim = head_size * p.n_kv_heads;
        let kv_mul = p.n_heads / p.n_kv_heads;
        let hidden_dim = p.hidden_dim;
        let gs = p.group_size;

        //
        x.copy_from_slice(
            &w.token_embedding_table[(token * dim) as usize..(token * dim + dim) as usize].data(),
        );

        for l in 0..p.n_layers {
            rmsnorm(
                &mut s.xb.data_mut(),
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
                        let vec: &mut [f32] = if v == 0 { s.q.data_mut() } else { k.data_mut() };
                        let v0: f32 = vec[((i * head_size) + j) as usize];
                        let v1: f32 = vec[(((i * head_size) + j) + (head_size / 2)) as usize];

                        vec[((i * head_size) + j) as usize] = v0 * fcr - v1 * fci;
                        vec[(((i * head_size) + j) + (head_size / 2)) as usize] =
                            v0 * fci + v1 * fcr;
                    }
                }
            }

            s.xb3
                .data_mut()
                .par_chunks_mut(head_size as usize)
                .enumerate()
                .for_each(|(h, xb)| {
                    let q = &s.q.data()[(h as u32 * head_size) as usize
                        ..(h as u32 * head_size + head_size) as usize];

                    let att = &mut vec![0.0; p.seq_len as usize];

                    for t in 0..pos + 1 {
                        let k = &s.key_cache[(loff + t * kv_dim + (h as u32 / kv_mul) * head_size)
                            as usize
                            ..(loff + t * kv_dim + (h as u32 / kv_mul) * head_size + head_size)
                                as usize];

                        let mut score: f32 = 0.0;

                        for i in 0..head_size {
                            score += q[i as usize] * k.data()[i as usize];
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
                            xb[i as usize] += a * v.data()[i as usize];
                        }
                    }
                });

            matmul(
                &mut s.xb2,
                &s.xb3,
                &w.wo[(l * dim * att_dim) as usize..(l * dim * att_dim + dim * att_dim) as usize],
            );

            let xb2 = s.xb2.data();
            for i in 0..dim {
                x[i as usize] += xb2[i as usize];
            }

            rmsnorm(
                &mut s.xb.data_mut(),
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

            let hb = s.hb.data_mut();
            let hb2 = s.hb2.data();
            for i in 0..hidden_dim {
                let mut val = hb[i as usize];

                val *= 1.0 / (1.0 + (-val).exp());

                val *= hb[i as usize];

                hb[i as usize] = val;
            }

            matmul(
                &mut s.xb,
                &s.hb,
                &w.w2[(l * dim * hidden_dim) as usize
                    ..(l * dim * hidden_dim + dim * hidden_dim) as usize],
            );

            let xb = s.xb.data();
            for i in 0..dim {
                x[i as usize] += xb[i as usize];
            }
        }

        s.xb.data_mut().copy_from_slice(x);

        rmsnorm(
            x,
            s.xb.data(),
            w.w_rms_final.as_matrix(),
            dim as usize,
            p.rms_norm_eps,
            p.model_type == ModelType::GEMMA,
        );

        matmul(&mut s.logits, &s.x, &w.w_cls.as_matrix());

        s.logits.data_mut()
    }
}
