//! f64 reference implementation — a direct port of reference/microgpt.py with
//! the same analytic backward pass as gpt-core. Used to validate both the
//! derivation (against finite differences) and the fixed-point precision
//! (against this).

use gpt_core::{BLOCK, BOS, HEAD_DIM, MLP_DIM, N_EMBD, N_HEAD, N_PARAMS, VOCAB};

pub const RMS_EPS: f64 = 1e-5;

#[derive(Clone)]
pub struct W64 {
    pub wte: Vec<Vec<f64>>, // VOCAB x N_EMBD
    pub wpe: Vec<Vec<f64>>, // BLOCK x N_EMBD
    pub wq: Vec<Vec<f64>>,
    pub wk: Vec<Vec<f64>>,
    pub wv: Vec<Vec<f64>>,
    pub wo: Vec<Vec<f64>>,
    pub w1: Vec<Vec<f64>>, // MLP_DIM x N_EMBD
    pub w2: Vec<Vec<f64>>, // N_EMBD x MLP_DIM
    pub lm: Vec<Vec<f64>>, // VOCAB x N_EMBD
}

fn mat(out: usize, inp: usize) -> Vec<Vec<f64>> {
    vec![vec![0.0; inp]; out]
}

impl W64 {
    pub fn zeros() -> Self {
        Self {
            wte: mat(VOCAB, N_EMBD),
            wpe: mat(BLOCK, N_EMBD),
            wq: mat(N_EMBD, N_EMBD),
            wk: mat(N_EMBD, N_EMBD),
            wv: mat(N_EMBD, N_EMBD),
            wo: mat(N_EMBD, N_EMBD),
            w1: mat(MLP_DIM, N_EMBD),
            w2: mat(N_EMBD, MLP_DIM),
            lm: mat(VOCAB, N_EMBD),
        }
    }

    /// Same flat parameter order as gpt_core::Weights.
    pub fn flat(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(N_PARAMS);
        for m in self.mats() {
            for row in m {
                out.extend_from_slice(row);
            }
        }
        out
    }

    pub fn set_flat(&mut self, flat: &[f64]) {
        let mut it = flat.iter();
        for m in self.mats_mut() {
            for row in m {
                for v in row.iter_mut() {
                    *v = *it.next().unwrap();
                }
            }
        }
    }

    fn mats(&self) -> [&Vec<Vec<f64>>; 9] {
        [&self.wte, &self.wpe, &self.wq, &self.wk, &self.wv, &self.wo, &self.w1, &self.w2, &self.lm]
    }

    fn mats_mut(&mut self) -> [&mut Vec<Vec<f64>>; 9] {
        [
            &mut self.wte,
            &mut self.wpe,
            &mut self.wq,
            &mut self.wk,
            &mut self.wv,
            &mut self.wo,
            &mut self.w1,
            &mut self.w2,
            &mut self.lm,
        ]
    }
}

fn linear(w: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    w.iter().map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum()).collect()
}

fn linear_t(w: &[Vec<f64>], dy: &[f64]) -> Vec<f64> {
    let mut dx = vec![0.0; w[0].len()];
    for (row, d) in w.iter().zip(dy) {
        for (dxi, wi) in dx.iter_mut().zip(row) {
            *dxi += wi * d;
        }
    }
    dx
}

fn outer_acc(dw: &mut [Vec<f64>], dy: &[f64], x: &[f64]) {
    for (row, d) in dw.iter_mut().zip(dy) {
        for (dwi, xi) in row.iter_mut().zip(x) {
            *dwi += d * xi;
        }
    }
}

fn rmsnorm(x: &[f64]) -> (Vec<f64>, f64) {
    let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + RMS_EPS;
    let s = 1.0 / ms.sqrt();
    (x.iter().map(|v| v * s).collect(), s)
}

fn rmsnorm_bwd(x: &[f64], s: f64, dy: &[f64]) -> Vec<f64> {
    let d: f64 = dy.iter().zip(x).map(|(a, b)| a * b).sum();
    let c = s * s * s * d / x.len() as f64;
    dy.iter().zip(x).map(|(dyi, xi)| s * dyi - c * xi).collect()
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::MIN, f64::max);
    let exps: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

pub struct Acts {
    x0: Vec<Vec<f64>>,
    x1: Vec<Vec<f64>>,
    xa: Vec<Vec<f64>>,
    q: Vec<Vec<f64>>,
    k: Vec<Vec<f64>>,
    v: Vec<Vec<f64>>,
    o: Vec<Vec<f64>>,
    x2: Vec<Vec<f64>>,
    xm: Vec<Vec<f64>>,
    x3: Vec<Vec<f64>>,
    att: Vec<Vec<Vec<f64>>>, // [t][h][s]
    hpre: Vec<Vec<f64>>,
    h: Vec<Vec<f64>>,
    pub probs: Vec<Vec<f64>>,
    s_emb: Vec<f64>,
    s_att: Vec<f64>,
    s_mlp: Vec<f64>,
}

impl Acts {
    pub fn new() -> Self {
        let m = || vec![vec![0.0; N_EMBD]; BLOCK];
        Self {
            x0: m(),
            x1: m(),
            xa: m(),
            q: m(),
            k: m(),
            v: m(),
            o: m(),
            x2: m(),
            xm: m(),
            x3: m(),
            att: vec![vec![vec![0.0; BLOCK]; N_HEAD]; BLOCK],
            hpre: vec![vec![0.0; MLP_DIM]; BLOCK],
            h: vec![vec![0.0; MLP_DIM]; BLOCK],
            probs: vec![vec![0.0; VOCAB]; BLOCK],
            s_emb: vec![0.0; BLOCK],
            s_att: vec![0.0; BLOCK],
            s_mlp: vec![0.0; BLOCK],
        }
    }
}

pub fn forward_pos(w: &W64, a: &mut Acts, t: usize, token: usize) -> Vec<f64> {
    for i in 0..N_EMBD {
        a.x0[t][i] = w.wte[token][i] + w.wpe[t][i];
    }
    let (x1, s1) = rmsnorm(&a.x0[t]);
    a.x1[t] = x1;
    a.s_emb[t] = s1;

    let (xa, sa) = rmsnorm(&a.x1[t]);
    a.xa[t] = xa;
    a.s_att[t] = sa;
    a.q[t] = linear(&w.wq, &a.xa[t]);
    a.k[t] = linear(&w.wk, &a.xa[t]);
    a.v[t] = linear(&w.wv, &a.xa[t]);

    let scale = 1.0 / (HEAD_DIM as f64).sqrt();
    for head in 0..N_HEAD {
        let hs = head * HEAD_DIM;
        let logits: Vec<f64> = (0..=t)
            .map(|s| (0..HEAD_DIM).map(|j| a.q[t][hs + j] * a.k[s][hs + j]).sum::<f64>() * scale)
            .collect();
        let weights = softmax(&logits);
        for j in 0..HEAD_DIM {
            a.o[t][hs + j] = (0..=t).map(|s| weights[s] * a.v[s][hs + j]).sum();
        }
        for (s, wgt) in weights.into_iter().enumerate() {
            a.att[t][head][s] = wgt;
        }
    }
    let xo = linear(&w.wo, &a.o[t]);
    for i in 0..N_EMBD {
        a.x2[t][i] = a.x1[t][i] + xo[i];
    }

    let (xm, sm) = rmsnorm(&a.x2[t]);
    a.xm[t] = xm;
    a.s_mlp[t] = sm;
    a.hpre[t] = linear(&w.w1, &a.xm[t]);
    a.h[t] = a.hpre[t].iter().map(|v| v.max(0.0)).collect();
    let xf = linear(&w.w2, &a.h[t]);
    for i in 0..N_EMBD {
        a.x3[t][i] = a.x2[t][i] + xf[i];
    }

    linear(&w.lm, &a.x3[t])
}

/// Forward + backward over one doc. Returns (loss, grads).
pub fn train_fwd_bwd(w: &W64, tokens: &[u8]) -> (f64, W64) {
    let n = BLOCK.min(tokens.len() - 1);
    let mut a = Acts::new();
    let mut g = W64::zeros();
    let mut dk = vec![vec![0.0; N_EMBD]; BLOCK];
    let mut dv = vec![vec![0.0; N_EMBD]; BLOCK];

    let mut loss = 0.0;
    for t in 0..n {
        let logits = forward_pos(w, &mut a, t, tokens[t] as usize);
        a.probs[t] = softmax(&logits);
        loss -= a.probs[t][tokens[t + 1] as usize].ln();
    }
    loss /= n as f64;

    let scale = 1.0 / (HEAD_DIM as f64).sqrt();
    for t in (0..n).rev() {
        let mut dlogits = a.probs[t].clone();
        dlogits[tokens[t + 1] as usize] -= 1.0;
        for d in dlogits.iter_mut() {
            *d /= n as f64;
        }

        outer_acc(&mut g.lm, &dlogits, &a.x3[t]);
        let dx3 = linear_t(&w.lm, &dlogits);

        let mut dx2 = dx3.clone();
        outer_acc(&mut g.w2, &dx3, &a.h[t]);
        let dh = linear_t(&w.w2, &dx3);
        let dhpre: Vec<f64> =
            dh.iter().zip(&a.hpre[t]).map(|(d, p)| if *p > 0.0 { *d } else { 0.0 }).collect();
        outer_acc(&mut g.w1, &dhpre, &a.xm[t]);
        let dxm = linear_t(&w.w1, &dhpre);
        for (i, d) in rmsnorm_bwd(&a.x2[t], a.s_mlp[t], &dxm).into_iter().enumerate() {
            dx2[i] += d;
        }

        let mut dx1 = dx2.clone();
        outer_acc(&mut g.wo, &dx2, &a.o[t]);
        let do_ = linear_t(&w.wo, &dx2);

        let mut dq = vec![0.0; N_EMBD];
        for head in 0..N_HEAD {
            let hs = head * HEAD_DIM;
            let att = &a.att[t][head];
            let da: Vec<f64> = (0..=t)
                .map(|s| (0..HEAD_DIM).map(|j| do_[hs + j] * a.v[s][hs + j]).sum())
                .collect();
            let dsum: f64 = (0..=t).map(|s| att[s] * da[s]).sum();
            for s in 0..=t {
                let dscore = att[s] * (da[s] - dsum) * scale;
                for j in 0..HEAD_DIM {
                    dq[hs + j] += dscore * a.k[s][hs + j];
                    dk[s][hs + j] += dscore * a.q[t][hs + j];
                    dv[s][hs + j] += att[s] * do_[hs + j];
                }
            }
        }

        outer_acc(&mut g.wq, &dq, &a.xa[t]);
        let mut dxa = linear_t(&w.wq, &dq);
        outer_acc(&mut g.wk, &dk[t], &a.xa[t]);
        for (i, d) in linear_t(&w.wk, &dk[t]).into_iter().enumerate() {
            dxa[i] += d;
        }
        outer_acc(&mut g.wv, &dv[t], &a.xa[t]);
        for (i, d) in linear_t(&w.wv, &dv[t]).into_iter().enumerate() {
            dxa[i] += d;
        }

        for (i, d) in rmsnorm_bwd(&a.x1[t], a.s_att[t], &dxa).into_iter().enumerate() {
            dx1[i] += d;
        }
        let dx0 = rmsnorm_bwd(&a.x0[t], a.s_emb[t], &dx1);
        let token = tokens[t] as usize;
        for i in 0..N_EMBD {
            g.wte[token][i] += dx0[i];
            g.wpe[t][i] += dx0[i];
        }
    }

    (loss, g)
}

/// Loss only (for finite differences).
pub fn loss_only(w: &W64, tokens: &[u8]) -> f64 {
    let n = BLOCK.min(tokens.len() - 1);
    let mut a = Acts::new();
    let mut loss = 0.0;
    for t in 0..n {
        let logits = forward_pos(w, &mut a, t, tokens[t] as usize);
        let probs = softmax(&logits);
        loss -= probs[tokens[t + 1] as usize].ln();
    }
    loss / n as f64
}

/// Adam, mirroring microgpt.py exactly.
pub struct Adam64 {
    pub m: Vec<f64>,
    pub v: Vec<f64>,
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
}

impl Adam64 {
    pub fn new() -> Self {
        Self {
            m: vec![0.0; N_PARAMS],
            v: vec![0.0; N_PARAMS],
            lr: 0.01,
            beta1: 0.85,
            beta2: 0.99,
            eps: 1e-8,
        }
    }

    pub fn step(&mut self, w: &mut W64, g: &W64, step: u64) {
        let mut flat = w.flat();
        let gflat = g.flat();
        let lr_t = (self.lr * (1.0 - step as f64 / 1000.0)).max(0.001);
        for i in 0..N_PARAMS {
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * gflat[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * gflat[i] * gflat[i];
            let m_hat = self.m[i] / (1.0 - self.beta1.powi(step as i32 + 1));
            let v_hat = self.v[i] / (1.0 - self.beta2.powi(step as i32 + 1));
            flat[i] -= lr_t * m_hat / (v_hat.sqrt() + self.eps);
        }
        w.set_flat(&flat);
    }
}

/// BOS-delimited token sequence for a name.
pub fn doc_tokens(name: &str) -> Vec<u8> {
    let mut t = vec![BOS];
    t.extend(name.bytes().map(|b| b - b'a'));
    t.push(BOS);
    t
}
