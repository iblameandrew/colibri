#!/usr/bin/env python3
"""Generate the tiny Ling-3.0-flash checkpoint used by ling30-tiny-check.

numpy-only and deterministic: CI regenerates the shards from SEED and
byte-identical weights come out on every platform. ref.json is produced HERE
by an independent numpy forward of the vendor math (modeling_bailing_moe_v3.py:
group-limited topk routing, head-wise gated MLA with interleaved RoPE, KDA
gated delta rule via the same formulas delta_attention.h implements, SwiGLU
experts with per-layer clamp lists), so the C engine and its oracle share no
code -- only the architecture.

The fixture stores every tensor the way the released checkpoint does (BF16
names, F32 values for exactness at this scale), because the names are where a
reading of BailingMoeV3 goes wrong while a float reference still agrees:

  - MLA layers carry q_proj / kv_a_proj_with_mqa / kv_a_layernorm / kv_b_proj /
    g_proj / dense (o_proj spelling also accepted by the engine);
  - KDA layers carry q/k/v/f/g/o/b projections plus q/k/v_conv1d, A_log, dt_bias;
  - MoE layers carry mlp.gate.weight + expert_bias, experts.{e}.gate/up/down
    and shared_experts.gate/up/down; layers 0..1 are dense MLPs;
  - layer choice follows (i+1)%group==0 or i>=n//g*g with group_size 3 here
    (2 MLA + 4 KDA over 6 layers, the same rule as the 42-layer release);
  - model.layers.42.* (the MTP draft head) is present and must be ignored.

Geometry: hidden 64, heads 4, qk_nope 8, qk_rope 4, v_head 8, kv_lora 8,
kda_hd 8, conv kernel 2, 8 experts top-2 over 4 groups (top-2 groups kept),
moe_inter 16, dense_inter 64, vocab 128. No tokenizer.json on purpose: the
engine prints raw generated ids, which is what the token-exact test parses.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

SEED = 20261012

VOCAB = 128
HIDDEN = 64
LAYERS = 6
GROUP = 3
FIRST_DENSE = 2
HEADS = 4
QK_NOPE = 8
QK_ROPE = 4
QK_HEAD = QK_NOPE + QK_ROPE
V_HEAD = 8
KV_LORA = 8
KDA_HD = 8
CONV_K = 2
EXPERTS = 8
TOPK = 2
N_GROUP = 4
TOPK_GROUP = 2
MOE_INTER = 16
DENSE_INTER = 64
SHARED = 1
EOS_ID = 1
ROUTE_SCALE = 2.5
EPS = 1e-6
THETA = 6000000.0
KDA_LB = -5.0
PROMPT = [5, 17, 42, 9, 100, 33, 77, 12]
NGEN = 8

MLA_LAYERS = [i for i in range(LAYERS) if (i + 1) % GROUP == 0 or i >= LAYERS // GROUP * GROUP]

CONFIG = {
    "architectures": ["BailingMoeV3ForCausalLM"],
    "model_type": "bailing_hybrid",
    "vocab_size": VOCAB,
    "hidden_size": HIDDEN,
    "num_hidden_layers": LAYERS,
    "layer_group_size": GROUP,
    "intermediate_size": DENSE_INTER,
    "moe_intermediate_size": MOE_INTER,
    "moe_shared_expert_intermediate_size": MOE_INTER,
    "num_experts": EXPERTS,
    "num_experts_per_tok": TOPK,
    "n_group": N_GROUP,
    "topk_group": TOPK_GROUP,
    "num_shared_experts": SHARED,
    "first_k_dense_replace": FIRST_DENSE,
    "num_attention_heads": HEADS,
    "num_key_value_heads": HEADS,
    "qk_nope_head_dim": QK_NOPE,
    "qk_rope_head_dim": QK_ROPE,
    "v_head_dim": V_HEAD,
    "kv_lora_rank": KV_LORA,
    "q_lora_rank": None,
    "head_dim": KDA_HD,
    "short_conv_kernel_size": CONV_K,
    "rope_theta": THETA,
    "rope_interleave": True,
    "rms_norm_eps": EPS,
    "hidden_act": "silu",
    "tie_word_embeddings": False,
    "max_position_embeddings": 256,
    "eos_token_id": EOS_ID,
    "pad_token_id": 0,
    "routed_scaling_factor": ROUTE_SCALE,
    "norm_topk_prob": True,
    "moe_router_enable_expert_bias": True,
    "use_qk_norm": True,
    "gated_attention_proj_granularity_type": "head_wise",
    "kda_lower_bound": KDA_LB,
    "kda_safe_gate": True,
    "no_kda_lora": True,
    "expert_swiglu_limit_list": [0] * LAYERS,
    "share_expert_swiglu_limit_list": [0] * LAYERS,
    "num_nextn_predict_layers": 1,
}


def normal(rng, *shape, scale=0.08):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def build_tensors(rng):
    T = {}
    T["model.embed_tokens.weight"] = normal(rng, VOCAB, HIDDEN, scale=0.05)
    T["model.norm.weight"] = np.ones(HIDDEN, np.float32)
    T["lm_head.weight"] = normal(rng, VOCAB, HIDDEN, scale=0.25)
    P = HEADS * KDA_HD
    for i in range(LAYERS):
        T[f"model.layers.{i}.input_layernorm.weight"] = np.ones(HIDDEN, np.float32)
        T[f"model.layers.{i}.post_attention_layernorm.weight"] = np.ones(HIDDEN, np.float32)
        if i in MLA_LAYERS:
            T[f"model.layers.{i}.self_attn.q_proj.weight"] = normal(rng, HEADS * QK_HEAD, HIDDEN)
            T[f"model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight"] = normal(rng, KV_LORA + QK_ROPE, HIDDEN)
            T[f"model.layers.{i}.self_attn.kv_a_layernorm.weight"] = np.ones(KV_LORA, np.float32)
            T[f"model.layers.{i}.self_attn.kv_b_proj.weight"] = normal(rng, HEADS * (QK_NOPE + V_HEAD), KV_LORA)
            T[f"model.layers.{i}.self_attn.g_proj.weight"] = normal(rng, HEADS, HIDDEN, scale=0.02)
            T[f"model.layers.{i}.self_attn.dense.weight"] = normal(rng, HIDDEN, HEADS * V_HEAD)
        else:
            T[f"model.layers.{i}.self_attn.q_proj.weight"] = normal(rng, P, HIDDEN)
            T[f"model.layers.{i}.self_attn.k_proj.weight"] = normal(rng, P, HIDDEN)
            T[f"model.layers.{i}.self_attn.v_proj.weight"] = normal(rng, P, HIDDEN)
            T[f"model.layers.{i}.self_attn.f_proj.weight"] = normal(rng, P, HIDDEN)
            T[f"model.layers.{i}.self_attn.g_proj.weight"] = normal(rng, P, HIDDEN)
            T[f"model.layers.{i}.self_attn.o_proj.weight"] = normal(rng, HIDDEN, P)
            T[f"model.layers.{i}.self_attn.b_proj.weight"] = normal(rng, HEADS, HIDDEN, scale=0.02)
            T[f"model.layers.{i}.self_attn.q_conv1d.weight"] = normal(rng, P, CONV_K, scale=0.1)
            T[f"model.layers.{i}.self_attn.k_conv1d.weight"] = normal(rng, P, CONV_K, scale=0.1)
            T[f"model.layers.{i}.self_attn.v_conv1d.weight"] = normal(rng, P, CONV_K, scale=0.1)
            T[f"model.layers.{i}.self_attn.A_log"] = np.log(rng.uniform(1, 16, size=HEADS).astype(np.float32))
            T[f"model.layers.{i}.self_attn.dt_bias"] = normal(rng, P, scale=0.02)
        if i < FIRST_DENSE:
            T[f"model.layers.{i}.mlp.gate_proj.weight"] = normal(rng, DENSE_INTER, HIDDEN)
            T[f"model.layers.{i}.mlp.up_proj.weight"] = normal(rng, DENSE_INTER, HIDDEN)
            T[f"model.layers.{i}.mlp.down_proj.weight"] = normal(rng, HIDDEN, DENSE_INTER)
        else:
            T[f"model.layers.{i}.mlp.gate.weight"] = normal(rng, EXPERTS, HIDDEN, scale=0.05)
            T[f"model.layers.{i}.mlp.expert_bias"] = np.zeros(EXPERTS, np.float32)
            T[f"model.layers.{i}.mlp.shared_experts.gate_proj.weight"] = normal(rng, MOE_INTER, HIDDEN)
            T[f"model.layers.{i}.mlp.shared_experts.up_proj.weight"] = normal(rng, MOE_INTER, HIDDEN)
            T[f"model.layers.{i}.mlp.shared_experts.down_proj.weight"] = normal(rng, HIDDEN, MOE_INTER)
            for e in range(EXPERTS):
                T[f"model.layers.{i}.mlp.experts.{e}.gate_proj.weight"] = normal(rng, MOE_INTER, HIDDEN)
                T[f"model.layers.{i}.mlp.experts.{e}.up_proj.weight"] = normal(rng, MOE_INTER, HIDDEN)
                T[f"model.layers.{i}.mlp.experts.{e}.down_proj.weight"] = normal(rng, HIDDEN, MOE_INTER)
    # MTP draft head the engine must ignore (release layout: model.layers.42.*)
    T["model.layers.42.input_layernorm.weight"] = np.ones(HIDDEN, np.float32)
    T["model.layers.42.self_attn.q_proj.weight"] = normal(rng, HEADS * QK_HEAD, HIDDEN)
    return T


def silu(x):
    return x / (1.0 + np.exp(-x))


def rmsn(x, w):
    return x / np.sqrt((x ** 2).mean() + EPS) * w


def route(scores):
    per = EXPERTS // N_GROUP
    gs = np.zeros(N_GROUP, np.float32)
    for g in range(N_GROUP):
        top2 = np.sort(scores[g * per:(g + 1) * per])[-2:]
        gs[g] = top2.sum()
    keep = set(np.argsort(-gs)[:TOPK_GROUP].tolist())
    masked = np.array([s if (e // per) in keep else -1e30 for e, s in enumerate(scores)], np.float32)
    idx = np.argsort(-masked)[:TOPK]
    val = masked[idx]
    val = val / val.sum() * ROUTE_SCALE
    return idx, val


def ref_forward(T, ids):
    H = HIDDEN
    P = HEADS * KDA_HD
    kstate = {i: np.zeros((HEADS, KDA_HD, KDA_HD), np.float32) for i in range(LAYERS) if i not in MLA_LAYERS}
    kwin = {i: np.zeros((3 * P, CONV_K), np.float32) for i in range(LAYERS) if i not in MLA_LAYERS}
    mK = {i: [] for i in MLA_LAYERS}
    mV = {i: [] for i in MLA_LAYERS}
    invf = 1.0 / (THETA ** (np.arange(QK_ROPE // 2) * 2.0 / QK_ROPE))
    logits = None
    for pos, tok in enumerate(ids):
        h = T["model.embed_tokens.weight"][tok].copy()
        for i in range(LAYERS):
            h2 = rmsn(h, T[f"model.layers.{i}.input_layernorm.weight"])
            if i in MLA_LAYERS:
                q = T[f"model.layers.{i}.self_attn.q_proj.weight"] @ h2
                ckv = T[f"model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight"] @ h2
                kp, kr = ckv[:KV_LORA], ckv[KV_LORA:]
                kp = rmsn(kp, T[f"model.layers.{i}.self_attn.kv_a_layernorm.weight"])
                kb = T[f"model.layers.{i}.self_attn.kv_b_proj.weight"] @ kp
                q = q.reshape(HEADS, QK_HEAD)
                kbh = kb.reshape(HEADS, QK_NOPE + V_HEAD)
                q = q / np.sqrt((q ** 2).mean(axis=1, keepdims=True) + EPS)
                kbh[:, :QK_NOPE] /= np.sqrt((kbh[:, :QK_NOPE] ** 2).mean(axis=1, keepdims=True) + EPS)
                for hh in range(HEADS):
                    for j in range(0, QK_ROPE, 2):
                        f = invf[j // 2] * pos
                        co, si = np.cos(f), np.sin(f)
                        a0, a1 = q[hh, QK_NOPE + j], q[hh, QK_NOPE + j + 1]
                        q[hh, QK_NOPE + j], q[hh, QK_NOPE + j + 1] = a0 * co - a1 * si, a0 * si + a1 * co
                mK[i].append(np.concatenate([kp, kr]))
                mV[i].append(kbh[:, QK_NOPE:].copy())
                outs = np.zeros((HEADS, V_HEAD), np.float32)
                for hh in range(HEADS):
                    sc = []
                    for t in range(pos + 1):
                        Kt = mK[i][t]
                        kfull = T[f"model.layers.{i}.self_attn.kv_b_proj.weight"] @ Kt[:KV_LORA]
                        kh = kfull.reshape(HEADS, QK_NOPE + V_HEAD)[hh]
                        d = float(q[hh, :QK_NOPE] @ kh[:QK_NOPE])
                        for j in range(0, QK_ROPE, 2):
                            f = invf[j // 2] * t
                            co, si = np.cos(f), np.sin(f)
                            k0, k1 = Kt[KV_LORA + j], Kt[KV_LORA + j + 1]
                            rk0, rk1 = k0 * co - k1 * si, k0 * si + k1 * co
                            d += float(q[hh, QK_NOPE + j] * rk0 + q[hh, QK_NOPE + j + 1] * rk1)
                        sc.append(d / np.sqrt(QK_HEAD))
                    sc = np.array(sc, np.float32)
                    sc -= sc.max()
                    w = np.exp(sc)
                    w /= w.sum()
                    for t in range(pos + 1):
                        outs[hh] += w[t] * mV[i][t][hh]
                g = 1.0 / (1.0 + np.exp(-(T[f"model.layers.{i}.self_attn.g_proj.weight"] @ h2)))
                outs = outs * g[:, None]
                att = T[f"model.layers.{i}.self_attn.dense.weight"] @ outs.reshape(-1)
            else:
                qq = T[f"model.layers.{i}.self_attn.q_proj.weight"] @ h2
                kk = T[f"model.layers.{i}.self_attn.k_proj.weight"] @ h2
                vv = T[f"model.layers.{i}.self_attn.v_proj.weight"] @ h2
                ff = T[f"model.layers.{i}.self_attn.f_proj.weight"] @ h2 + T[f"model.layers.{i}.self_attn.dt_bias"]
                gg = T[f"model.layers.{i}.self_attn.g_proj.weight"] @ h2
                mix = np.concatenate([qq, kk, vv])
                cw = np.concatenate([T[f"model.layers.{i}.self_attn.q_conv1d.weight"],
                                     T[f"model.layers.{i}.self_attn.k_conv1d.weight"],
                                     T[f"model.layers.{i}.self_attn.v_conv1d.weight"]], axis=0)
                win = kwin[i]
                win[:, :-1] = win[:, 1:]
                win[:, -1] = mix
                mix = silu((win * cw).sum(axis=1))
                q3, k3, v3 = mix[:P], mix[P:2 * P], mix[2 * P:]
                q3 = q3.reshape(HEADS, KDA_HD)
                k3 = k3.reshape(HEADS, KDA_HD)
                v3 = v3.reshape(HEADS, KDA_HD)
                q3 = q3 / np.sqrt((q3 ** 2).sum(axis=1, keepdims=True) + EPS) / np.sqrt(KDA_HD)
                k3 = k3 / np.sqrt((k3 ** 2).sum(axis=1, keepdims=True) + EPS)
                ea = np.exp(T[f"model.layers.{i}.self_attn.A_log"])
                gate = KDA_LB * (1.0 / (1.0 + np.exp(-(ea[:, None] * ff.reshape(HEADS, KDA_HD)))))
                beta = 1.0 / (1.0 + np.exp(-(T[f"model.layers.{i}.self_attn.b_proj.weight"] @ h2)))
                S = kstate[i]
                out = np.zeros((HEADS, KDA_HD), np.float32)
                for hh in range(HEADS):
                    S[hh] *= np.exp(gate[hh])[:, None]
                    mem = (k3[hh][:, None] * S[hh]).sum(axis=0)
                    S[hh] += k3[hh][:, None] * ((v3[hh] - mem) * beta[hh])[None, :]
                    out[hh] = S[hh].T @ q3[hh]
                out = out * (1.0 / (1.0 + np.exp(-gg.reshape(HEADS, KDA_HD))))
                out = out / np.sqrt((out ** 2).mean(axis=1, keepdims=True) + EPS)
                att = T[f"model.layers.{i}.self_attn.o_proj.weight"] @ out.reshape(-1)
            h = h + att
            h2 = rmsn(h, T[f"model.layers.{i}.post_attention_layernorm.weight"])
            if i < FIRST_DENSE:
                g = silu(T[f"model.layers.{i}.mlp.gate_proj.weight"] @ h2)
                u = T[f"model.layers.{i}.mlp.up_proj.weight"] @ h2
                h = h + T[f"model.layers.{i}.mlp.down_proj.weight"] @ (g * u)
            else:
                sg = silu(T[f"model.layers.{i}.mlp.shared_experts.gate_proj.weight"] @ h2)
                su = T[f"model.layers.{i}.mlp.shared_experts.up_proj.weight"] @ h2
                mout = T[f"model.layers.{i}.mlp.shared_experts.down_proj.weight"] @ (sg * su)
                lg = T[f"model.layers.{i}.mlp.gate.weight"] @ h2
                sc = 1.0 / (1.0 + np.exp(-(lg + T[f"model.layers.{i}.mlp.expert_bias"])))
                idx, val = route(sc)
                for k, e in enumerate(idx):
                    g = silu(T[f"model.layers.{i}.mlp.experts.{e}.gate_proj.weight"] @ h2)
                    u = T[f"model.layers.{i}.mlp.experts.{e}.up_proj.weight"] @ h2
                    mout = mout + val[k] * (T[f"model.layers.{i}.mlp.experts.{e}.down_proj.weight"] @ (g * u))
                h = h + mout
        logits = rmsn(h, T["model.norm.weight"]) @ T["lm_head.weight"].T
    return logits


def write_safetensors(path, tensors):
    header, offset = {}, 0
    for name, arr in tensors.items():
        header[name] = {"dtype": "F32", "shape": list(arr.shape),
                        "data_offsets": [offset, offset + arr.nbytes]}
        offset += arr.nbytes
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * ((-len(blob)) % 8)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        for arr in tensors.values():
            fh.write(np.ascontiguousarray(arr).tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="./ling30_tiny")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "model.safetensors").exists() and not args.force:
        print(f"{out} exists; use --force", file=sys.stderr)
        return 1
    rng = np.random.default_rng(SEED)
    tensors = build_tensors(rng)
    write_safetensors(out / "model.safetensors", tensors);
    (out / "config.json").write_text(json.dumps(dict(CONFIG), indent=2) + "\n")
    ids = list(PROMPT)
    for _ in range(NGEN):
        lg = ref_forward(tensors, ids)
        ids.append(int(np.argmax(lg)))
    (out / "ref.json").write_text(json.dumps(
        {"prompt_ids": PROMPT, "full_ids": ids, "ngen": NGEN,
         "mla_layers": MLA_LAYERS, "note": "numpy oracle, greedy"}, indent=2) + "\n")
    print(f"wrote {out}/model.safetensors + config.json + ref.json "
          f"(mla={MLA_LAYERS}, full_tail={ids[len(PROMPT):]})")


if __name__ == "__main__":
    main()
