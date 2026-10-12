# Ling-3.0-flash (124B/5.1B, BailingMoeV3)

Ant Group's native hybrid-reasoning MoE: 124B total, 5.1B active per token,
256K native context (to 1M), full tool calling and thinking. `ling30.c` runs
the official BF16 checkpoint with no conversion.

## Architecture (from `inclusionAI/Ling-3.0-flash`)

| item | value |
|---|---|
| `model_type` | `bailing_hybrid` (`BailingMoeV3ForCausalLM`) |
| layers | 42 (`layer_group_size` 6: MLA iff `(i+1)%6==0` or `i>=36`, i.e. 5,11,17,23,29,35,41; rest KDA) |
| hidden / heads | 2560 / 32 |
| MLA | `q_proj` direct (no LoRA), `kv_a_proj_with_mqa` 576, `kv_a_layernorm` 512, `kv_b_proj` 8192, head-wise `g_proj`, `dense`; `qk_nope` 128, `qk_rope` 64, `v_head` 128, `kv_lora` 512; qk-norm; interleaved RoPE, theta 6M |
| KDA | 32 heads x 128, short conv kernel 4 + SiLU, `f_proj`/`g_proj`, `b_proj` beta, `A_log`, `dt_bias`, lower bound -5.0, safe gate; recurrence in `delta_attention.h:coli_kda_step` (shared with Kimi/Qwen3.6/GLM-5.3-Flash) |
| MoE | 512 routed top-8 + 1 shared (768 each); dense layers 0..1 (6144); sigmoid + expert bias, group-limited topk (`n_group` 8, `topk_group` 4), renorm, `*2.5`; per-layer SwiGLU clamp lists (`0` = off) |
| MTP | `model.layers.42.*` draft head: downloaded, never loaded (no speculative decoding yet) |
| vocab | 157184 (`<|role_end|>` 156895 eos) |
| prompt | `<role>SYSTEM|HUMAN|ASSISTANT|OBSERVATION</role>...<|role_end|>`, cue `<role>ASSISTANT</role>\n<think>`; tools `<tools>{json}</tools>` + `<tool_call>name<arg_key>k</arg_key>\n<arg_value>v</arg_value>...</tool_call>`; tool results `<role>OBSERVATION</role>\n<tool_response>...</tool_response><|role_end|>` |

## Placement (the colibri contract)

Dense (embed, all attention projections, norms, head) stays resident;
experts stream from disk through a per-layer LRU (`LING30_CAP`, default 8 =
one routing step). Placement changes speed, never the router's decisions or
precision -- the same `route/union/place/overlap/learn` walk as every engine.

`coli plan` prices only the 7 MLA layers' positional cache
(`kv_lora+qk_rope+heads*v_head` per token); KDA state + conv windows are
context-independent fixed bytes. The gateway owns the chat template
(`render_chat_ling30`, `parse_ling30_tool_calls`); the engine tokenizes the
rendered prompt as-is.

## Run it

```bash
./coli chat --model /nvme/ling30        # needs inclusionAI/Ling-3.0-flash (~245 GB)
LING30_CAP=16 ./ling30 /nvme/ling30 --prompt "Hello" --ngen 32
make ling30-tiny-check                  # 6-layer fixture, token-exact vs numpy oracle
```

`--model ling-3.0-flash` in the one-step setup downloads the official
checkpoint (MTP shard excluded). CPU-only in this revision: no CUDA/Metal
backend and no Vulkan tier/chain yet (unmeasured on this hybrid, as mimo's
was at introduction) -- `supports_accelerator=False` so the planner offers
no VRAM tier it cannot execute.

## Files

`c/ling30.c` (engine) over `st.h` (shard reads), `delta_attention.h` (KDA),
`tok.h`, `serve_codec.h` (SERVE), `family_registry.py` (`ling30` descriptor +
`_ling30_geometry`), `setup_catalog.py` (`ling-3.0-flash`), `openai_server.py`
(gateway template), `tools/make_ling30_tiny.py` + `tests/test_ling30_chat_template.py`.
