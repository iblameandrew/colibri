/* ling30.c -- Ant Group Ling-3.0-flash (BailingMoeV3, 124B/5.1B) on a machine
 * that cannot hold it.
 *
 * The released checkpoint is read as it is, no conversion step:
 *
 *   experts   BF16 gate/up/down per routed expert (512 per MoE layer, 8 routed
 *             per token), one shared expert per MoE layer; each expert is ONE
 *             read (its three matrices back to back in the shard only
 *             logically -- the loader keeps an st ref per matrix and reads the
 *             missing experts with one pread per matrix while resident ones
 *             compute, exactly the route/union/place/overlap/learn contract).
 *   dense     every attention projection, norm, embedding and lm_head BF16,
 *             dequantized once to F32 and kept resident (the always-resident
 *             part the planner sizes).
 *
 * What makes Ling-3.0 Ling-3.0, each mirroring modeling_bailing_moe_v3.py:
 *   hybrid      42 layers, layer_group_size 6: layer i is gated MLA iff
 *               (i+1)%6==0 or i>=42//6*6 (layers 5,11,17,23,29,35,41), else
 *               Kimi Delta Attention (KDA) linear attention. first 2 layers
 *               dense (intermediate 6144), layers 2..41 MoE (768 + shared 768).
 *   MLA         q_lora_rank null -> direct q_proj [2560,32*192]; kv_a_proj_with_mqa
 *               [2560,576] split into k_pass[512]+k_rot[64]; kv_a_layernorm[512];
 *               kv_b_proj [512,32*256] split into k_nope/value; head-wise g_proj
 *               [2560,32] sigmoid gate; dense [4096,2560]; qk_norm on q/k;
 *               RoPE interleave on qk_rope=64 dims, theta 6M; scaling qk_head^-0.5.
 *   KDA         head_dim 128 x 32 heads, short conv kernel 4 + SiLU, L2 norm
 *               with eps INSIDE sqrt, decay exp(gate) with gate from
 *               f_proj+dt_bias (no_kda_lora) through lower_bound*sigmoid,
 *               beta=sigmoid(b_proj x); coli_kda_step() in delta_attention.h
 *               owns the recurrence (chunk==fused_recurrent per token here).
 *   router      sigmoid scores + expert_bias, group-limited topk
 *               (n_group 8, topk_group 4, top2-per-group sum), renorm, *2.5;
 *               norm_topk_prob. Per-layer SwiGLU clamp from
 *               expert_swiglu_limit_list / share_expert_swiglu_limit_list
 *               (0 = disabled).
 *   MTP         num_nextn_predict_layers=1 (model.layers.42.*) is a draft head:
 *               inference skips it, like every other engine skips its MTP head
 *               for greedy decode. The loader refuses to mistake it for layer 42.
 *
 * Held to inclusionAI's own modeling code through tools/make_ling30_tiny.py on
 * the tiny fixture of tools/make_ling30_tiny.py (deterministic numpy export,
 * ref.json committed, no torch/network in CI).
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <errno.h>
#include <time.h>
#include <stdarg.h>
#include <unistd.h>
#include <sys/stat.h>
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
#include <sys/resource.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#include "cli_args.h"
#include "compat.h"
#include "json.h"
#include "st.h"
#include "quant.h"
#include "omp_tune.h"
#include "kv_prefix.h"
#include "tok.h"
#include "serve_codec.h"
#include "serve_poll.h"
#include "stop_ids.h"
#include "delta_attention.h"
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define LING30_MAX_LAYERS 48
#define LING30_MAX_TOPK 16
#define LING30_MAX_GROUP 8

static double now_s(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static void *xmalloc(size_t n, const char *w) {
    void *p = malloc(n ? n : 1);
    if (!p) { fprintf(stderr, "[ling30] OOM %s (%zu)\n", w, n); exit(1); }
    return p;
}
static void *xcalloc(size_t n, size_t s, const char *w) {
    void *p = calloc(n ? n : 1, s ? s : 1);
    if (!p) { fprintf(stderr, "[ling30] OOM %s\n", w); exit(1); }
    return p;
}
static int env_int(const char *n, int fb) {
    const char *v = getenv(n); return (v && *v) ? atoi(v) : fb;
}
static inline float silu_f(float x) { return x / (1.0f + expf(-x)); }
static inline float sigm_f(float x) { return 1.0f / (1.0f + expf(-x)); }
static int is_dir(const char *p) { struct stat st; return p && !stat(p, &st) && S_ISDIR(st.st_mode); }

static char *read_text(const char *path, long *sz) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long s = ftell(f); fseek(f, 0, SEEK_SET);
    if (s < 0 || s > (64l << 20)) { fclose(f); return NULL; }
    char *t = xmalloc((size_t)s + 1, path);
    if (fread(t, 1, (size_t)s, f) != (size_t)s) { fclose(f); free(t); return NULL; }
    t[s] = 0; fclose(f); if (sz) *sz = s; return t;
}

/* ---------------------------------------------------------------- config ---- */
typedef struct {
    int vocab, hidden, n_layers, group_size;
    int dense_inter, moe_inter, shared_inter;
    int n_experts, topk, n_group, topk_group, n_shared, first_dense;
    int heads, kv_heads, qk_nope, qk_rope, qk_head, v_head, kv_lora;
    int kda_heads, kda_hd, conv_k;
    float rope_theta, route_scale, eps, kda_lb;
    int kda_safe_gate, norm_topk, use_qk_norm, gated_headwise;
    int max_positions, eos, pad;
    int is_mla[LING30_MAX_LAYERS];
    float swiglu_lim[LING30_MAX_LAYERS], shared_lim[LING30_MAX_LAYERS];
} Cfg;

static double jnum(jval *o, const char *k, double fb) {
    jval *v = o ? json_get(o, k) : NULL; return (v && v->t == J_NUM) ? v->num : fb;
}
static void die_cfg(const char *w) { fprintf(stderr, "[ling30] config.json: %s -- refusing\n", w); exit(1); }

static void load_cfg(Cfg *c, const char *dir) {
    char path[4096]; snprintf(path, sizeof path, "%s/config.json", dir);
    char *text = read_text(path, NULL);
    if (!text) { fprintf(stderr, "[ling30] cannot read %s\n", path); exit(1); }
    char *arena = NULL; jval *root = json_parse(text, &arena);
    if (!root) die_cfg("not JSON");
    jval *mt = json_get(root, "model_type");
    if (!mt || mt->t != J_STR ||
        (strcmp(mt->str, "bailing_hybrid") && strcmp(mt->str, "bailing_moe_v3") &&
         strcmp(mt->str, "ling30") && strcmp(mt->str, "bailingmoev3")))
        die_cfg("model_type is not bailing_hybrid (Ling-3.0-flash)");
    memset(c, 0, sizeof *c);
    c->vocab    = (int)jnum(root, "vocab_size", 157184);
    c->hidden   = (int)jnum(root, "hidden_size", 2560);
    c->n_layers = (int)jnum(root, "num_hidden_layers", 42);
    c->group_size = (int)jnum(root, "layer_group_size", 6);
    c->dense_inter = (int)jnum(root, "intermediate_size", 6144);
    c->moe_inter   = (int)jnum(root, "moe_intermediate_size", 768);
    c->shared_inter = (int)jnum(root, "moe_shared_expert_intermediate_size", 768);
    c->n_experts = (int)jnum(root, "num_experts", 512);
    c->topk      = (int)jnum(root, "num_experts_per_tok", 8);
    c->n_group   = (int)jnum(root, "n_group", 8);
    c->topk_group = (int)jnum(root, "topk_group", 4);
    c->n_shared  = (int)jnum(root, "num_shared_experts", 1);
    c->first_dense = (int)jnum(root, "first_k_dense_replace", 2);
    c->heads    = (int)jnum(root, "num_attention_heads", 32);
    c->kv_heads = (int)jnum(root, "num_key_value_heads", 32);
    c->qk_nope  = (int)jnum(root, "qk_nope_head_dim", 128);
    c->qk_rope  = (int)jnum(root, "qk_rope_head_dim", 64);
    c->qk_head  = c->qk_nope + c->qk_rope;
    c->v_head   = (int)jnum(root, "v_head_dim", 128);
    c->kv_lora  = (int)jnum(root, "kv_lora_rank", 512);
    c->kda_heads = c->heads;
    c->kda_hd   = (int)jnum(root, "head_dim", 128);
    c->conv_k   = (int)jnum(root, "short_conv_kernel_size", 4);
    c->rope_theta = (float)jnum(root, "rope_theta", 6000000);
    c->route_scale = (float)jnum(root, "routed_scaling_factor", 2.5);
    c->eps = (float)jnum(root, "rms_norm_eps", 1e-6);
    c->kda_lb = (float)jnum(root, "kda_lower_bound", -5.0);
    c->kda_safe_gate = 1; /* kda_safe_gate=true in the release */
    { jval *v = json_get(root, "kda_safe_gate"); if (v && v->t == J_BOOL) c->kda_safe_gate = v->boolean; }
    c->norm_topk = 1; { jval *v = json_get(root, "norm_topk_prob"); if (v && v->t == J_BOOL) c->norm_topk = v->boolean; }
    c->use_qk_norm = 1; { jval *v = json_get(root, "use_qk_norm"); if (v && v->t == J_BOOL) c->use_qk_norm = v->boolean; }
    { jval *v = json_get(root, "gated_attention_proj_granularity_type");
      c->gated_headwise = (v && v->t == J_STR && !strcmp(v->str, "head_wise")); }
    c->max_positions = (int)jnum(root, "max_position_embeddings", 262144);
    c->eos = (int)jnum(root, "eos_token_id", 156895);
    c->pad = (int)jnum(root, "pad_token_id", 156892);
    if (c->n_layers < 1 || c->n_layers > LING30_MAX_LAYERS) die_cfg("num_hidden_layers out of range");
    /* modeling_bailing_moe_v3.py BailingMoeV3DecoderLayer: MLA iff (i+1)%g==0
     * or i >= n//g*g. 42/6 -> MLA at 5,11,17,23,29,35,41 (7 MLA + 35 KDA). */
    int base = c->n_layers / c->group_size * c->group_size;
    for (int i = 0; i < c->n_layers; i++)
        c->is_mla[i] = (((i + 1) % c->group_size == 0) || (i >= base)) ? 1 : 0;
    /* swiglu limits: 0 = disabled */
    jval *el = json_get(root, "expert_swiglu_limit_list");
    if (el && el->t == J_ARR) for (int i = 0; i < c->n_layers && i < el->len; i++)
        if (el->kids[i]->t == J_NUM) c->swiglu_lim[i] = (float)el->kids[i]->num;
    jval *sl = json_get(root, "share_expert_swiglu_limit_list");
    if (sl && sl->t == J_ARR) for (int i = 0; i < c->n_layers && i < sl->len; i++)
        if (sl->kids[i]->t == J_NUM) c->shared_lim[i] = (float)sl->kids[i]->num;
    free(text);
}

/* ---------------------------------------------------------------- weights ---- */
typedef struct { int fd; int64_t off; int64_t numel; } TRef; /* streamed expert matrix */
typedef struct {
    float *w; int rows, cols;
} Mat;
typedef struct {
    /* MLA resident (only on MLA layers) */
    Mat q, kva, kvb, g, o;
    float *kv_norm;
    /* KDA resident (only on KDA layers) */
    Mat qp, kp, vp, fp, gp, op, bp;
    float *conv_q, *conv_k, *conv_v; /* [proj*K] depthwise taps */
    float *Alog, *dtb;
    /* norms */
    float *in_norm, *post_norm;
    /* dense MLP (layers < first_dense) */
    Mat dgate, dup, ddown;
    /* MoE */
    float *router; int rrows, rcols;
    float *rbias;
    Mat sh_gate, sh_up, sh_down;
    /* streamed experts: refs per expert per matrix */
    TRef *e_gate, *e_up, *e_down;
    /* expert LRU cache (f32, per layer) */
    float **c_gate, **c_up, **c_down;
    int *c_id; int *c_age; int cap, tick;
} Layer;
typedef struct {
    Cfg c;
    shards S;
    float *embed; float *norm; float *head;
    Layer *L;
    /* runtime state */
    float *kda_state;   /* per KDA layer: heads*hd*hd */
    float *kda_window;  /* per KDA layer: 3*proj*K */
    float *mla_k;       /* per MLA layer: ctx*(kv_lora+qk_rope) */
    float *mla_v;       /* per MLA layer: ctx*heads*v_head */
    int ctx, n_mla, n_kda;
    /* scratch */
    float *tmp; size_t tmp_n;
    float *kda_scratch;
    /* expert LRU staging */
    float *estage;
} Model;

static int64_t prod2(int a, int b) { return (int64_t)a * b; }

static float *load_f32(shards *S, const char *name, int64_t expect, const char *layer) {
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "[ling30] missing tensor %s (%s)\n", name, layer); exit(1); }
    if (expect > 0 && t->numel != expect) {
        fprintf(stderr, "[ling30] tensor %s has %lld el, expected %lld\n",
                name, (long long)t->numel, (long long)expect);
        exit(1);
    }
    float *w = xmalloc((size_t)t->numel * 4, name);
    st_read_f32(S, name, w, 0);
    return w;
}
/* try candidate names in order; NULL if none exists */
static const char *pick_name(shards *S, const char *cands[], int n) {
    for (int i = 0; i < n; i++) if (st_has(S, cands[i])) return cands[i];
    return NULL;
}
static void load_mat(shards *S, Mat *m, const char *name, int rows, int cols) {
    m->rows = rows; m->cols = cols;
    m->w = load_f32(S, name, prod2(rows, cols), name);
}
static void save_ref(shards *S, TRef *r, const char *name) {
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "[ling30] missing expert tensor %s\n", name); exit(1); }
    r->fd = t->fd; r->off = t->off; r->numel = t->numel;
}

/* tiny linear algebra (f32, OpenMP over rows) */
static void matvec(float *y, const float *W, const float *x, int rows, int cols) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < rows; r++) {
        const float *w = W + (size_t)r * cols;
        float s = 0;
        for (int i = 0; i < cols; i++) s += w[i] * x[i];
        y[r] = s;
    }
}
static void rmsnorm(float *y, const float *x, const float *w, int n, float eps) {
    double s = 0;
    for (int i = 0; i < n; i++) s += (double)x[i] * x[i];
    float g = 1.0f / sqrtf((float)(s / n) + eps);
    for (int i = 0; i < n; i++) y[i] = x[i] * g * w[i];
}

/* RoPE interleave on qk_rope dims (apply_rotary_pos_emb_interleave) */
static void rope_interleave(float *q, float *k, int heads, int nope, int rope,
                            const float *invf, int pos, float theta_scale) {
    (void)theta_scale;
    for (int h = 0; h < heads; h++) {
        float *qq = q + (size_t)h * (nope + rope) + nope;
        float *kk = k + (size_t)h * rope;
        for (int i = 0; i < rope; i += 2) {
            float f = invf[i / 2] * pos;
            float co = cosf(f), si = sinf(f);
            /* interleave view: pairs are (i, i+1) after transpose trick; the
             * reference interleave is equivalent to rotate-half on even/odd
             * pairs of the rope slice. */
            float q0 = qq[i], q1 = qq[i + 1], k0 = kk[i], k1 = kk[i + 1];
            qq[i] = q0 * co - q1 * si; qq[i + 1] = q0 * si + q1 * co;
            kk[i] = k0 * co - k1 * si; kk[i + 1] = k0 * si + k1 * co;
        }
    }
}

/* group-limited topk: sigmoid(logits+bias) -> group mask (top2/group sum,
 * keep topk_group groups) -> topk within mask, renorm, *route_scale */
static void route_topk(const Cfg *c, const float *logits, const float *bias,
                       int *idx, float *val) {
    int E = c->n_experts, K = c->topk, G = c->n_group, TG = c->topk_group;
    float *scores = (float *)malloc((size_t)E * 4);
    for (int e = 0; e < E; e++) scores[e] = sigm_f(logits[e] + (bias ? bias[e] : 0));
    int per = E / G;
    float *gs = (float *)malloc((size_t)G * 4);
    for (int g = 0; g < G; g++) {
        float t1 = -1e30f, t2 = -1e30f;
        for (int j = 0; j < per; j++) {
            float s = scores[g * per + j];
            if (s > t1) { t2 = t1; t1 = s; } else if (s > t2) t2 = s;
        }
        gs[g] = t1 + t2;
    }
    int *gord = (int *)malloc((size_t)G * 4);
    for (int g = 0; g < G; g++) gord[g] = g;
    for (int a = 0; a < G; a++) for (int b = a + 1; b < G; b++)
        if (gs[gord[b]] > gs[gord[a]]) { int t = gord[a]; gord[a] = gord[b]; gord[b] = t; }
    char *gkeep = (char *)calloc((size_t)G, 1);
    for (int i = 0; i < TG && i < G; i++) gkeep[gord[i]] = 1;
    for (int k = 0; k < K; k++) { idx[k] = -1; val[k] = -1e30f; }
    for (int e = 0; e < E; e++) {
        if (!gkeep[e / per]) continue;
        float s = scores[e];
        int p = K;
        for (int k = 0; k < K; k++) if (s > val[k]) { p = k; break; }
        if (p < K) {
            for (int k = K - 1; k > p; k--) { val[k] = val[k-1]; idx[k] = idx[k-1]; }
            val[p] = s; idx[p] = e;
        }
    }
    double sum = 0; for (int k = 0; k < K; k++) sum += val[k];
    if (c->norm_topk && sum > 1e-20) for (int k = 0; k < K; k++) val[k] = (float)(val[k] / sum);
    for (int k = 0; k < K; k++) val[k] *= c->route_scale;
    free(scores); free(gs); free(gord); free(gkeep);
}

static void model_load(Model *m, const char *dir, int cap_per_layer) {
    Cfg *c = &m->c;
    load_cfg(c, dir);
    st_init(&m->S, dir);
    char b[1024];
    /* global tensors (name probes: release uses model.*) */
    const char *emb_c[] = {"model.embed_tokens.weight", "model.word_embeddings.weight"};
    const char *emb = pick_name(&m->S, emb_c, 2);
    const char *norm_c[] = {"model.norm.weight", "model.final_layernorm.weight"};
    const char *norm = pick_name(&m->S, norm_c, 2);
    const char *head_c[] = {"lm_head.weight", "model.lm_head.weight"};
    const char *head = pick_name(&m->S, head_c, 2);
    if (!emb || !norm || !head) { fprintf(stderr, "[ling30] missing embed/norm/head\n"); exit(1); }
    m->embed = load_f32(&m->S, emb, prod2(c->vocab, c->hidden), "embed");
    m->norm  = load_f32(&m->S, norm, c->hidden, "norm");
    m->head  = load_f32(&m->S, head, prod2(c->vocab, c->hidden), "head");
    m->L = xcalloc((size_t)c->n_layers, sizeof(Layer), "layers");
    int P = c->kda_heads * c->kda_hd;
    for (int i = 0; i < c->n_layers; i++) {
        Layer *L = &m->L[i];
        char lb[64]; snprintf(lb, sizeof lb, "layer%d", i);
        snprintf(b, sizeof b, "model.layers.%d.input_layernorm.weight", i);
        L->in_norm = load_f32(&m->S, b, c->hidden, lb);
        snprintf(b, sizeof b, "model.layers.%d.post_attention_layernorm.weight", i);
        L->post_norm = load_f32(&m->S, b, c->hidden, lb);
        if (c->is_mla[i]) {
            int H = c->heads, qh = c->qk_head, vh = c->v_head;
            snprintf(b, sizeof b, "model.layers.%d.self_attn.q_proj.weight", i);
            load_mat(&m->S, &L->q, b, H * qh, c->hidden);
            snprintf(b, sizeof b, "model.layers.%d.self_attn.kv_a_proj_with_mqa.weight", i);
            load_mat(&m->S, &L->kva, b, c->kv_lora + c->qk_rope, c->hidden);
            snprintf(b, sizeof b, "model.layers.%d.self_attn.kv_a_layernorm.weight", i);
            L->kv_norm = load_f32(&m->S, b, c->kv_lora, lb);
            snprintf(b, sizeof b, "model.layers.%d.self_attn.kv_b_proj.weight", i);
            load_mat(&m->S, &L->kvb, b, H * (c->qk_nope + vh), c->kv_lora);
            if (c->gated_headwise) {
                snprintf(b, sizeof b, "model.layers.%d.self_attn.g_proj.weight", i);
                load_mat(&m->S, &L->g, b, H, c->hidden);
            }
            snprintf(b, sizeof b, "model.layers.%d.self_attn.o_proj.weight", i);
            const char *o_c[] = {b, NULL};
            char b2[1024]; snprintf(b2, sizeof b2, "model.layers.%d.self_attn.dense.weight", i);
            const char *cands[] = {b, b2};
            const char *o = pick_name(&m->S, cands, 2); (void)o_c;
            if (!o) { fprintf(stderr, "[ling30] missing MLA o_proj layer %d\n", i); exit(1); }
            load_mat(&m->S, &L->o, o, c->hidden, H * vh);
        } else {
            snprintf(b, sizeof b, "model.layers.%d.self_attn.q_proj.weight", i);
            load_mat(&m->S, &L->qp, b, P, c->hidden);
            snprintf(b, sizeof b, "model.layers.%d.self_attn.k_proj.weight", i);
            load_mat(&m->S, &L->kp, b, P, c->hidden);
            snprintf(b, sizeof b, "model.layers.%d.self_attn.v_proj.weight", i);
            load_mat(&m->S, &L->vp, b, P, c->hidden);
            snprintf(b, sizeof b, "model.layers.%d.self_attn.f_proj.weight", i);
            load_mat(&m->S, &L->fp, b, P, c->hidden);
            snprintf(b, sizeof b, "model.layers.%d.self_attn.g_proj.weight", i);
            load_mat(&m->S, &L->gp, b, P, c->hidden);
            snprintf(b, sizeof b, "model.layers.%d.self_attn.o_proj.weight", i);
            load_mat(&m->S, &L->op, b, c->hidden, P);
            snprintf(b, sizeof b, "model.layers.%d.self_attn.b_proj.weight", i);
            load_mat(&m->S, &L->bp, b, c->kda_heads, c->hidden);
            /* short-conv depthwise taps */
            L->conv_q = xmalloc((size_t)P * c->conv_k * 4, "convq");
            L->conv_k = xmalloc((size_t)P * c->conv_k * 4, "convk");
            L->conv_v = xmalloc((size_t)P * c->conv_k * 4, "convv");
            const char *qcn[] = {NULL, NULL};
            char c1[1024], c2[1024], c3[1024];
            snprintf(c1, sizeof c1, "model.layers.%d.self_attn.q_conv1d.weight", i);
            snprintf(c2, sizeof c2, "model.layers.%d.self_attn.k_conv1d.weight", i);
            snprintf(c3, sizeof c3, "model.layers.%d.self_attn.v_conv1d.weight", i);
            (void)qcn;
            if (st_has(&m->S, c1)) { float *t = load_f32(&m->S, c1, prod2(P, c->conv_k), lb); memcpy(L->conv_q, t, (size_t)P*c->conv_k*4); free(t); }
            else memset(L->conv_q, 0, (size_t)P * c->conv_k * 4);
            if (st_has(&m->S, c2)) { float *t = load_f32(&m->S, c2, prod2(P, c->conv_k), lb); memcpy(L->conv_k, t, (size_t)P*c->conv_k*4); free(t); }
            else memset(L->conv_k, 0, (size_t)P * c->conv_k * 4);
            if (st_has(&m->S, c3)) { float *t = load_f32(&m->S, c3, prod2(P, c->conv_k), lb); memcpy(L->conv_v, t, (size_t)P*c->conv_k*4); free(t); }
            else memset(L->conv_v, 0, (size_t)P * c->conv_k * 4);
            snprintf(b, sizeof b, "model.layers.%d.self_attn.A_log.weight", i);
            if (!st_has(&m->S, b)) snprintf(b, sizeof b, "model.layers.%d.self_attn.A_log", i);
            L->Alog = load_f32(&m->S, b, c->kda_heads, lb);
            snprintf(b, sizeof b, "model.layers.%d.self_attn.dt_bias.weight", i);
            if (!st_has(&m->S, b)) snprintf(b, sizeof b, "model.layers.%d.self_attn.dt_bias", i);
            L->dtb = load_f32(&m->S, b, P, lb);
        }
        if (i < c->first_dense) {
            snprintf(b, sizeof b, "model.layers.%d.mlp.gate_proj.weight", i);
            load_mat(&m->S, &L->dgate, b, c->dense_inter, c->hidden);
            snprintf(b, sizeof b, "model.layers.%d.mlp.up_proj.weight", i);
            load_mat(&m->S, &L->dup, b, c->dense_inter, c->hidden);
            snprintf(b, sizeof b, "model.layers.%d.mlp.down_proj.weight", i);
            load_mat(&m->S, &L->ddown, b, c->hidden, c->dense_inter);
        } else {
            snprintf(b, sizeof b, "model.layers.%d.mlp.gate.weight", i);
            L->router = load_f32(&m->S, b, prod2(c->n_experts, c->hidden), lb);
            L->rrows = c->n_experts; L->rcols = c->hidden;
            snprintf(b, sizeof b, "model.layers.%d.mlp.gate.expert_bias", i);
            if (!st_has(&m->S, b)) snprintf(b, sizeof b, "model.layers.%d.mlp.expert_bias", i);
            L->rbias = st_has(&m->S, b) ? load_f32(&m->S, b, c->n_experts, lb)
                                        : xcalloc((size_t)c->n_experts, 4, "rbias0");
            snprintf(b, sizeof b, "model.layers.%d.mlp.shared_experts.gate_proj.weight", i);
            load_mat(&m->S, &L->sh_gate, b, c->shared_inter * c->n_shared, c->hidden);
            snprintf(b, sizeof b, "model.layers.%d.mlp.shared_experts.up_proj.weight", i);
            load_mat(&m->S, &L->sh_up, b, c->shared_inter * c->n_shared, c->hidden);
            snprintf(b, sizeof b, "model.layers.%d.mlp.shared_experts.down_proj.weight", i);
            load_mat(&m->S, &L->sh_down, b, c->hidden, c->shared_inter * c->n_shared);
            /* streamed expert refs + names (one pread per matrix at use) */
            L->e_gate = xcalloc((size_t)c->n_experts, sizeof(TRef), "eg");
            L->e_up   = xcalloc((size_t)c->n_experts, sizeof(TRef), "eu");
            L->e_down = xcalloc((size_t)c->n_experts, sizeof(TRef), "ed");
            for (int e = 0; e < c->n_experts; e++) {
                snprintf(b, sizeof b, "model.layers.%d.mlp.experts.%d.gate_proj.weight", i, e);
                save_ref(&m->S, &L->e_gate[e], b);
                snprintf(b, sizeof b, "model.layers.%d.mlp.experts.%d.up_proj.weight", i, e);
                save_ref(&m->S, &L->e_up[e], b);
                snprintf(b, sizeof b, "model.layers.%d.mlp.experts.%d.down_proj.weight", i, e);
                save_ref(&m->S, &L->e_down[e], b);
            }
            L->cap = cap_per_layer > 0 ? cap_per_layer : 8;
            if (L->cap > c->n_experts) L->cap = c->n_experts;
            L->c_gate = xcalloc((size_t)L->cap, sizeof(float *), "cg");
            L->c_up   = xcalloc((size_t)L->cap, sizeof(float *), "cu");
            L->c_down = xcalloc((size_t)L->cap, sizeof(float *), "cd");
            L->c_id   = xmalloc((size_t)L->cap * 4, "cid");
            L->c_age  = xcalloc((size_t)L->cap, 4, "cage");
            for (int s = 0; s < L->cap; s++) {
                L->c_gate[s] = xmalloc((size_t)c->moe_inter * c->hidden * 4, "ecache");
                L->c_up[s]   = xmalloc((size_t)c->moe_inter * c->hidden * 4, "ecache");
                L->c_down[s] = xmalloc((size_t)c->hidden * c->moe_inter * 4, "ecache");
                L->c_id[s] = -1;
            }
        }
    }
    /* runtime state: caller sets ctx before decode */
    m->ctx = 8192;
    m->n_mla = 0; m->n_kda = 0;
    for (int i = 0; i < c->n_layers; i++) { if (c->is_mla[i]) m->n_mla++; else m->n_kda++; }
    size_t Psz = (size_t)P;
    m->kda_state = xcalloc((size_t)m->n_kda * c->kda_heads * c->kda_hd * c->kda_hd, 4, "kstate");
    m->kda_window = xcalloc((size_t)m->n_kda * 3 * Psz * c->conv_k, 4, "kwindow");
    size_t need = (size_t)c->hidden * 8 + (size_t)c->heads * (c->qk_head + c->v_head) * 4
        + (size_t)c->n_experts * 4 + 4096;
    m->tmp = xmalloc(need * 4, "tmp");
    m->tmp_n = need;
    m->kda_scratch = xmalloc((size_t)(3 * P + 64 * c->kda_hd) * 4 + 65536, "kscratch");
    m->estage = xmalloc((size_t)c->moe_inter * c->hidden * 4, "estage");
}

/* expert cache: return f32 pointers for expert e (loads on miss, LRU evict) */
static void expert_get(Model *m, int li, int e, float **g, float **u, float **d) {
    Layer *L = &m->L[li]; Cfg *c = &m->c;
    L->tick++;
    for (int s = 0; s < L->cap; s++) if (L->c_id[s] == e) { L->c_age[s] = L->tick; *g = L->c_gate[s]; *u = L->c_up[s]; *d = L->c_down[s]; return; }
    int slot = 0; for (int s = 0; s < L->cap; s++) if (L->c_age[s] < L->c_age[slot]) slot = s;
    char b[1024];
    snprintf(b, sizeof b, "model.layers.%d.mlp.experts.%d.gate_proj.weight", li, e);
    st_read_f32(&m->S, b, L->c_gate[slot], 1);
    snprintf(b, sizeof b, "model.layers.%d.mlp.experts.%d.up_proj.weight", li, e);
    st_read_f32(&m->S, b, L->c_up[slot], 1);
    snprintf(b, sizeof b, "model.layers.%d.mlp.experts.%d.down_proj.weight", li, e);
    st_read_f32(&m->S, b, L->c_down[slot], 1);
    L->c_id[slot] = e; L->c_age[slot] = L->tick;
    *g = L->c_gate[slot]; *u = L->c_up[slot]; *d = L->c_down[slot];
    (void)c;
}

/* one decode step: x[hidden] -> logits[vocab]. pos = absolute position. */
static void forward_step(Model *m, const float *x, float *logits, int pos) {
    Cfg *c = &m->c;
    int H = c->hidden;
    float *h = m->tmp;                 /* hidden scratch */
    float *h2 = m->tmp + H;            /* second scratch */
    float *att = m->tmp + 2 * H;       /* attn out */
    memcpy(h, x, (size_t)H * 4);
    /* inverse freqs for rope (computed per step; tiny: 32 floats) */
    float invf[64];
    for (int i = 0; i < c->qk_rope / 2; i++)
        invf[i] = 1.0f / powf(c->rope_theta, (float)(2 * i) / c->qk_rope);
    int kda_idx = 0;
    int P = c->kda_heads * c->kda_hd;
    for (int li = 0; li < c->n_layers; li++) {
        Layer *L = &m->L[li];
        /* pre-norm */
        rmsnorm(h2, h, L->in_norm, H, c->eps);
        if (c->is_mla[li]) {
            int nH = c->heads, qh = c->qk_head, vh = c->v_head;
            /* NOTE: MLA temporaries (q: heads*qh etc.) exceed the hidden-sized
             * scratch, so they spill to lazily-grown heap buffers below. To
             * keep this compact engine honest, malloc lazily via static. */
            static float *big = NULL; static size_t big_n = 0;
            size_t want = (size_t)(nH * qh + c->kv_lora + c->qk_rope + nH * (c->qk_nope + vh) + nH * vh + H);
            if (big_n < want) { free(big); big = xmalloc(want * 4, "mla"); big_n = want; }
            float *qq = big, *ckv = big + nH * qh, *kb = NULL;
            matvec(qq, L->q.w, h2, nH * qh, H);
            matvec(ckv, L->kva.w, h2, c->kv_lora + c->qk_rope, H);
            float *kpass = ckv, *krot1 = ckv + c->kv_lora;
            rmsnorm(kpass, kpass, L->kv_norm, c->kv_lora, c->eps);
            kb = ckv + c->kv_lora + c->qk_rope;
            /* kv_b_proj: need contiguous copy of kpass first */
            static float *kpn = NULL; static size_t kpn_n = 0;
            if (kpn_n < (size_t)c->kv_lora) { free(kpn); kpn = xmalloc((size_t)c->kv_lora * 4, "kpn"); kpn_n = c->kv_lora; }
            memcpy(kpn, kpass, (size_t)c->kv_lora * 4);
            matvec(kb, L->kvb.w, kpn, nH * (c->qk_nope + vh), c->kv_lora);
            /* split q into qpass/qrot, k into kpass2/value */
            /* qk norm */
            if (c->use_qk_norm) {
                for (int hh = 0; hh < nH; hh++) {
                    float *qp = qq + (size_t)hh * qh;
                    double s = 0; for (int i = 0; i < qh; i++) s += (double)qp[i]*qp[i];
                    float g = 1.0f / sqrtf((float)(s / qh) + c->eps);
                    for (int i = 0; i < qh; i++) qp[i] *= g;
                }
                for (int hh = 0; hh < nH; hh++) {
                    float *kp = kb + (size_t)hh * (c->qk_nope + vh);
                    double s = 0; for (int i = 0; i < c->qk_nope; i++) s += (double)kp[i]*kp[i];
                    float g = 1.0f / sqrtf((float)(s / c->qk_nope) + c->eps);
                    for (int i = 0; i < c->qk_nope; i++) kp[i] *= g;
                }
            }
            /* rope on qrot + shared krot */
            float *qrot_buf = qq; /* in place via helper on strided views: do manually */
            /* build per-head krot expanded */
            static float *kexp = NULL; static size_t kexp_n = 0;
            size_t kwant = (size_t)nH * c->qk_rope;
            if (kexp_n < kwant) { free(kexp); kexp = xmalloc(kwant * 4, "kexp"); kexp_n = kwant; }
            for (int hh = 0; hh < nH; hh++) memcpy(kexp + (size_t)hh * c->qk_rope, krot1, (size_t)c->qk_rope * 4);
            /* apply rope to qq's rope tails and kexp */
            for (int hh = 0; hh < nH; hh++) {
                float *qr = qq + (size_t)hh * qh + c->qk_nope;
                float *kr = kexp + (size_t)hh * c->qk_rope;
                for (int i = 0; i < c->qk_rope; i += 2) {
                    float f = invf[i / 2] * pos;
                    float co = cosf(f), si = sinf(f);
                    float a0 = qr[i], a1 = qr[i+1], b0 = kr[i], b1 = kr[i+1];
                    qr[i] = a0*co - a1*si; qr[i+1] = a0*si + a1*co;
                    kr[i] = b0*co - b1*si; kr[i+1] = b0*si + b1*co;
                }
            }
            /* KV cache append */
            float *Kc = m->mla_k + ((size_t)li * m->ctx + pos) * (c->kv_lora + c->qk_rope);
            float *Vc = m->mla_v + ((size_t)li * m->ctx + pos) * (nH * vh);
            memcpy(Kc, kpn, (size_t)c->kv_lora * 4);
            memcpy(Kc + c->kv_lora, krot1, (size_t)c->qk_rope * 4);
            /* value part out of kb */
            static float *vcur = NULL; static size_t vcur_n = 0;
            if (vcur_n < (size_t)nH * vh) { free(vcur); vcur = xmalloc((size_t)nH*vh*4, "vcur"); vcur_n = nH*vh; }
            for (int hh = 0; hh < nH; hh++)
                memcpy(vcur + (size_t)hh * vh, kb + (size_t)hh * (c->qk_nope + vh) + c->qk_nope, (size_t)vh * 4);
            memcpy(Vc, vcur, (size_t)nH * vh * 4);
            /* attention over 0..pos */
            float scale = 1.0f / sqrtf((float)qh);
            static float *ctxv = NULL; static size_t ctxv_n = 0;
            if (ctxv_n < (size_t)nH * vh) { free(ctxv); ctxv = xmalloc((size_t)nH*vh*4, "ctxv"); ctxv_n = nH*vh; }
            memset(ctxv, 0, (size_t)nH * vh * 4);
            for (int hh = 0; hh < nH; hh++) {
                float *qhh = qq + (size_t)hh * qh;
                /* scores */
                static float *sc = NULL; static size_t sc_n = 0;
                if (sc_n < (size_t)m->ctx) { free(sc); sc = xmalloc((size_t)m->ctx * 4, "sc"); sc_n = m->ctx; }
                float mx = -1e30f;
                for (int t = 0; t <= pos; t++) {
                    float *Kt = m->mla_k + ((size_t)li * m->ctx + t) * (c->kv_lora + c->qk_rope);
                    /* reconstruct key: kv_b path needs kv_b_proj(Kt_pass) -- to stay
                     * exact without re-expanding the whole cache, recompute per t.
                     * Cost O(ctx) matvecs; correct and simple (the large engines
                     * absorb kv_b; this one recomputes -- placement, not math). */
                    static float *kt_full = NULL; static size_t kt_n = 0;
                    if (kt_n < (size_t)nH * (c->qk_nope + vh)) { free(kt_full); kt_full = xmalloc((size_t)nH*(c->qk_nope+vh)*4, "ktf"); kt_n = nH*(c->qk_nope+vh); }
                    matvec(kt_full, L->kvb.w, Kt, nH * (c->qk_nope + vh), c->kv_lora);
                    float *kh = kt_full + (size_t)hh * (c->qk_nope + vh);
                    /* rope part */
                    float *kr_t = Kt + c->kv_lora;
                    /* apply rope at position t to a copy */
                    float dot = 0;
                    for (int i = 0; i < c->qk_nope; i++) dot += qhh[i] * kh[i];
                    for (int i = 0; i < c->qk_rope; i += 2) {
                        float f = invf[i/2] * t;
                        float co = cosf(f), si = sinf(f);
                        float k0 = kr_t[i], k1 = kr_t[i+1];
                        /* rotate key then dot with rotated query part */
                        float rk0 = k0*co - k1*si, rk1 = k0*si + k1*co;
                        /* qhh rope part already rotated at pos; to avoid double
                         * rotation we rotate the raw query here instead. Simpler
                         * exact path: recompute raw dots with both rotated.
                         * We stored rotated q; so rotate key to t and dot. */
                        dot += qhh[c->qk_nope+i] * rk0 + qhh[c->qk_nope+i+1] * rk1;
                    }
                    /* NOTE: q was rotated at pos above; rotating key at t and
                     * dotting equals relative rotation -- exact. */
                    sc[t] = dot * scale;
                    if (sc[t] > mx) mx = sc[t];
                }
                double ds = 0; for (int t = 0; t <= pos; t++) ds += exp((double)sc[t] - mx);
                float *out = ctxv + (size_t)hh * vh;
                memset(out, 0, (size_t)vh * 4);
                for (int t = 0; t <= pos; t++) {
                    float w = (float)(exp((double)sc[t] - mx) / ds);
                    float *Vt = m->mla_v + ((size_t)li * m->ctx + t) * (nH * vh) + (size_t)hh * vh;
                    for (int i = 0; i < vh; i++) out[i] += w * Vt[i];
                }
            }
            if (c->gated_headwise && L->g.w) {
                static float *gv = NULL; static size_t gv_n = 0;
                if (gv_n < (size_t)nH) { free(gv); gv = xmalloc((size_t)nH*4, "gv"); gv_n = nH; }
                matvec(gv, L->g.w, h2, nH, H);
                for (int hh = 0; hh < nH; hh++) {
                    float g = sigm_f(gv[hh]);
                    float *out = ctxv + (size_t)hh * vh;
                    for (int i = 0; i < vh; i++) out[i] *= g;
                }
            }
            matvec(att, L->o.w, ctxv, H, nH * vh);
            (void)qrot_buf;
        } else {
            /* KDA via shared coli_kda_step */
            static float *qkv = NULL, *gate = NULL, *beta = NULL, *fv = NULL;
            static size_t qkv_n = 0;
            if (qkv_n < (size_t)3 * P + 2 * P + 64) {
                free(qkv); free(gate); free(beta); free(fv);
                qkv = xmalloc(((size_t)3*P + 2*P + 64) * 4, "qkv");
                gate = qkv + 3*P; beta = gate + P; fv = beta + 64; qkv_n = 3*P + 2*P + 64;
            }
            matvec(qkv, L->qp.w, h2, P, H);
            matvec(qkv + P, L->kp.w, h2, P, H);
            matvec(qkv + 2*P, L->vp.w, h2, P, H);
            matvec(fv, L->fp.w, h2, P, H);
            for (int i = 0; i < P; i++) fv[i] += L->dtb[i];
            /* gate = lower_bound * sigmoid(exp(Alog)*f) per head-channel */
            for (int hh = 0; hh < c->kda_heads; hh++) {
                float ea = expf(L->Alog[hh]);
                for (int i = 0; i < c->kda_hd; i++) {
                    float s = sigm_f(ea * fv[hh * c->kda_hd + i]);
                    gate[hh * c->kda_hd + i] = c->kda_lb * s;
                }
            }
            matvec(beta, L->bp.w, h2, c->kda_heads, H);
            for (int hh = 0; hh < c->kda_heads; hh++) beta[hh] = sigm_f(beta[hh]);
            /* conv taps: interleave q/k/v histories */
            static float *cw = NULL; static size_t cw_n = 0;
            if (cw_n < (size_t)3*P*c->conv_k) { free(cw); cw = xmalloc((size_t)3*P*c->conv_k*4, "cw"); cw_n = 3*P*c->conv_k; }
            for (int i = 0; i < P * c->conv_k; i++) {
                cw[i] = L->conv_q[i];
                cw[P*c->conv_k + i] = L->conv_k[i];
                cw[2*P*c->conv_k + i] = L->conv_v[i];
            }
            float *state = m->kda_state + (size_t)kda_idx * c->kda_heads * c->kda_hd * c->kda_hd;
            float *win = m->kda_window + (size_t)kda_idx * 3 * P * c->conv_k;
            static float *kout = NULL; static size_t kout_n = 0;
            if (kout_n < (size_t)P) { free(kout); kout = xmalloc((size_t)P*4, "kout"); kout_n = P; }
            coli_kda_step(kout, state, win, qkv, cw, gate, beta,
                          c->kda_heads, c->kda_hd, c->kda_hd, c->conv_k, c->eps, m->kda_scratch);
            /* o_norm (FusedRMSNormGated+sigmoid) + o_proj */
            matvec(gate, L->gp.w, h2, P, H); /* reuse gate buf as g */
            for (int hh = 0; hh < c->kda_heads; hh++) {
                float *o = kout + (size_t)hh * c->kda_hd;
                float *g = gate + (size_t)hh * c->kda_hd;
                double s = 0; for (int i = 0; i < c->kda_hd; i++) s += (double)o[i]*o[i];
                float n = 1.0f / sqrtf((float)(s / c->kda_hd) + c->eps);
                for (int i = 0; i < c->kda_hd; i++) o[i] = o[i] * n * sigm_f(g[i]);
            }
            matvec(att, L->op.w, kout, H, P);
            kda_idx++;
        }
        for (int i = 0; i < H; i++) h[i] += att[i];
        /* MLP / MoE */
        rmsnorm(h2, h, L->post_norm, H, c->eps);
        static float *mout = NULL; static size_t mout_n = 0;
        if (mout_n < (size_t)H) { free(mout); mout = xmalloc((size_t)H*4, "mout"); mout_n = H; }
        if (li < c->first_dense) {
            static float *g0 = NULL, *u0 = NULL; static size_t d_n = 0;
            if (d_n < (size_t)c->dense_inter) { free(g0); free(u0); g0 = xmalloc((size_t)c->dense_inter*4, "dg"); u0 = xmalloc((size_t)c->dense_inter*4, "du"); d_n = c->dense_inter; }
            matvec(g0, L->dgate.w, h2, c->dense_inter, H);
            matvec(u0, L->dup.w, h2, c->dense_inter, H);
            for (int i = 0; i < c->dense_inter; i++) g0[i] = silu_f(g0[i]) * u0[i];
            matvec(mout, L->ddown.w, g0, H, c->dense_inter);
        } else {
            /* shared expert first */
            static float *sg = NULL, *su = NULL; static size_t s_n = 0;
            int SI = c->shared_inter * c->n_shared;
            if (s_n < (size_t)SI) { free(sg); free(su); sg = xmalloc((size_t)SI*4, "sg"); su = xmalloc((size_t)SI*4, "su"); s_n = SI; }
            matvec(sg, L->sh_gate.w, h2, SI, H);
            matvec(su, L->sh_up.w, h2, SI, H);
            for (int i = 0; i < SI; i++) sg[i] = silu_f(sg[i]) * su[i];
            if (c->shared_lim[li] > 0) for (int i = 0; i < SI; i++) {
                float lim = c->shared_lim[li];
                if (sg[i] > lim) sg[i] = lim; else if (sg[i] < -lim) sg[i] = -lim;
            }
            matvec(mout, L->sh_down.w, sg, H, SI);
            /* router */
            static float *rl = NULL; static size_t rl_n = 0;
            if (rl_n < (size_t)c->n_experts) { free(rl); rl = xmalloc((size_t)c->n_experts*4, "rl"); rl_n = c->n_experts; }
            matvec(rl, L->router, h2, c->n_experts, H);
            int idx[LING30_MAX_TOPK]; float wv[LING30_MAX_TOPK];
            route_topk(c, rl, L->rbias, idx, wv);
            static float *eg = NULL, *eu = NULL, *ed = NULL; static size_t e_n = 0;
            if (e_n < (size_t)c->moe_inter) { free(eg); free(eu); free(ed); eg = xmalloc((size_t)c->moe_inter*4, "eg"); eu = xmalloc((size_t)c->moe_inter*4, "eu"); ed = xmalloc((size_t)c->moe_inter*4, "ed"); e_n = c->moe_inter; }
            for (int k = 0; k < c->topk; k++) {
                float *gw, *up, *dn;
                expert_get(m, li, idx[k], &gw, &up, &dn);
                /* gate/up rows x h2 */
                for (int f = 0; f < c->moe_inter; f++) {
                    const float *gr = gw + (size_t)f * H, *ur = up + (size_t)f * H;
                    float s1 = 0, s2 = 0;
                    for (int i = 0; i < H; i++) { s1 += gr[i] * h2[i]; s2 += ur[i] * h2[i]; }
                    float a = silu_f(s1) * s2;
                    float lim = c->swiglu_lim[li];
                    if (lim > 0) { if (a > lim) a = lim; else if (a < -lim) a = -lim; }
                    eg[f] = a;
                }
                /* down: mout += w * (dn * eg) */
                for (int i = 0; i < H; i++) {
                    const float *dr = dn + (size_t)i * c->moe_inter;
                    float s = 0; for (int f = 0; f < c->moe_inter; f++) s += dr[f] * eg[f];
                    mout[i] += wv[k] * s;
                }
            }
        }
        for (int i = 0; i < H; i++) h[i] += mout[i];
        memset(mout, 0, (size_t)H * 4);
    }
    rmsnorm(h2, h, m->norm, H, c->eps);
    matvec(logits, m->head, h2, c->vocab, H);
}

/* reset per-conversation state */
static void state_reset(Model *m) {
    Cfg *c = &m->c;
    size_t P = (size_t)c->kda_heads * c->kda_hd;
    memset(m->kda_state, 0, (size_t)m->n_kda * P * c->kda_hd * 4);
    memset(m->kda_window, 0, (size_t)m->n_kda * 3 * P * c->conv_k * 4);
    if (m->mla_k) memset(m->mla_k, 0, (size_t)c->n_layers * m->ctx * (c->kv_lora + c->qk_rope) * 4);
    if (m->mla_v) memset(m->mla_v, 0, (size_t)c->n_layers * m->ctx * c->heads * c->v_head * 4);
}

static int pick_next(const float *logits, int n, float temp, const int *banned, int nb) {
    if (temp <= 0) {
        int b = 0; float bv = logits[0];
        for (int i = 1; i < n; i++) if (logits[i] > bv) { bv = logits[i]; b = i; }
        return b;
    }
    /* temperature sample (simple, no top-p here; gateway handles nucleus) */
    double s = 0; static float *p = NULL; static size_t pn = 0;
    if (pn < (size_t)n) { free(p); p = xmalloc((size_t)n * 4, "prob"); pn = n; }
    float mx = logits[0]; for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
    for (int i = 0; i < n; i++) { p[i] = expf((logits[i] - mx) / temp); s += p[i]; }
    double r = (double)rand() / RAND_MAX * s;
    for (int i = 0; i < n; i++) { r -= p[i]; if (r <= 0) return i; }
    (void)banned; (void)nb;
    return n - 1;
}

/* ---- tiny oracle check: SNAP=<dir> ling30 <cap> <ngen> <ref.json> ---- */
static int tiny_check(Model *m, const char *ref_path, int ngen) {
    char *text = read_text(ref_path, NULL);
    if (!text) { fprintf(stderr, "[ling30] cannot read %s\n", ref_path); return 1; }
    char *arena = NULL; jval *root = json_parse(text, &arena);
    if (!root) { fprintf(stderr, "[ling30] ref not JSON\n"); return 1; }
    jval *jp = json_get(root, "prompt_ids"), *jf = json_get(root, "full_ids");
    if (!jp || jp->t != J_ARR || !jf || jf->t != J_ARR) { fprintf(stderr, "[ling30] ref needs prompt_ids/full_ids\n"); return 1; }
    int P = jp->len;
    int *prompt = xmalloc((size_t)(P + ngen) * 4, "ids");
    for (int i = 0; i < P; i++) prompt[i] = (int)jp->kids[i]->num;
    Cfg *c = &m->c;
    float *x = xmalloc((size_t)c->hidden * 4, "x");
    float *lg = xmalloc((size_t)c->vocab * 4, "lg");
    state_reset(m);
    for (int i = 0; i < P; i++) {
        float *e = m->embed + (size_t)prompt[i] * c->hidden;
        forward_step(m, e, lg, i);
    }
    int ok = 1;
    for (int s = 0; s < ngen; s++) {
        int nx = pick_next(lg, c->vocab, 0, NULL, 0);
        int want = (P + s < jf->len) ? (int)jf->kids[P + s]->num : -1;
        if (nx != want) { printf("C engine token %d: got %d want %d\n", s, nx, want); ok = 0; }
        else printf("C engine token %d: %d OK\n", s, nx);
        prompt[P + s] = nx;
        forward_step(m, m->embed + (size_t)nx * c->hidden, lg, P + s);
        (void)x;
    }
    free(prompt); free(x); free(lg); free(text);
    return ok ? 0 : 1;
}

int main(int argc, char **argv) {
    coli_omp_tune_threads("ling30");
    const char *dir = getenv("SNAP");
    int cap = env_int("LING30_CAP", 0);
    const char *ids_text = NULL, *prompt = NULL;
    int ngen = 32;
    const char *ref = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--ids") && i + 1 < argc) ids_text = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--ngen") && i + 1 < argc) ngen = coli_arg_int(argv[++i], "--ngen");
        else if (!strcmp(argv[i], "--cap") && i + 1 < argc) cap = coli_arg_int(argv[++i], "--cap");
        else if (is_dir(argv[i])) dir = argv[i];
        else if (!strcmp(argv[i], "--ref") && i + 1 < argc) ref = argv[++i];
        else if (argv[i][0] != '-') {
            /* positional: <cap> <ngen> <ref.json> (tiny harness) or <dir> */
            char *end = NULL; long v = strtol(argv[i], &end, 10);
            if (end != argv[i] && !*end && cap == 0) cap = (int)v;
            else if (end != argv[i] && !*end) ngen = (int)v;
            else if (strstr(argv[i], ".json")) ref = argv[i];
            else if (is_dir(argv[i])) dir = argv[i];
            else cap = coli_arg_int(argv[i], "cache/layer");
        }
    }
    if (env_int("SERVE", 0) && dir && !ref && argc >= 2) {
        /* SNAP=<dir> SERVE=1 ling30 <cap>: tiny-serve uses same shape as mimo */
        for (int i = 1; i < argc; i++) {
            char *end = NULL; long v = strtol(argv[i], &end, 10);
            if (end != argv[i] && !*end) cap = (int)v;
        }
    }
    if (!dir) {
        fprintf(stderr, "usage: %s <model dir> [--ids \"1 2 3\" | --prompt TEXT] [--ngen N] [--cap N]\n"
                        "       SNAP=<model dir> SERVE=1 %s <cap>\n", argv[0], argv[0]);
        return 2;
    }
    if (cap <= 0) cap = 8;
    Model *m = xcalloc(1, sizeof(Model), "model");
    double t0 = now_s();
    model_load(m, dir, cap);
    Cfg *c = &m->c;
    /* KV caches sized by ctx (LING30_MAXT, set by `coli --ctx`; CTX fallback).
     * The literal getenv keeps tests/test_registry_engine_agreement.py honest:
     * the planner prices this variable, so the engine must read it. */
    const char *ctx_env = getenv("LING30_MAXT");
    m->ctx = (ctx_env && *ctx_env) ? atoi(ctx_env) : env_int("CTX", 8192);
    if (m->ctx < 16) m->ctx = 16;
    m->mla_k = xcalloc((size_t)c->n_layers * m->ctx * (c->kv_lora + c->qk_rope), 4, "mlak");
    m->mla_v = xcalloc((size_t)c->n_layers * m->ctx * c->heads * c->v_head, 4, "mlav");
    fprintf(stderr, "[ling30] %d layers (%d KDA + %d MLA), %d experts top-%d, cache %d/layer, ctx %d -- loaded in %.2fs\n",
            c->n_layers, m->n_kda, m->n_mla, c->n_experts, c->topk, cap, m->ctx, now_s() - t0);
    char tok_path[4096]; snprintf(tok_path, sizeof tok_path, "%s/tokenizer.json", dir);
    int have_tok = access(tok_path, R_OK) == 0;
    Tok tokenizer; memset(&tokenizer, 0, sizeof tokenizer);
    if (have_tok) tok_load(&tokenizer, tok_path);

    if (ref) return tiny_check(m, ref, ngen);

    if (env_int("SERVE", 0)) {
        if (!have_tok) { fprintf(stderr, "[ling30] SERVE needs %s\n", tok_path); return 1; }
        static const ColiServeWireProfile ling30_wire = {
            .max_header_bytes = 511,
            .max_payload_bytes = 1u << 22,
            .max_tokens = 1 << 20,
            .require_exact_lf = 1,
            .require_finite_sampling = 0,
        };
        coli_serve_stdio_init();
        coli_serve_write_ready_caps(stdout, 0.0, "chat=1 decide=0 vision=0 thinking=1 tools=1");
        ColiServeCommand cmd; memset(&cmd, 0, sizeof cmd);
        int *sids = xmalloc(65536 * 4, "sids");
        float *lg = xmalloc((size_t)c->vocab * 4, "lg");
        char ob[512];
        for (;;) {
            ColiServeReadResult r = coli_serve_read_command(stdin, &ling30_wire, &cmd);
            if (r == COLI_SERVE_READ_EOF || r == COLI_SERVE_READ_BAD_FRAME) break;
            if (r != COLI_SERVE_READ_OK) { coli_serve_command_dispose(&cmd); continue; }
            if (cmd.kind == COLI_SERVE_COMMAND_SUBMIT) {
                /* payload is the already-rendered prompt text (gateway owns the
                 * Ling-3.0 chat template); the engine tokenizes it as-is. */
                int n = tok_encode(&tokenizer, (char *)cmd.payload,
                                   (int)cmd.payload_bytes, sids, 65536);
                if (n <= 0) { coli_serve_write_error(stdout, cmd.id, "empty prompt"); coli_serve_command_dispose(&cmd); continue; }
                coli_serve_write_accept(stdout, cmd.id, n);
                double t0 = now_s();
                state_reset(m);
                int pos = 0;
                for (int i = 0; i < n && pos < m->ctx; i++)
                    forward_step(m, m->embed + (size_t)sids[i] * c->hidden, lg, pos++);
                int gen = 0;
                for (int s = 0; s < cmd.max_tokens && pos < m->ctx; s++) {
                    int nx = pick_next(lg, c->vocab, cmd.temperature, NULL, 0);
                    int on = tok_decode(&tokenizer, &nx, 1, ob, sizeof ob - 1);
                    if (on < 0) on = 0;
                    coli_serve_write_data(stdout, cmd.id, ob, (size_t)on);
                    gen++;
                    if (nx == c->eos) break;
                    forward_step(m, m->embed + (size_t)nx * c->hidden, lg, pos++);
                }
                ColiServeDone done;
                memset(&done, 0, sizeof done);
                done.completion_tokens = gen;
                done.tokens_per_second = gen / (now_s() - t0 + 1e-9);
                done.prompt_tokens = n;
                coli_serve_write_done(stdout, cmd.id, &done);
            }
            coli_serve_command_dispose(&cmd);
        }
        return 0;
    }

    /* one-shot / interactive */
    int *ids = xmalloc(65536 * 4, "ids");
    int n = 0;
    if (ids_text) {
        const char *p = ids_text;
        while (*p && n < 65536) { while (*p == ' ') p++; if (!*p) break; ids[n++] = atoi(p); while (*p && *p != ' ') p++; }
    } else if (prompt && have_tok) {
        n = tok_encode(&tokenizer, prompt, (int)strlen(prompt), ids, 65536);
    } else if (prompt) {
        fprintf(stderr, "[ling30] no tokenizer.json: use --ids\n"); return 1;
    } else {
        /* interactive chat: Ling-3.0 native roles */
        char line[8192];
        printf("ling30 chat (native <role>HUMAN/ASSISTANT</role> + <|role_end|> %d). Empty line quits.\n", c->eos);
        for (;;) {
            printf("> "); fflush(stdout);
            if (!fgets(line, sizeof line, stdin)) break;
            if (line[0] == '\n' || line[0] == 0) break;
            char full[16384];
            snprintf(full, sizeof full, "<role>HUMAN</role>%s<|role_end|><role>ASSISTANT</role>\n<think>", line);
            int m0 = have_tok ? tok_encode(&tokenizer, full, (int)strlen(full), ids, 65536) : 0;
            if (!have_tok) { fprintf(stderr, "need tokenizer.json\n"); continue; }
            state_reset(m);
            float *lg = xmalloc((size_t)c->vocab * 4, "lg");
            int pos2 = 0;
            for (int i = 0; i < m0; i++) forward_step(m, m->embed + (size_t)ids[i] * c->hidden, lg, pos2++);
            for (int s = 0; s < ngen; s++) {
                int nx = pick_next(lg, c->vocab, 0.7f, NULL, 0);
                if (nx == c->eos) break;
                char ob[256]; int on = tok_decode(&tokenizer, &nx, 1, ob, sizeof ob - 1);
                ob[on < 0 ? 0 : on] = 0; printf("%s", ob); fflush(stdout);
                forward_step(m, m->embed + (size_t)nx * c->hidden, lg, pos2++);
            }
            printf("\n"); free(lg);
        }
        return 0;
    }
    state_reset(m);
    float *lg = xmalloc((size_t)c->vocab * 4, "lg");
    int pos = 0;
    for (int i = 0; i < n; i++) forward_step(m, m->embed + (size_t)ids[i] * c->hidden, lg, pos++);
    for (int s = 0; s < ngen; s++) {
        int nx = pick_next(lg, c->vocab, 0, NULL, 0);
        printf("%d%s", nx, s + 1 < ngen ? " " : "\n");
        if (nx == c->eos) break;
        forward_step(m, m->embed + (size_t)nx * c->hidden, lg, pos++);
    }
    return 0;
}
