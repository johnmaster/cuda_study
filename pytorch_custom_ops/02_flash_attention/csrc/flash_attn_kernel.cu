/**
 * Flash Attention — PyTorch Custom Op (Forward + Backward)
 *
 * Forward:  tile-based online-softmax attention, saves logsumexp L for backward
 * Backward: tile-based recomputation (never materializes the full N×N attention matrix)
 *
 * Grid strategy: one warp (32 threads) per query/key row
 * Each thread owns a slice of the head dimension d, accumulates results in registers
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <vector>

constexpr int TILE = 32;

// ============================================================================
// Warp-level primitives
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    return __shfl_sync(0xffffffff, v, 0);
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
    return __shfl_sync(0xffffffff, v, 0);
}

// ============================================================================
// Forward kernel — one block per query row
// Grid(N, B)  Block(32)
// ============================================================================

__global__ void flash_attn_fwd_kernel(
    const float* __restrict__ Q,   // [B, N, d]
    const float* __restrict__ K,
    const float* __restrict__ V,
    float*       __restrict__ O,
    float*       __restrict__ L,   // [B, N] logsumexp per row
    int N, int d)
{
    const int row   = blockIdx.x;
    const int batch = blockIdx.y;
    const int tx    = threadIdx.x;          // 0..31
    if (row >= N) return;

    const float scale = rsqrtf((float)d);

    const float* q_row  = Q + (batch * N + row) * d;
    const float* K_base = K + batch * N * d;
    const float* V_base = V + batch * N * d;

    // shared: sK[TILE*d] + sV[TILE*d] + sq[d]
    extern __shared__ float smem[];
    float* sK = smem;
    float* sV = sK + TILE * d;
    float* sq = sV + TILE * d;

    for (int i = tx; i < d; i += 32) sq[i] = q_row[i];
    __syncthreads();

    // per-thread output slice
    const int dpt     = (d + 31) / 32;   // dims per thread
    const int d_start = tx * dpt;
    const int d_end   = min(d_start + dpt, d);

    float o_acc[8] = {0.f};
    float m_i = -INFINITY;
    float l_i = 0.f;

    const int num_tiles = (N + TILE - 1) / TILE;
    for (int t = 0; t < num_tiles; ++t) {
        const int base = t * TILE;
        const int len  = min(TILE, N - base);

        // cooperative load of K and V tiles
        for (int i = tx; i < len * d; i += 32) {
            int r = i / d, c = i % d;
            sK[r * d + c] = K_base[(base + r) * d + c];
            sV[r * d + c] = V_base[(base + r) * d + c];
        }
        __syncthreads();

        for (int j = 0; j < len; ++j) {
            // score = q · k_j  (warp-wide reduce)
            float s = 0.f;
            for (int k = tx; k < d; k += 32) s += sq[k] * sK[j * d + k];
            s = warp_reduce_sum(s) * scale;
            s = __shfl_sync(0xffffffff, s, 0);

            // online softmax update
            float m_new = fmaxf(m_i, s);
            float p     = expf(s   - m_new);
            float alpha = expf(m_i - m_new);

            for (int k = d_start; k < d_end; ++k)
                o_acc[k - d_start] = o_acc[k - d_start] * alpha + p * sV[j * d + k];

            l_i = l_i * alpha + p;
            m_i = m_new;
        }
        __syncthreads();
    }

    // normalise & write O
    float inv_l = 1.f / l_i;
    float* o_row = O + (batch * N + row) * d;
    for (int k = d_start; k < d_end; ++k)
        o_row[k] = o_acc[k - d_start] * inv_l;

    // save logsumexp = m + log(l) for backward
    if (tx == 0)
        L[batch * N + row] = m_i + logf(l_i);
}

// ============================================================================
// Backward helper: D[b,i] = dot(dO[b,i,:], O[b,i,:])
// Grid(N, B) Block(32)
// ============================================================================

__global__ void flash_attn_bwd_D_kernel(
    const float* __restrict__ dO,
    const float* __restrict__ O,
    float*       __restrict__ D,
    int N, int d)
{
    const int row   = blockIdx.x;
    const int batch = blockIdx.y;
    if (row >= N) return;

    const float* do_r = dO + (batch * N + row) * d;
    const float* o_r  = O  + (batch * N + row) * d;

    float v = 0.f;
    for (int k = threadIdx.x; k < d; k += 32) v += do_r[k] * o_r[k];
    v = warp_reduce_sum(v);

    if (threadIdx.x == 0) D[batch * N + row] = v;
}

// ============================================================================
// Backward: dQ — one block per query row i, iterate over K/V tiles
// Grid(N, B)  Block(32)
// ============================================================================

__global__ void flash_attn_bwd_dq_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ dO,
    const float* __restrict__ Lse,   // logsumexp [B,N]
    const float* __restrict__ D,     // [B,N]
    float*       __restrict__ dQ,
    int N, int d)
{
    const int row   = blockIdx.x;
    const int batch = blockIdx.y;
    const int tx    = threadIdx.x;
    if (row >= N) return;

    const float scale = rsqrtf((float)d);
    const float lse_i = Lse[batch * N + row];
    const float d_i   = D  [batch * N + row];

    extern __shared__ float smem[];
    float* sq   = smem;                 // [d]
    float* s_do = sq   + d;             // [d]
    float* sK   = s_do + d;             // [TILE*d]
    float* sV   = sK   + TILE * d;      // [TILE*d]

    const float* q_row  = Q  + (batch * N + row) * d;
    const float* do_row = dO + (batch * N + row) * d;

    for (int i = tx; i < d; i += 32) { sq[i] = q_row[i]; s_do[i] = do_row[i]; }
    __syncthreads();

    const int dpt     = (d + 31) / 32;
    const int d_start = tx * dpt;
    const int d_end   = min(d_start + dpt, d);

    float dq_acc[8] = {0.f};

    for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {
        const int base = t * TILE;
        const int len  = min(TILE, N - base);

        for (int i = tx; i < len * d; i += 32) {
            int r = i / d, c = i % d;
            sK[r * d + c] = K[(batch * N + base + r) * d + c];
            sV[r * d + c] = V[(batch * N + base + r) * d + c];
        }
        __syncthreads();

        for (int j = 0; j < len; ++j) {
            // recompute S_ij = q_i · k_j * scale
            float s = 0.f;
            for (int k = tx; k < d; k += 32) s += sq[k] * sK[j * d + k];
            s = warp_reduce_sum(s) * scale;
            s = __shfl_sync(0xffffffff, s, 0);

            float p_ij = expf(s - lse_i);

            // dP_ij = dO_i · v_j
            float dp = 0.f;
            for (int k = tx; k < d; k += 32) dp += s_do[k] * sV[j * d + k];
            dp = warp_reduce_sum(dp);
            dp = __shfl_sync(0xffffffff, dp, 0);

            // dS_ij = P_ij * (dP_ij − D_i)
            float ds = p_ij * (dp - d_i);

            for (int k = d_start; k < d_end; ++k)
                dq_acc[k - d_start] += ds * sK[j * d + k] * scale;
        }
        __syncthreads();
    }

    float* dq_out = dQ + (batch * N + row) * d;
    for (int k = d_start; k < d_end; ++k) dq_out[k] = dq_acc[k - d_start];
}

// ============================================================================
// Backward: dK, dV — one block per key/value row j, iterate over Q tiles
// Grid(N, B)  Block(32)
// ============================================================================

__global__ void flash_attn_bwd_dkv_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ dO,
    const float* __restrict__ Lse,
    const float* __restrict__ D,
    float*       __restrict__ dK,
    float*       __restrict__ dV,
    int N, int d)
{
    const int col   = blockIdx.x;  // key/value row index
    const int batch = blockIdx.y;
    const int tx    = threadIdx.x;
    if (col >= N) return;

    const float scale = rsqrtf((float)d);

    extern __shared__ float smem[];
    float* sk   = smem;                 // [d]
    float* sv   = sk   + d;             // [d]
    float* sQ   = sv   + d;             // [TILE*d]
    float* s_dO = sQ   + TILE * d;      // [TILE*d]

    const float* k_row = K + (batch * N + col) * d;
    const float* v_row = V + (batch * N + col) * d;

    for (int i = tx; i < d; i += 32) { sk[i] = k_row[i]; sv[i] = v_row[i]; }
    __syncthreads();

    const int dpt     = (d + 31) / 32;
    const int d_start = tx * dpt;
    const int d_end   = min(d_start + dpt, d);

    float dk_acc[8] = {0.f};
    float dv_acc[8] = {0.f};

    for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {
        const int base = t * TILE;
        const int len  = min(TILE, N - base);

        for (int i = tx; i < len * d; i += 32) {
            int r = i / d, c = i % d;
            sQ  [r * d + c] = Q [(batch * N + base + r) * d + c];
            s_dO[r * d + c] = dO[(batch * N + base + r) * d + c];
        }
        __syncthreads();

        for (int i = 0; i < len; ++i) {
            const int gi = base + i;

            // recompute S_{i,j} = Q_i · K_j * scale
            float s = 0.f;
            for (int k = tx; k < d; k += 32) s += sQ[i * d + k] * sk[k];
            s = warp_reduce_sum(s) * scale;
            s = __shfl_sync(0xffffffff, s, 0);

            float p_ij = expf(s - Lse[batch * N + gi]);

            // dP_{i,j} = dO_i · V_j
            float dp = 0.f;
            for (int k = tx; k < d; k += 32) dp += s_dO[i * d + k] * sv[k];
            dp = warp_reduce_sum(dp);
            dp = __shfl_sync(0xffffffff, dp, 0);

            float ds = p_ij * (dp - D[batch * N + gi]);

            for (int k = d_start; k < d_end; ++k) {
                dk_acc[k - d_start] += ds   * sQ[i * d + k] * scale;
                dv_acc[k - d_start] += p_ij * s_dO[i * d + k];
            }
        }
        __syncthreads();
    }

    float* dk_out = dK + (batch * N + col) * d;
    float* dv_out = dV + (batch * N + col) * d;
    for (int k = d_start; k < d_end; ++k) {
        dk_out[k] = dk_acc[k - d_start];
        dv_out[k] = dv_acc[k - d_start];
    }
}

// ============================================================================
// Host entry points (called from binding.cpp)
// ============================================================================

std::vector<torch::Tensor> flash_attn_forward_cuda(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V)
{
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(Q.dim() == 3, "Q must be [B, N, d]");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat32, "only float32 supported");

    const int B = Q.size(0), N = Q.size(1), d = Q.size(2);
    TORCH_CHECK(d <= 256, "head dim d must be <= 256 (8 dims/thread × 32 threads)");

    auto O = torch::zeros_like(Q);
    auto L = torch::empty({B, N}, Q.options());

    dim3 grid(N, B), block(32);
    size_t smem = (size_t)(2 * TILE * d + d) * sizeof(float);

    flash_attn_fwd_kernel<<<grid, block, smem>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        O.data_ptr<float>(), L.data_ptr<float>(), N, d);

    return {O, L};
}

std::vector<torch::Tensor> flash_attn_backward_cuda(
    torch::Tensor dO, torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor O, torch::Tensor L)
{
    const int B = Q.size(0), N = Q.size(1), d = Q.size(2);

    auto D_vec = torch::empty({B, N}, Q.options());
    auto dQ = torch::zeros_like(Q);
    auto dK = torch::zeros_like(K);
    auto dV = torch::zeros_like(V);

    dim3 grid(N, B), block(32);

    // Step 1 — D_i = <dO_i, O_i>
    flash_attn_bwd_D_kernel<<<grid, block>>>(
        dO.data_ptr<float>(), O.data_ptr<float>(),
        D_vec.data_ptr<float>(), N, d);

    size_t smem = (size_t)(2 * d + 2 * TILE * d) * sizeof(float);

    // Step 2 — dQ
    flash_attn_bwd_dq_kernel<<<grid, block, smem>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        dO.data_ptr<float>(), L.data_ptr<float>(), D_vec.data_ptr<float>(),
        dQ.data_ptr<float>(), N, d);

    // Step 3 — dK, dV
    flash_attn_bwd_dkv_kernel<<<grid, block, smem>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        dO.data_ptr<float>(), L.data_ptr<float>(), D_vec.data_ptr<float>(),
        dK.data_ptr<float>(), dV.data_ptr<float>(), N, d);

    return {dQ, dK, dV};
}
