/**
 * DMX v2 — Complete CUDA Kernels for Compression + Decompression
 *
 * COMPRESSION KERNELS:
 *   quantize_int16:     float32 -> int16 (aligned scale)
 *   quantize_int32:     float32 -> int32 (aligned scale, practically lossless)
 *   delta_compute_i16:  quantize both + subtract -> int16 delta
 *   delta_compute_i32:  quantize both + subtract -> int32 delta
 *   bfp_compress:       float16/bf16 -> shared exponent + truncated mantissa
 *
 * DECOMPRESSION KERNELS:
 *   dequant_int16:      int16 -> float32 (from v1)
 *   dequant_int32:      int32 -> float32
 *   delta_apply_i16:    base + delta_int16 -> float32 (from v1)
 *   delta_apply_i32:    base + delta_int32 -> float32
 *   bfp_decompress:     shared exponent + mantissa -> float16
 *
 * Patent Pending. (c) 2026 William J. Riley. MIT License.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// ============================================================
// COMPRESSION: Float -> Aligned Integer Quantization
// ============================================================

__global__ void quantize_int16_kernel(
    const float* __restrict__ input,
    float inv_scale,  // 1.0 / scale = 32767.0 / max_abs
    int16_t* __restrict__ output,
    int num_elements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        float v = input[i] * inv_scale;
        v = fminf(fmaxf(v, -32767.0f), 32767.0f);
        output[i] = (int16_t)rintf(v);
    }
}

__global__ void quantize_int32_kernel(
    const float* __restrict__ input,
    double inv_scale,  // 1.0 / scale = 2147483647.0 / max_abs
    int32_t* __restrict__ output,
    int num_elements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        double v = (double)input[i] * inv_scale;
        v = fmin(fmax(v, -2147483647.0), 2147483647.0);
        output[i] = (int32_t)rint(v);
    }
}

// ============================================================
// COMPRESSION: Fused Delta Compute (quantize both + subtract)
// ============================================================

__global__ void delta_compute_i16_kernel(
    const float* __restrict__ base,
    const float* __restrict__ target,
    float inv_scale,
    int16_t* __restrict__ delta,
    int num_elements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        float bv = base[i] * inv_scale;
        float tv = target[i] * inv_scale;
        bv = fminf(fmaxf(bv, -32767.0f), 32767.0f);
        tv = fminf(fmaxf(tv, -32767.0f), 32767.0f);
        int32_t bi = (int32_t)rintf(bv);
        int32_t ti = (int32_t)rintf(tv);
        delta[i] = (int16_t)(ti - bi);
    }
}

__global__ void delta_compute_i32_kernel(
    const float* __restrict__ base,
    const float* __restrict__ target,
    double inv_scale,
    int32_t* __restrict__ delta,
    int num_elements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        double bv = (double)base[i] * inv_scale;
        double tv = (double)target[i] * inv_scale;
        bv = fmin(fmax(bv, -2147483647.0), 2147483647.0);
        tv = fmin(fmax(tv, -2147483647.0), 2147483647.0);
        int64_t bi = (int64_t)rint(bv);
        int64_t ti = (int64_t)rint(tv);
        delta[i] = (int32_t)(ti - bi);
    }
}

// ============================================================
// COMPRESSION: BFP (Block Floating Point)
// float16 -> shared exponent (uint8) + truncated mantissa (uint8)
// ============================================================

__global__ void bfp_compress_kernel(
    const uint16_t* __restrict__ fp16_input,  // raw FP16 bits
    uint8_t* __restrict__ exponents,           // one per group
    uint8_t* __restrict__ mantissas,           // sign + truncated mantissa
    int num_elements,
    int group_size,
    int mantissa_bits)
{
    int group_id = blockIdx.x * blockDim.x + threadIdx.x;
    int num_groups = (num_elements + group_size - 1) / group_size;
    if (group_id >= num_groups) return;

    int start = group_id * group_size;
    int end = min(start + group_size, num_elements);

    // Find max exponent in group
    uint8_t max_exp = 0;
    for (int i = start; i < end; i++) {
        uint16_t bits = fp16_input[i];
        uint8_t exp = (bits >> 10) & 0x1F;
        if (exp > max_exp) max_exp = exp;
    }
    exponents[group_id] = max_exp;

    // Extract sign + truncated mantissa for each element
    int shift = 10 - mantissa_bits;  // bits to discard from 10-bit mantissa
    uint8_t mant_mask = (1 << mantissa_bits) - 1;

    for (int i = start; i < end; i++) {
        uint16_t bits = fp16_input[i];
        uint8_t sign = (bits >> 15) & 1;
        uint8_t exp = (bits >> 10) & 0x1F;
        uint16_t mant = bits & 0x3FF;  // 10-bit mantissa

        // Adjust mantissa for exponent difference
        int exp_diff = (int)max_exp - (int)exp;
        if (exp_diff > 0 && exp > 0) {
            // Shift mantissa right, adding implicit 1
            mant = (mant | 0x400) >> exp_diff;
        } else if (exp == 0) {
            mant = 0;  // subnormal → zero
        }

        // Truncate to mantissa_bits
        uint8_t trunc = (uint8_t)((mant >> shift) & mant_mask);
        mantissas[i] = (sign << mantissa_bits) | trunc;
    }
}

// ============================================================
// DECOMPRESSION: BFP -> float16
// ============================================================

__global__ void bfp_decompress_kernel(
    const uint8_t* __restrict__ exponents,
    const uint8_t* __restrict__ mantissas,
    uint16_t* __restrict__ fp16_output,
    int num_elements,
    int group_size,
    int mantissa_bits)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;

    int group_id = i / group_size;
    int shared_exp = (int)exponents[group_id];

    uint8_t mant_byte = mantissas[i];
    uint8_t mant_mask = (1 << mantissa_bits) - 1;
    uint8_t sign = (mant_byte >> mantissa_bits) & 1;
    uint16_t truncated = (uint16_t)(mant_byte & mant_mask);

    // Zero case: truncated == 0 → output ±0
    if (truncated == 0) {
        fp16_output[i] = ((uint16_t)sign << 15);
        return;
    }

    // Reconstruct to 11-bit position (same as compression's inverse)
    int shift_amount = 11 - mantissa_bits;
    uint32_t recon_11 = (uint32_t)truncated << shift_amount;

    // Leading-one-bit detection: find highest set bit in the 11-bit value.
    // __clz counts leading zeros in a 32-bit int, so for an 11-bit value
    // stored in a 32-bit int, bit_pos = 31 - __clz(recon_11).
    // bit_pos 10 means the value had the same exponent as shared (offset=0),
    // bit_pos 9 means offset=1, etc.
    int bit_pos = 31 - __clz(recon_11);
    int offset = 10 - bit_pos;
    int actual_exp = shared_exp - offset;

    // Shift recon_11 left by offset to align the leading 1 to bit 10,
    // then mask to get the 10-bit mantissa (stripping the implicit leading 1)
    uint16_t mant_10 = (uint16_t)(((recon_11 << offset) & 0x3FFu));

    // Clamp exponent to valid FP16 range [0, 31]
    if (actual_exp < 0) actual_exp = 0;
    if (actual_exp > 31) actual_exp = 31;

    // Reassemble FP16: sign(1) | exponent(5) | mantissa(10)
    uint16_t bits = ((uint16_t)sign << 15) | ((uint16_t)actual_exp << 10) | mant_10;
    fp16_output[i] = bits;
}

// ============================================================
// DECOMPRESSION: BFP -> bfloat16
// ============================================================

__global__ void bfp_decompress_kernel_bf16(
    const uint8_t* __restrict__ exponents,
    const uint8_t* __restrict__ mantissas,
    uint16_t* __restrict__ bf16_output,
    int num_elements,
    int group_size,
    int mantissa_bits)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;

    int group_id = i / group_size;
    int shared_exp = (int)exponents[group_id];

    uint8_t mant_byte = mantissas[i];
    uint8_t mant_mask = (1 << mantissa_bits) - 1;
    uint8_t sign = (mant_byte >> mantissa_bits) & 1;
    uint16_t truncated = (uint16_t)(mant_byte & mant_mask);

    // Zero case: truncated == 0 → output ±0
    if (truncated == 0) {
        bf16_output[i] = ((uint16_t)sign << 15);
        return;
    }

    // BF16: implicit 1 + 7 mantissa bits = 8 total mantissa bits
    int total_mant_bits = 8;
    int shift_amount = total_mant_bits - mantissa_bits;
    uint32_t recon = (uint32_t)truncated << shift_amount;

    // Leading-one scan: find highest set bit
    int top_bit = total_mant_bits - 1;  // = 7
    int bit_pos = 31 - __clz(recon);
    int offset = top_bit - bit_pos;
    int actual_exp = shared_exp - offset;

    // Shift up to align leading 1 to bit 7, then mask to get 7-bit mantissa
    uint16_t mant_7 = (uint16_t)(((recon << offset) & 0x7Fu));

    // Clamp exponent to BF16's 8-bit range [0, 255]
    if (actual_exp < 0) actual_exp = 0;
    if (actual_exp > 255) actual_exp = 255;

    uint16_t bits = ((uint16_t)sign << 15) | ((uint16_t)actual_exp << 7) | mant_7;
    bf16_output[i] = bits;
}

// ============================================================
// DECOMPRESSION: INT16 -> FP32 (from v1)
// ============================================================

__global__ void dequant_int16_kernel(
    const int16_t* __restrict__ quantized,
    float scale,
    float* __restrict__ output,
    int num_elements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        output[i] = (float)quantized[i] * scale;
    }
}

// ============================================================
// DECOMPRESSION: INT32 -> FP32
// ============================================================

__global__ void dequant_int32_kernel(
    const int32_t* __restrict__ quantized,
    double scale,
    float* __restrict__ output,
    int num_elements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        output[i] = (float)((double)quantized[i] * scale);
    }
}

// ============================================================
// DECOMPRESSION: Dense delta apply int16 (from v1)
// ============================================================

__global__ void delta_apply_i16_kernel(
    const int16_t* __restrict__ base,
    const int16_t* __restrict__ delta,
    float scale,
    float* __restrict__ output,
    int num_elements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        int32_t sum = (int32_t)base[i] + (int32_t)delta[i];
        output[i] = (float)sum * scale;
    }
}

// ============================================================
// DECOMPRESSION: Dense delta apply int32 (new)
// ============================================================

__global__ void delta_apply_i32_kernel(
    const int32_t* __restrict__ base,
    const int32_t* __restrict__ delta,
    double scale,
    float* __restrict__ output,
    int num_elements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        int64_t sum = (int64_t)base[i] + (int64_t)delta[i];
        output[i] = (float)((double)sum * scale);
    }
}

// ============================================================
// DECOMPRESSION: Sparse delta apply (from v1, extended)
// ============================================================

__global__ void delta_sparse_scatter_i16_kernel(
    float* __restrict__ output,
    const int16_t* __restrict__ base,
    const int* __restrict__ indices,
    const int16_t* __restrict__ values,
    float scale,
    int nnz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnz) {
        int idx = indices[i];
        int32_t sum = (int32_t)base[idx] + (int32_t)values[i];
        output[idx] = (float)sum * scale;
    }
}

// ============================================================
// UTILITY: Count zeros in int16/int32 tensor (for sparsity stats)
// ============================================================

__global__ void count_zeros_i16_kernel(
    const int16_t* __restrict__ data,
    int* __restrict__ count,
    int num_elements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements && data[i] == 0) {
        atomicAdd(count, 1);
    }
}

__global__ void count_zeros_i32_kernel(
    const int32_t* __restrict__ data,
    int* __restrict__ count,
    int num_elements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements && data[i] == 0) {
        atomicAdd(count, 1);
    }
}

// ============================================================
// FUSED DEQUANT-GEMM: BFP compressed weight × FP16 input
// BF1 (v1) skeleton preserved as fallback, BF2 optimized below
// ============================================================

#include <cuda_fp16.h>

#define FUSED_TILE_M 64
#define FUSED_TILE_N 64
#define FUSED_TILE_K 32
#define FUSED_BLOCK_SIZE 128  // 4 warps

// Device function: dequant a single BFP element to half
__device__ __forceinline__ __half bfp_dequant_element(uint8_t shared_exp, uint8_t mant_byte, int mantissa_bits) {
    uint8_t mant_mask = (1 << mantissa_bits) - 1;
    uint8_t sign = (mant_byte >> mantissa_bits) & 1;
    uint16_t truncated = (uint16_t)(mant_byte & mant_mask);

    if (truncated == 0) {
        uint16_t bits = ((uint16_t)sign << 15);
        return *reinterpret_cast<__half*>(&bits);
    }

    int shift_amount = 11 - mantissa_bits;
    uint32_t recon_11 = (uint32_t)truncated << shift_amount;
    int bit_pos = 31 - __clz(recon_11);
    int offset = 10 - bit_pos;
    int actual_exp = (int)shared_exp - offset;

    uint16_t mant_10 = (uint16_t)(((recon_11 << offset) & 0x3FFu));
    if (actual_exp < 0) actual_exp = 0;
    if (actual_exp > 31) actual_exp = 31;

    uint16_t bits = ((uint16_t)sign << 15) | ((uint16_t)actual_exp << 10) | mant_10;
    return *reinterpret_cast<__half*>(&bits);
}

// ============================================================
// BF1 FALLBACK: Original unoptimized fused kernel (renamed)
// ============================================================

__global__ void bfp_fused_linear_kernel_v1(
    const __half* __restrict__ input,
    const uint8_t* __restrict__ exponents,
    const uint8_t* __restrict__ sign_mant,
    __half* __restrict__ output,
    const __half* __restrict__ bias,
    int M, int N, int K,
    int group_size,
    int mantissa_bits,
    int pad_len,
    int orig_len
)
{
    int tile_m = blockIdx.x * FUSED_TILE_M;
    int tile_n = blockIdx.y * FUSED_TILE_N;
    int tid = threadIdx.x;

    __shared__ __half smem_weight[FUSED_TILE_N * FUSED_TILE_K];
    __shared__ __half smem_input[FUSED_TILE_M * FUSED_TILE_K];

    float accum[32];
    for (int i = 0; i < 32; i++) accum[i] = 0.0f;

    int K_padded = ((K + group_size - 1) / group_size) * group_size;
    int groups_per_row = K_padded / group_size;

    for (int k_step = 0; k_step < K; k_step += FUSED_TILE_K) {
        int weight_elems = FUSED_TILE_N * FUSED_TILE_K;
        for (int idx = tid; idx < weight_elems; idx += FUSED_BLOCK_SIZE) {
            int local_n = idx / FUSED_TILE_K;
            int local_k = idx % FUSED_TILE_K;
            int global_n = tile_n + local_n;
            int global_k = k_step + local_k;

            if (global_n < N && global_k < K) {
                int flat_idx = global_n * K_padded + global_k;
                int group_id = global_n * groups_per_row + global_k / group_size;
                uint8_t exp = exponents[group_id];
                uint8_t mant = sign_mant[flat_idx];
                smem_weight[idx] = bfp_dequant_element(exp, mant, mantissa_bits);
            } else {
                smem_weight[idx] = __float2half(0.0f);
            }
        }

        int input_elems = FUSED_TILE_M * FUSED_TILE_K;
        for (int idx = tid; idx < input_elems; idx += FUSED_BLOCK_SIZE) {
            int local_m = idx / FUSED_TILE_K;
            int local_k = idx % FUSED_TILE_K;
            int global_m = tile_m + local_m;
            int global_k = k_step + local_k;

            if (global_m < M && global_k < K) {
                smem_input[idx] = input[global_m * K + global_k];
            } else {
                smem_input[idx] = __float2half(0.0f);
            }
        }

        __syncthreads();

        for (int out_idx = tid; out_idx < FUSED_TILE_M * FUSED_TILE_N; out_idx += FUSED_BLOCK_SIZE) {
            int local_m = out_idx / FUSED_TILE_N;
            int local_n = out_idx % FUSED_TILE_N;
            int accum_slot = out_idx / FUSED_BLOCK_SIZE;

            float dot = 0.0f;
            for (int kk = 0; kk < FUSED_TILE_K; kk++) {
                float a = __half2float(smem_input[local_m * FUSED_TILE_K + kk]);
                float w = __half2float(smem_weight[local_n * FUSED_TILE_K + kk]);
                dot += a * w;
            }
            accum[accum_slot] += dot;
        }

        __syncthreads();
    }

    for (int out_idx = tid; out_idx < FUSED_TILE_M * FUSED_TILE_N; out_idx += FUSED_BLOCK_SIZE) {
        int local_m = out_idx / FUSED_TILE_N;
        int local_n = out_idx % FUSED_TILE_N;
        int global_m = tile_m + local_m;
        int global_n = tile_n + local_n;
        int accum_slot = out_idx / FUSED_BLOCK_SIZE;

        if (global_m < M && global_n < N) {
            float val = accum[accum_slot];
            if (bias != nullptr) {
                val += __half2float(bias[global_n]);
            }
            output[global_m * N + global_n] = __float2half(val);
        }
    }
}

// ============================================================
// BF2 OPTIMIZED: WMMA tensor-core + double-buffered fused kernel
//
// Optimizations applied:
//   1. WMMA tensor cores: 16x16x16 MMA fragments via nvcuda::wmma
//      (massive throughput gain over scalar FMA on sm_89)
//   2. Double-buffered shared memory: overlap next tile load with
//      current tile compute
//   3. Register blocking: each warp computes a 16x32 output sub-tile
//      using 2 WMMA fragments along N (16x16 each)
//   4. Shared memory padding: +8 pad to avoid bank conflicts on
//      half-precision 16-wide access patterns
// ============================================================

#include <mma.h>
using namespace nvcuda;

// WMMA fragment dimensions
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// We keep TILE_M=64, TILE_N=64, but TILE_K=32 (two WMMA K-steps per tile)
// Each block has 128 threads = 4 warps
// Warp layout: 2 warps along M (each handles 32 rows via 2 WMMA-M fragments)
//              2 warps along N (each handles 32 cols via 2 WMMA-N fragments)
// Actually: 4 warps in a 2x2 grid, each warp handles a 32x32 sub-tile
// with 2x2 = 4 WMMA fragments (16x16 each), iterated over 2 WMMA K-steps

// Smem padded stride for bank conflict avoidance
#define SMEM_PAD 8
#define SMEM_STRIDE_K (FUSED_TILE_K + SMEM_PAD)

__global__ void bfp_fused_linear_kernel(
    const __half* __restrict__ input,        // [M, K]
    const uint8_t* __restrict__ exponents,   // [N_groups]
    const uint8_t* __restrict__ sign_mant,   // [N * K_padded] packed mantissa
    __half* __restrict__ output,             // [M, N]
    const __half* __restrict__ bias,         // [N] or nullptr
    int M, int N, int K,
    int group_size,
    int mantissa_bits,
    int pad_len,
    int orig_len
)
{
    const int tile_m = blockIdx.x * FUSED_TILE_M;
    const int tile_n = blockIdx.y * FUSED_TILE_N;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Warp layout: 2x2 grid of warps
    // warp_row: 0 or 1 (handles rows 0-31 or 32-63 of the tile)
    // warp_col: 0 or 1 (handles cols 0-31 or 32-63 of the tile)
    const int warp_row = warp_id / 2;  // 0 or 1
    const int warp_col = warp_id % 2;  // 0 or 1

    // BFP layout params
    const int K_padded = ((K + group_size - 1) / group_size) * group_size;
    const int groups_per_row = K_padded / group_size;

    // Double-buffered shared memory
    // Weight: [TILE_N, TILE_K+PAD] in row-major (N is row, K is col)
    // Input:  [TILE_M, TILE_K+PAD] in row-major (M is row, K is col)
    __shared__ __half smem_weight[2][FUSED_TILE_N * SMEM_STRIDE_K];
    __shared__ __half smem_input[2][FUSED_TILE_M * SMEM_STRIDE_K];

    // WMMA accumulators: each warp computes a 32x32 sub-tile = 2x2 WMMA fragments
    // Fragment: accumulator for 16x16 output, stored as FP32
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);

    const int num_k_steps = (K + FUSED_TILE_K - 1) / FUSED_TILE_K;

    // ---- Tile loading helpers (same structure as before, into double-buffered smem) ----

    #define LOAD_TILES(buf, k_step_val)                                              \
    {                                                                                \
        const int _ks = (k_step_val);                                                \
        /* Load weight tile [TILE_N x TILE_K] */                                     \
        const int weight_elems = FUSED_TILE_N * FUSED_TILE_K;                        \
        for (int idx = tid; idx < weight_elems; idx += FUSED_BLOCK_SIZE) {           \
            int local_n = idx / FUSED_TILE_K;                                        \
            int local_k = idx % FUSED_TILE_K;                                        \
            int global_n = tile_n + local_n;                                         \
            int global_k = _ks + local_k;                                            \
            if (global_n < N && global_k < K) {                                      \
                int flat_idx = global_n * K_padded + global_k;                       \
                int group_id = global_n * groups_per_row + global_k / group_size;    \
                uint8_t exp = exponents[group_id];                                   \
                uint8_t mant = sign_mant[flat_idx];                                  \
                smem_weight[(buf)][local_n * SMEM_STRIDE_K + local_k] =              \
                    bfp_dequant_element(exp, mant, mantissa_bits);                   \
            } else {                                                                 \
                smem_weight[(buf)][local_n * SMEM_STRIDE_K + local_k] =              \
                    __float2half(0.0f);                                               \
            }                                                                        \
        }                                                                            \
        /* Load input tile [TILE_M x TILE_K] */                                      \
        const int input_elems = FUSED_TILE_M * FUSED_TILE_K;                         \
        for (int idx = tid; idx < input_elems; idx += FUSED_BLOCK_SIZE) {            \
            int local_m = idx / FUSED_TILE_K;                                        \
            int local_k = idx % FUSED_TILE_K;                                        \
            int global_m = tile_m + local_m;                                         \
            int global_k = _ks + local_k;                                            \
            if (global_m < M && global_k < K) {                                      \
                smem_input[(buf)][local_m * SMEM_STRIDE_K + local_k] =               \
                    input[global_m * K + global_k];                                  \
            } else {                                                                 \
                smem_input[(buf)][local_m * SMEM_STRIDE_K + local_k] =               \
                    __float2half(0.0f);                                               \
            }                                                                        \
        }                                                                            \
    }

    // ---- Compute macro: WMMA matmul from smem[buf] ----
    // Each warp loads its 32x32 sub-tile as 2x2 WMMA 16x16 fragments
    // TILE_K = 32 = 2 * WMMA_K(16), so we iterate twice over the K-dimension
    #define COMPUTE_WMMA(buf)                                                        \
    {                                                                                \
        for (int wk = 0; wk < 2; wk++) {                                            \
            /* K offset within the tile for this WMMA K-step */                      \
            int k_off = wk * WMMA_K;                                                 \
            /* Load 2 input fragments (rows: warp_row*32 + {0,16}, K: k_off) */      \
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half,           \
                           wmma::row_major> a_frag[2];                               \
            _Pragma("unroll")                                                        \
            for (int mi = 0; mi < 2; mi++) {                                         \
                int m_off = warp_row * 32 + mi * WMMA_M;                             \
                wmma::load_matrix_sync(a_frag[mi],                                   \
                    &smem_input[(buf)][m_off * SMEM_STRIDE_K + k_off],               \
                    SMEM_STRIDE_K);                                                  \
            }                                                                        \
            /* Load 2 weight fragments (rows: warp_col*32 + {0,16}, K: k_off) */     \
            /* Weight is [N, K] in row-major. For B matrix we need col_major */      \
            /* so that the N dimension is along columns. */                           \
            /* Actually, wmma C = A * B where A is [M,K] and B is [K,N]. */          \
            /* Our weight is [N,K], so we load it as col_major to get [K,N]. */      \
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half,           \
                           wmma::col_major> b_frag[2];                               \
            _Pragma("unroll")                                                        \
            for (int ni = 0; ni < 2; ni++) {                                         \
                int n_off = warp_col * 32 + ni * WMMA_N;                             \
                wmma::load_matrix_sync(b_frag[ni],                                   \
                    &smem_weight[(buf)][n_off * SMEM_STRIDE_K + k_off],              \
                    SMEM_STRIDE_K);                                                  \
            }                                                                        \
            /* 2x2 WMMA MMA operations */                                            \
            _Pragma("unroll")                                                        \
            for (int mi = 0; mi < 2; mi++)                                           \
                _Pragma("unroll")                                                    \
                for (int ni = 0; ni < 2; ni++)                                       \
                    wmma::mma_sync(acc[mi][ni], a_frag[mi], b_frag[ni], acc[mi][ni]);\
        }                                                                            \
    }

    // ---- Main double-buffered loop ----
    LOAD_TILES(0, 0);
    __syncthreads();

    for (int step = 0; step < num_k_steps; step++) {
        int cur_buf = step & 1;
        int nxt_buf = 1 - cur_buf;

        if (step + 1 < num_k_steps) {
            int next_k = (step + 1) * FUSED_TILE_K;
            LOAD_TILES(nxt_buf, next_k);
        }

        COMPUTE_WMMA(cur_buf);

        __syncthreads();
    }

    // ---- Write output from WMMA accumulators ----
    // Each warp stores its 2x2 WMMA fragments (16x16 each) to output.
    // We store the FP32 accumulator to a per-warp scratch region in smem,
    // then add bias and convert to FP16 for the global write.
    // Each warp gets 256 floats of scratch = 1024 bytes. 4 warps = 4KB.
    // Reuse smem_input[0] (2560 halves = 5120 bytes, plenty of room).

    // Per-warp scratch: 256 floats = 1024 bytes each. 4 warps = 4096 bytes.
    // smem_input[0] has 2560 halves = 5120 bytes, plenty of room.
    // Index by float offset: warp_id * 256 floats = warp_id * 1024 bytes.
    float* all_scratch = reinterpret_cast<float*>(&smem_input[0][0]);
    float* warp_scratch = all_scratch + warp_id * 256;

    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 2; ni++) {
            int frag_m = tile_m + warp_row * 32 + mi * WMMA_M;
            int frag_n = tile_n + warp_col * 32 + ni * WMMA_N;

            // Store accumulator fragment to per-warp scratch (FP32)
            wmma::store_matrix_sync(warp_scratch, acc[mi][ni], WMMA_N,
                                    wmma::mem_row_major);

            // Each thread in warp writes 8 of the 256 elements
            for (int idx = lane_id; idx < WMMA_M * WMMA_N; idx += 32) {
                int local_m = idx / WMMA_N;
                int local_n = idx % WMMA_N;
                int gm = frag_m + local_m;
                int gn = frag_n + local_n;
                if (gm < M && gn < N) {
                    float val = warp_scratch[idx];
                    if (bias != nullptr) {
                        val += __half2float(bias[gn]);
                    }
                    output[gm * N + gn] = __float2half(val);
                }
            }
        }
    }

    #undef LOAD_TILES
    #undef COMPUTE_WMMA
}

// ============================================================
// Python-callable functions via pybind11
// ============================================================

// --- Compression ---

torch::Tensor quantize_int16(torch::Tensor input, double scale) {
    auto n = input.numel();
    auto output = torch::empty({n}, torch::TensorOptions().dtype(torch::kInt16).device(input.device()));
    float inv_scale = (scale > 0) ? (float)(32767.0 / scale) : 1.0f;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    quantize_int16_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), inv_scale, output.data_ptr<int16_t>(), n);
    return output.reshape_as(input);
}

torch::Tensor quantize_int32(torch::Tensor input, double scale) {
    auto n = input.numel();
    auto output = torch::empty({n}, torch::TensorOptions().dtype(torch::kInt32).device(input.device()));
    double inv_scale = (scale > 0) ? (2147483647.0 / scale) : 1.0;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    quantize_int32_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), inv_scale, output.data_ptr<int32_t>(), n);
    return output.reshape_as(input);
}

torch::Tensor delta_compute_i16(torch::Tensor base, torch::Tensor target, double scale) {
    auto n = base.numel();
    auto delta = torch::empty({n}, torch::TensorOptions().dtype(torch::kInt16).device(base.device()));
    float inv_scale = (scale > 0) ? (float)(32767.0 / scale) : 1.0f;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    delta_compute_i16_kernel<<<blocks, threads>>>(
        base.data_ptr<float>(), target.data_ptr<float>(),
        inv_scale, delta.data_ptr<int16_t>(), n);
    return delta.reshape_as(base);
}

torch::Tensor delta_compute_i32(torch::Tensor base, torch::Tensor target, double scale) {
    auto n = base.numel();
    auto delta = torch::empty({n}, torch::TensorOptions().dtype(torch::kInt32).device(base.device()));
    double inv_scale = (scale > 0) ? (2147483647.0 / scale) : 1.0;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    delta_compute_i32_kernel<<<blocks, threads>>>(
        base.data_ptr<float>(), target.data_ptr<float>(),
        inv_scale, delta.data_ptr<int32_t>(), n);
    return delta.reshape_as(base);
}

std::vector<torch::Tensor> bfp_compress(torch::Tensor fp16_input, int group_size, int mantissa_bits) {
    auto n = fp16_input.numel();
    int num_groups = (n + group_size - 1) / group_size;

    auto exponents = torch::empty({num_groups}, torch::TensorOptions().dtype(torch::kUInt8).device(fp16_input.device()));
    auto mantissas = torch::empty({n}, torch::TensorOptions().dtype(torch::kUInt8).device(fp16_input.device()));

    // Reinterpret FP16 tensor as uint16
    auto input_u16 = fp16_input.view(torch::kInt16);

    int threads = 256;
    int blocks = (num_groups + threads - 1) / threads;
    bfp_compress_kernel<<<blocks, threads>>>(
        reinterpret_cast<const uint16_t*>(input_u16.data_ptr<int16_t>()),
        exponents.data_ptr<uint8_t>(),
        mantissas.data_ptr<uint8_t>(),
        n, group_size, mantissa_bits);

    return {exponents, mantissas};
}

// --- Decompression ---

torch::Tensor dequantize_int16(torch::Tensor quantized, double scale) {
    auto n = quantized.numel();
    auto output = torch::empty({n}, torch::TensorOptions().dtype(torch::kFloat32).device(quantized.device()));
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    dequant_int16_kernel<<<blocks, threads>>>(
        quantized.data_ptr<int16_t>(), (float)scale,
        output.data_ptr<float>(), n);
    return output.reshape_as(quantized);
}

torch::Tensor dequantize_int32(torch::Tensor quantized, double scale) {
    auto n = quantized.numel();
    auto output = torch::empty({n}, torch::TensorOptions().dtype(torch::kFloat32).device(quantized.device()));
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    dequant_int32_kernel<<<blocks, threads>>>(
        quantized.data_ptr<int32_t>(), scale,
        output.data_ptr<float>(), n);
    return output.reshape_as(quantized);
}

torch::Tensor delta_apply_i16(torch::Tensor base, torch::Tensor delta, double scale) {
    auto n = base.numel();
    auto output = torch::empty({n}, torch::TensorOptions().dtype(torch::kFloat32).device(base.device()));
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    delta_apply_i16_kernel<<<blocks, threads>>>(
        base.data_ptr<int16_t>(), delta.data_ptr<int16_t>(),
        (float)scale, output.data_ptr<float>(), n);
    return output.reshape_as(base);
}

torch::Tensor delta_apply_i32(torch::Tensor base, torch::Tensor delta, double scale) {
    auto n = base.numel();
    auto output = torch::empty({n}, torch::TensorOptions().dtype(torch::kFloat32).device(base.device()));
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    delta_apply_i32_kernel<<<blocks, threads>>>(
        base.data_ptr<int32_t>(), delta.data_ptr<int32_t>(),
        scale, output.data_ptr<float>(), n);
    return output.reshape_as(base);
}

torch::Tensor bfp_decompress(torch::Tensor exponents, torch::Tensor mantissas,
                              int group_size, int mantissa_bits) {
    auto n = mantissas.numel();
    auto output = torch::empty({n}, torch::TensorOptions().dtype(torch::kInt16).device(mantissas.device()));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    bfp_decompress_kernel<<<blocks, threads>>>(
        exponents.data_ptr<uint8_t>(),
        mantissas.data_ptr<uint8_t>(),
        reinterpret_cast<uint16_t*>(output.data_ptr<int16_t>()),
        n, group_size, mantissa_bits);

    return output.view(torch::kFloat16);
}

torch::Tensor bfp_decompress_bf16(torch::Tensor exponents, torch::Tensor mantissas,
                                  int group_size, int mantissa_bits) {
    auto n = mantissas.numel();
    auto output = torch::empty({n}, torch::TensorOptions().dtype(torch::kInt16).device(mantissas.device()));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    bfp_decompress_kernel_bf16<<<blocks, threads>>>(
        exponents.data_ptr<uint8_t>(),
        mantissas.data_ptr<uint8_t>(),
        reinterpret_cast<uint16_t*>(output.data_ptr<int16_t>()),
        n, group_size, mantissa_bits);

    return output.view(torch::kBFloat16);
}

// --- Fused dequant-GEMM ---

torch::Tensor bfp_fused_linear(
    torch::Tensor input,         // float16 [M, K]
    torch::Tensor exponents,     // uint8 [N_groups]
    torch::Tensor sign_mant,     // uint8 [N * K_padded]
    torch::Tensor bias,          // float16 [N] or empty tensor (numel==0 means no bias)
    int M, int N, int K,
    int group_size,
    int mantissa_bits,
    int pad_len,
    int orig_len
) {
    auto output = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kFloat16).device(input.device()));

    const __half* bias_ptr = nullptr;
    if (bias.numel() > 0) {
        bias_ptr = reinterpret_cast<const __half*>(bias.data_ptr<at::Half>());
    }

    dim3 grid(
        (M + FUSED_TILE_M - 1) / FUSED_TILE_M,
        (N + FUSED_TILE_N - 1) / FUSED_TILE_N
    );
    dim3 block(FUSED_BLOCK_SIZE);

    bfp_fused_linear_kernel<<<grid, block>>>(
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        exponents.data_ptr<uint8_t>(),
        sign_mant.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        bias_ptr,
        M, N, K,
        group_size,
        mantissa_bits,
        pad_len,
        orig_len
    );

    return output;
}

// --- Utility ---

int count_zeros_i16(torch::Tensor data) {
    auto n = data.numel();
    auto count = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).device(data.device()));
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    count_zeros_i16_kernel<<<blocks, threads>>>(
        data.data_ptr<int16_t>(), count.data_ptr<int>(), n);
    return count.item<int>();
}

int count_zeros_i32(torch::Tensor data) {
    auto n = data.numel();
    auto count = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).device(data.device()));
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    count_zeros_i32_kernel<<<blocks, threads>>>(
        data.data_ptr<int32_t>(), count.data_ptr<int>(), n);
    return count.item<int>();
}

// ============================================================
// Module registration
// ============================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Compression
    m.def("quantize_int16", &quantize_int16, "Float32 -> INT16 aligned quantization (CUDA)");
    m.def("quantize_int32", &quantize_int32, "Float32 -> INT32 aligned quantization (CUDA)");
    m.def("delta_compute_i16", &delta_compute_i16, "Fused quantize + subtract -> INT16 delta (CUDA)");
    m.def("delta_compute_i32", &delta_compute_i32, "Fused quantize + subtract -> INT32 delta (CUDA)");
    m.def("bfp_compress", &bfp_compress, "FP16 -> BFP shared exponent + truncated mantissa (CUDA)");

    // Decompression
    m.def("dequantize_int16", &dequantize_int16, "INT16 -> Float32 dequantization (CUDA)");
    m.def("dequantize_int32", &dequantize_int32, "INT32 -> Float32 dequantization (CUDA)");
    m.def("delta_apply_i16", &delta_apply_i16, "Apply INT16 delta to base and dequantize (CUDA)");
    m.def("delta_apply_i32", &delta_apply_i32, "Apply INT32 delta to base and dequantize (CUDA)");
    m.def("bfp_decompress", &bfp_decompress, "BFP shared exponent + mantissa -> FP16 (CUDA)");
    m.def("bfp_decompress_bf16", &bfp_decompress_bf16, "BFP shared exponent + mantissa -> BF16 (CUDA)");

    // Fused dequant-GEMM
    m.def("bfp_fused_linear", &bfp_fused_linear, "BFP fused dequant + linear (CUDA)");

    // Utility
    m.def("count_zeros_i16", &count_zeros_i16, "Count zero elements in INT16 tensor (CUDA)");
    m.def("count_zeros_i32", &count_zeros_i32, "Count zero elements in INT32 tensor (CUDA)");
}
