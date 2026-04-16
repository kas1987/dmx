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
// BF1 skeleton — correctness-first, not optimized
// ============================================================

#include <cuda_fp16.h>

#define FUSED_TILE_M 64
#define FUSED_TILE_N 64
#define FUSED_TILE_K 32
#define FUSED_BLOCK_SIZE 128  // 4 warps

// Device function: dequant a single BFP element to half
__device__ __half bfp_dequant_element(uint8_t shared_exp, uint8_t mant_byte, int mantissa_bits) {
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

__global__ void bfp_fused_linear_kernel(
    const __half* __restrict__ input,        // [M, K]
    const uint8_t* __restrict__ exponents,   // [N_groups] where N_groups = (N * K_padded) / group_size
    const uint8_t* __restrict__ sign_mant,   // [N * K_padded] packed mantissa
    __half* __restrict__ output,             // [M, N]
    const __half* __restrict__ bias,         // [N] or nullptr
    int M, int N, int K,
    int group_size,
    int mantissa_bits,
    int pad_len,                             // total padded length (N * K_padded_per_row or flat)
    int orig_len                             // original unpadded length
)
{
    // Each block computes a TILE_M x TILE_N output tile
    int tile_m = blockIdx.x * FUSED_TILE_M;
    int tile_n = blockIdx.y * FUSED_TILE_N;
    int tid = threadIdx.x;

    // Shared memory: weight tile (dequanted) + input tile
    __shared__ __half smem_weight[FUSED_TILE_N * FUSED_TILE_K];  // [TILE_N, TILE_K]
    __shared__ __half smem_input[FUSED_TILE_M * FUSED_TILE_K];   // [TILE_M, TILE_K]

    // Each thread accumulates a subset of the TILE_M x TILE_N output
    // Simple approach: each thread handles multiple (m, n) pairs
    // With 128 threads and 64x64=4096 outputs, each thread does 32 outputs
    // Layout: thread tid handles output elements tid, tid+128, tid+256, ...
    float accum[32];
    for (int i = 0; i < 32; i++) accum[i] = 0.0f;

    // K_padded is the padded K dimension per weight row (aligned to group_size)
    int K_padded = ((K + group_size - 1) / group_size) * group_size;
    int groups_per_row = K_padded / group_size;

    // Iterate over K in steps of TILE_K (= group_size = 32)
    for (int k_step = 0; k_step < K; k_step += FUSED_TILE_K) {
        // === Phase 1: Load and dequant weight tile [TILE_N x TILE_K] into shared memory ===
        // Weight layout: weight[n, k] is stored at sign_mant[n * K_padded + k]
        // Exponent for weight[n, k] is at exponents[n * groups_per_row + k / group_size]
        int weight_elems = FUSED_TILE_N * FUSED_TILE_K;  // 2048
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

        // === Phase 2: Load input tile [TILE_M x TILE_K] into shared memory ===
        int input_elems = FUSED_TILE_M * FUSED_TILE_K;  // 2048
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

        // === Phase 3: Compute partial matmul from shared memory ===
        // output[m, n] += sum_k(input[m, k] * weight[n, k])
        // Each thread handles elements: for flat index = tid, tid+128, tid+256, ...
        for (int out_idx = tid; out_idx < FUSED_TILE_M * FUSED_TILE_N; out_idx += FUSED_BLOCK_SIZE) {
            int local_m = out_idx / FUSED_TILE_N;
            int local_n = out_idx % FUSED_TILE_N;
            int accum_slot = out_idx / FUSED_BLOCK_SIZE;  // 0..31

            // Dot product over TILE_K
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

    // === Phase 4: Write output ===
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

    // Fused dequant-GEMM
    m.def("bfp_fused_linear", &bfp_fused_linear, "BFP fused dequant + linear (CUDA)");

    // Utility
    m.def("count_zeros_i16", &count_zeros_i16, "Count zero elements in INT16 tensor (CUDA)");
    m.def("count_zeros_i32", &count_zeros_i32, "Count zero elements in INT32 tensor (CUDA)");
}
