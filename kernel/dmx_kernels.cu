/**
 * DMX — CUDA Kernels for Delta Multiplexed Model Format
 *
 * Kernels:
 * 1. delta_apply_dense:  base_int16 + delta_int16 -> output_float32
 * 2. delta_apply_sparse: base_int16 + sparse_delta -> output_float32
 * 3. dequant_int16:      int16 -> float32 with scale
 * 4. dequant_int8:       int8 -> float32 with scale (from vram_pager)
 *
 * Derived from vram_pager kernel (MIT license, Will Riley 2026)
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ============================================================
// INT16 -> FP32 dequantization (aligned scale, one scale per tensor)
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
// Dense delta apply: base + delta -> dequantized output
// Fused: avoids materializing the int16 sum, goes straight to float32
// ============================================================
__global__ void delta_apply_dense_kernel(
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
// Sparse delta apply: base + sparse_delta -> dequantized output
// First dequantizes base to output, then patches non-zero deltas
// ============================================================
__global__ void sparse_delta_scatter_kernel(
    float* __restrict__ output,         // already contains dequantized base
    const int16_t* __restrict__ base,   // base quantized values (for re-summing)
    const int* __restrict__ indices,    // flat indices of non-zero deltas
    const int16_t* __restrict__ values, // non-zero delta values
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
// In-place delta accumulate: update base_int16 += delta
// For chained reconstruction (base becomes new base for next layer)
// ============================================================
__global__ void delta_accumulate_kernel(
    int16_t* __restrict__ base,
    const int16_t* __restrict__ delta,
    int num_elements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        base[i] = (int16_t)((int32_t)base[i] + (int32_t)delta[i]);
    }
}

// ============================================================
// INT8 -> FP32 dequantization (from vram_pager, per-block scale)
// ============================================================
__global__ void dequant_int8_kernel(
    const int8_t* __restrict__ quantized,
    const float* __restrict__ scales,
    float* __restrict__ output,
    int num_elements,
    int block_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        int scale_idx = i / block_size;
        output[i] = (float)quantized[i] * scales[scale_idx];
    }
}

// ============================================================
// Python-callable functions via pybind11
// ============================================================

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

torch::Tensor delta_apply_dense(torch::Tensor base, torch::Tensor delta, double scale) {
    auto n = base.numel();
    auto output = torch::empty({n}, torch::TensorOptions().dtype(torch::kFloat32).device(base.device()));
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    delta_apply_dense_kernel<<<blocks, threads>>>(
        base.data_ptr<int16_t>(), delta.data_ptr<int16_t>(),
        (float)scale, output.data_ptr<float>(), n);
    return output.reshape_as(base);
}

torch::Tensor delta_apply_sparse(torch::Tensor base, torch::Tensor indices, torch::Tensor values, double scale) {
    auto n = base.numel();
    auto nnz = indices.numel();

    // First: dequantize base to output
    auto output = torch::empty({n}, torch::TensorOptions().dtype(torch::kFloat32).device(base.device()));
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    dequant_int16_kernel<<<blocks, threads>>>(
        base.data_ptr<int16_t>(), (float)scale,
        output.data_ptr<float>(), n);

    // Then: scatter non-zero deltas
    if (nnz > 0) {
        int blocks_sparse = (nnz + threads - 1) / threads;
        sparse_delta_scatter_kernel<<<blocks_sparse, threads>>>(
            output.data_ptr<float>(),
            base.data_ptr<int16_t>(),
            indices.data_ptr<int>(),
            values.data_ptr<int16_t>(),
            (float)scale, nnz);
    }

    return output.reshape_as(base);
}

void delta_accumulate(torch::Tensor base, torch::Tensor delta) {
    auto n = base.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    delta_accumulate_kernel<<<blocks, threads>>>(
        base.data_ptr<int16_t>(), delta.data_ptr<int16_t>(), n);
}

torch::Tensor dequantize_int8(torch::Tensor quantized, torch::Tensor scales, int block_size) {
    auto n = quantized.numel();
    auto output = torch::empty({n}, torch::TensorOptions().dtype(torch::kFloat32).device(quantized.device()));
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    dequant_int8_kernel<<<blocks, threads>>>(
        quantized.data_ptr<int8_t>(), scales.data_ptr<float>(),
        output.data_ptr<float>(), n, block_size);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dequantize_int16", &dequantize_int16, "INT16 -> FP32 dequantization (CUDA)");
    m.def("delta_apply_dense", &delta_apply_dense, "Apply dense INT16 delta to base and dequantize (CUDA)");
    m.def("delta_apply_sparse", &delta_apply_sparse, "Apply sparse INT16 delta to base and dequantize (CUDA)");
    m.def("delta_accumulate", &delta_accumulate, "In-place INT16 base += delta (CUDA)");
    m.def("dequantize_int8", &dequantize_int8, "INT8 -> FP32 dequantization with per-block scale (CUDA)");
}
