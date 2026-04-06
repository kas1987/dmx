/**
 * DMX — Standalone CUDA Kernels (no torch headers)
 * Compiled with nvcc, loaded via ctypes in Python.
 *
 * All host functions are extern "C" __declspec(dllexport) for Windows DLL.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

// ============================================================
// INT16 -> FP32 dequantization
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

extern "C" __declspec(dllexport) void launch_dequant_int16(
    const int16_t* quantized, float scale, float* output, int n, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    dequant_int16_kernel<<<blocks, threads, 0, stream>>>(quantized, scale, output, n);
}

// ============================================================
// Dense delta apply: base + delta -> float32
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

extern "C" __declspec(dllexport) void launch_delta_apply_dense(
    const int16_t* base, const int16_t* delta, float scale, float* output, int n, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    delta_apply_dense_kernel<<<blocks, threads, 0, stream>>>(base, delta, scale, output, n);
}

// ============================================================
// Sparse delta scatter: patches non-zero deltas onto dequantized base
// ============================================================
__global__ void sparse_delta_scatter_kernel(
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

extern "C" __declspec(dllexport) void launch_delta_apply_sparse(
    float* output, const int16_t* base, const int* indices, const int16_t* values,
    float scale, int n, int nnz, cudaStream_t stream)
{
    // Step 1: dequantize base to output
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    dequant_int16_kernel<<<blocks, threads, 0, stream>>>(base, scale, output, n);

    // Step 2: scatter sparse deltas
    if (nnz > 0) {
        int blocks_sparse = (nnz + threads - 1) / threads;
        sparse_delta_scatter_kernel<<<blocks_sparse, threads, 0, stream>>>(
            output, base, indices, values, scale, nnz);
    }
}

// ============================================================
// In-place delta accumulate: base += delta
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

extern "C" __declspec(dllexport) void launch_delta_accumulate(
    int16_t* base, const int16_t* delta, int n, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    delta_accumulate_kernel<<<blocks, threads, 0, stream>>>(base, delta, n);
}

// ============================================================
// INT8 -> FP32 dequantization with per-block scale
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

extern "C" __declspec(dllexport) void launch_dequant_int8(
    const int8_t* quantized, const float* scales, float* output,
    int n, int block_size, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    dequant_int8_kernel<<<blocks, threads, 0, stream>>>(quantized, scales, output, n, block_size);
}
