/**
 * DMX Standalone CUDA Kernel — no torch dependency.
 *
 * Exposes bfp_decompress as a C function callable via ctypes/cffi.
 * The kernel is identical to dmx_kernels_v2.cu but the wrapper uses
 * raw CUDA APIs instead of torch::Tensor.
 *
 * Build:
 *   nvcc -shared -o dmx_cuda.so dmx_standalone.cu -O2 --compiler-options -fPIC
 *   nvcc -shared -o dmx_cuda.dll dmx_standalone.cu -O2 (Windows)
 *
 * Patent Pending. (c) 2026 William J. Riley. MIT License.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Windows DLL export macro
#ifdef _WIN32
#define DMX_API __declspec(dllexport)
#else
#define DMX_API
#endif

// ============================================================
// BFP Decompress Kernel (identical to dmx_kernels_v2.cu)
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

    if (truncated == 0) {
        fp16_output[i] = ((uint16_t)sign << 15);
        return;
    }

    int shift_amount = 11 - mantissa_bits;
    uint32_t recon_11 = (uint32_t)truncated << shift_amount;
    int bit_pos = 31 - __clz(recon_11);
    int offset = 10 - bit_pos;
    int actual_exp = shared_exp - offset;

    uint16_t mant_10 = (uint16_t)(((recon_11 << offset) & 0x3FFu));

    if (actual_exp < 0) actual_exp = 0;
    if (actual_exp > 31) actual_exp = 31;

    uint16_t bits = ((uint16_t)sign << 15) | ((uint16_t)actual_exp << 10) | mant_10;
    fp16_output[i] = bits;
}


// ============================================================
// C API (callable via ctypes)
// ============================================================

extern "C" {

/**
 * Decompress BFP data on GPU.
 *
 * All pointers must be device (GPU) memory.
 *
 * @param d_exponents   Device pointer to uint8 exponent array [n_groups]
 * @param d_mantissas   Device pointer to uint8 packed sign+mantissa array [num_elements]
 * @param d_output      Device pointer to output uint16 array [num_elements] (reinterpret as fp16)
 * @param num_elements  Total number of elements (including padding)
 * @param group_size    BFP group size (typically 32)
 * @param mantissa_bits Mantissa precision (1-7)
 * @param stream        CUDA stream (0 for default)
 * @return              0 on success, non-zero on error
 */
DMX_API int dmx_bfp_decompress(
    const void* d_exponents,
    const void* d_mantissas,
    void* d_output,
    int num_elements,
    int group_size,
    int mantissa_bits,
    void* stream)
{
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;

    cudaStream_t cuda_stream = (cudaStream_t)stream;

    bfp_decompress_kernel<<<blocks, threads, 0, cuda_stream>>>(
        (const uint8_t*)d_exponents,
        (const uint8_t*)d_mantissas,
        (uint16_t*)d_output,
        num_elements, group_size, mantissa_bits);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return (int)err;

    return 0;
}

/**
 * Get the library version.
 */
DMX_API int dmx_version() {
    return 10305;  // 1.3.5
}

/**
 * Check if CUDA is available.
 */
DMX_API int dmx_cuda_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) return 0;
    return count > 0 ? 1 : 0;
}

}  // extern "C"
