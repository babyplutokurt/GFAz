#ifndef CODEC_GPU_NVCOMP_CUH
#define CODEC_GPU_NVCOMP_CUH

#include <vector>
#include <cstdint>
#include <string>
#include <cuda_runtime.h>

namespace gpu_codec {

/**
 * nvComp GPU ZSTD compression result.
 * Contains compressed data copied back to host memory.
 */
struct NvcompCompressedBlock {
    std::vector<uint8_t> payload;
    size_t original_size = 0;
};

/**
 * Compress data using nvComp ZSTD on GPU.
 * 
 * Input data is on host, gets copied to device, compressed on GPU,
 * and the result is copied back to host.
 * 
 * @param input Host vector of bytes to compress
 * @param stream CUDA stream to use (default: 0)
 * @return NvcompCompressedBlock containing compressed payload and original size
 */
NvcompCompressedBlock nvcomp_zstd_compress(const std::vector<uint8_t>& input, cudaStream_t stream = 0);

/**
 * Compress data using nvComp ZSTD on GPU (from device memory).
 * 
 * Input data is already on device. Compressed result is copied back to host.
 * 
 * @param d_input Device pointer to input data
 * @param input_size Size of input data in bytes
 * @param stream CUDA stream to use (default: 0)
 * @return NvcompCompressedBlock containing compressed payload and original size
 */
NvcompCompressedBlock nvcomp_zstd_compress_device(const uint8_t* d_input, size_t input_size, cudaStream_t stream = 0);

/**
 * Decompress nvComp ZSTD compressed data on GPU.
 * 
 * Input compressed data is on host, gets copied to device, decompressed on GPU,
 * and the result is copied back to host.
 * 
 * @param compressed NvcompCompressedBlock containing compressed data
 * @param stream CUDA stream to use (default: 0)
 * @return Host vector of decompressed bytes
 */
std::vector<uint8_t> nvcomp_zstd_decompress(const NvcompCompressedBlock& compressed, cudaStream_t stream = 0);

// Convenience wrappers for string compression
NvcompCompressedBlock nvcomp_zstd_compress_string(const std::string& input, cudaStream_t stream = 0);
std::string nvcomp_zstd_decompress_string(const NvcompCompressedBlock& compressed, cudaStream_t stream = 0);

// Convenience wrappers for uint32 vector compression  
NvcompCompressedBlock nvcomp_zstd_compress_uint32(const std::vector<uint32_t>& input, cudaStream_t stream = 0);
std::vector<uint32_t> nvcomp_zstd_decompress_uint32(const NvcompCompressedBlock& compressed, cudaStream_t stream = 0);

// Convenience wrappers for int32 vector compression
NvcompCompressedBlock nvcomp_zstd_compress_int32(const std::vector<int32_t>& input, cudaStream_t stream = 0);
std::vector<int32_t> nvcomp_zstd_decompress_int32(const NvcompCompressedBlock& compressed, cudaStream_t stream = 0);

// ============================================================================
// Device-resident compression (avoids unnecessary H<->D copies)
// ============================================================================

/**
 * Compress int32_t data directly from device memory.
 * Use this for data that's already on GPU (e.g., after compression rounds).
 * 
 * @param d_data Device pointer to int32_t array
 * @param count Number of int32_t elements
 * @param stream CUDA stream to use
 * @return NvcompCompressedBlock (result is on host)
 */
NvcompCompressedBlock nvcomp_zstd_compress_int32_device(
    const int32_t* d_data, size_t count, cudaStream_t stream = 0);

/**
 * Compress uint32_t data directly from device memory.
 *
 * @param d_data Device pointer to uint32_t array
 * @param count Number of uint32_t elements
 * @param stream CUDA stream to use
 * @return NvcompCompressedBlock (result is on host)
 */
NvcompCompressedBlock nvcomp_zstd_compress_uint32_device(
    const uint32_t* d_data, size_t count, cudaStream_t stream = 0);

// ============================================================================
// Device-resident decompression (avoids unnecessary D->H->D copies)
// ============================================================================

/**
 * Decompress nvComp ZSTD compressed data directly to device memory.
 *
 * Input compressed data is on host, gets copied to device, decompressed on GPU,
 * and the result stays on device (returned as device pointer).
 *
 * @param compressed NvcompCompressedBlock containing compressed data
 * @param d_output Output device pointer (will be allocated by this function)
 * @param output_size Output: size of decompressed data in bytes
 * @param stream CUDA stream to use (default: 0)
 * @note Caller is responsible for freeing d_output with cudaFree
 */
void nvcomp_zstd_decompress_to_device(
    const NvcompCompressedBlock& compressed,
    uint8_t** d_output,
    size_t* output_size,
    cudaStream_t stream = 0);

/**
 * Decompress int32_t data directly to device memory.
 *
 * @param compressed NvcompCompressedBlock containing compressed int32 data
 * @param d_output Output device pointer (will be allocated by this function)
 * @param count Output: number of int32_t elements
 * @param stream CUDA stream to use
 * @note Caller is responsible for freeing d_output with cudaFree
 */
void nvcomp_zstd_decompress_int32_to_device(
    const NvcompCompressedBlock& compressed,
    int32_t** d_output,
    size_t* count,
    cudaStream_t stream = 0);

/**
 * Decompress uint32_t data directly to device memory.
 *
 * @param compressed NvcompCompressedBlock containing compressed uint32 data
 * @param d_output Output device pointer (will be allocated by this function)
 * @param count Output: number of uint32_t elements
 * @param stream CUDA stream to use
 * @note Caller is responsible for freeing d_output with cudaFree
 */
void nvcomp_zstd_decompress_uint32_to_device(
    const NvcompCompressedBlock& compressed,
    uint32_t** d_output,
    size_t* count,
    cudaStream_t stream = 0);

} // namespace gpu_codec

#endif // CODEC_GPU_NVCOMP_CUH
