#include "gpu/codec_gpu_nvcomp.cuh"
#include <stdexcept>
#include <iostream>
#include <cstring>

#ifdef ENABLE_NVCOMP
#include <nvcomp/zstd.hpp>
#include <nvcomp/nvcompManager.hpp>
#endif

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + \
                                     std::to_string(__LINE__) + " - " + \
                                     cudaGetErrorString(err)); \
        } \
    } while(0)

namespace gpu_codec {

#ifdef ENABLE_NVCOMP

NvcompCompressedBlock nvcomp_zstd_compress(const std::vector<uint8_t>& input, cudaStream_t stream) {
    if (input.empty()) {
        return NvcompCompressedBlock{{}, 0};
    }
    
    // Allocate device memory for input
    uint8_t* d_input = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, input.size()));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), input.size(), cudaMemcpyHostToDevice));
    
    // Compress from device memory
    NvcompCompressedBlock result = nvcomp_zstd_compress_device(d_input, input.size(), stream);
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    
    return result;
}

NvcompCompressedBlock nvcomp_zstd_compress_device(const uint8_t* d_input, size_t input_size, cudaStream_t stream) {
    NvcompCompressedBlock result;
    result.original_size = input_size;
    
    if (input_size == 0) {
        return result;
    }
    
    // Use 1MB chunks for better performance with large data
    const size_t chunk_size = 1 * 1024 * 1024;
    
    // Create ZstdManager
    nvcomp::ZstdManager manager(
        chunk_size,
        nvcompBatchedZstdCompressDefaultOpts,
        nvcompBatchedZstdDecompressDefaultOpts,
        stream,
        nvcomp::NoComputeNoVerify,
        nvcomp::BitstreamKind::NVCOMP_NATIVE
    );
    
    // Configure compression
    auto comp_config = manager.configure_compression(input_size);
    
    // Get required output buffer size
    size_t max_comp_size = comp_config.max_compressed_buffer_size;
    
    // Allocate device memory for compressed output
    uint8_t* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_output, max_comp_size));
    
    // Compress
    size_t comp_size = 0;
    manager.compress(d_input, d_output, comp_config, &comp_size);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Copy result to host
    result.payload.resize(comp_size);
    CUDA_CHECK(cudaMemcpy(result.payload.data(), d_output, comp_size, cudaMemcpyDeviceToHost));
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_output));
    
    return result;
}

std::vector<uint8_t> nvcomp_zstd_decompress(const NvcompCompressedBlock& compressed, cudaStream_t stream) {
    if (compressed.payload.empty() || compressed.original_size == 0) {
        return std::vector<uint8_t>();
    }
    
    // Use 1MB chunks (same as compression)
    const size_t chunk_size = 1 * 1024 * 1024;
    
    // Create ZstdManager
    nvcomp::ZstdManager manager(
        chunk_size,
        nvcompBatchedZstdCompressDefaultOpts,
        nvcompBatchedZstdDecompressDefaultOpts,
        stream,
        nvcomp::NoComputeNoVerify,
        nvcomp::BitstreamKind::NVCOMP_NATIVE
    );
    
    // Allocate device memory for compressed input
    uint8_t* d_comp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_comp, compressed.payload.size()));
    CUDA_CHECK(cudaMemcpy(d_comp, compressed.payload.data(), compressed.payload.size(), cudaMemcpyHostToDevice));
    
    // Configure decompression
    auto decomp_config = manager.configure_decompression(d_comp);
    
    // Get expected output size
    size_t decomp_size = decomp_config.decomp_data_size;
    
    // Allocate device memory for decompressed output
    uint8_t* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_output, decomp_size));
    
    // Decompress
    manager.decompress(&d_output, &d_comp, std::vector<nvcomp::DecompressionConfig>{decomp_config});
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Copy result to host
    std::vector<uint8_t> result(decomp_size);
    CUDA_CHECK(cudaMemcpy(result.data(), d_output, decomp_size, cudaMemcpyDeviceToHost));
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_comp));
    CUDA_CHECK(cudaFree(d_output));
    
    return result;
}

#else // ENABLE_NVCOMP not defined - fallback stubs

NvcompCompressedBlock nvcomp_zstd_compress(const std::vector<uint8_t>& input, cudaStream_t) {
    throw std::runtime_error("nvComp not available. Rebuild with nvComp support.");
}

NvcompCompressedBlock nvcomp_zstd_compress_device(const uint8_t*, size_t, cudaStream_t) {
    throw std::runtime_error("nvComp not available. Rebuild with nvComp support.");
}

std::vector<uint8_t> nvcomp_zstd_decompress(const NvcompCompressedBlock&, cudaStream_t) {
    throw std::runtime_error("nvComp not available. Rebuild with nvComp support.");
}

#endif // ENABLE_NVCOMP

// ============================================================================
// Convenience wrappers (work with or without nvComp)
// ============================================================================

NvcompCompressedBlock nvcomp_zstd_compress_string(const std::string& input, cudaStream_t stream) {
    std::vector<uint8_t> bytes(input.begin(), input.end());
    return nvcomp_zstd_compress(bytes, stream);
}

std::string nvcomp_zstd_decompress_string(const NvcompCompressedBlock& compressed, cudaStream_t stream) {
    auto bytes = nvcomp_zstd_decompress(compressed, stream);
    return std::string(bytes.begin(), bytes.end());
}

NvcompCompressedBlock nvcomp_zstd_compress_uint32(const std::vector<uint32_t>& input, cudaStream_t stream) {
    std::vector<uint8_t> bytes(
        reinterpret_cast<const uint8_t*>(input.data()),
        reinterpret_cast<const uint8_t*>(input.data() + input.size())
    );
    return nvcomp_zstd_compress(bytes, stream);
}

std::vector<uint32_t> nvcomp_zstd_decompress_uint32(const NvcompCompressedBlock& compressed, cudaStream_t stream) {
    auto bytes = nvcomp_zstd_decompress(compressed, stream);
    size_t count = bytes.size() / sizeof(uint32_t);
    std::vector<uint32_t> result(count);
    std::memcpy(result.data(), bytes.data(), bytes.size());
    return result;
}

NvcompCompressedBlock nvcomp_zstd_compress_int32(const std::vector<int32_t>& input, cudaStream_t stream) {
    std::vector<uint8_t> bytes(
        reinterpret_cast<const uint8_t*>(input.data()),
        reinterpret_cast<const uint8_t*>(input.data() + input.size())
    );
    return nvcomp_zstd_compress(bytes, stream);
}

std::vector<int32_t> nvcomp_zstd_decompress_int32(const NvcompCompressedBlock& compressed, cudaStream_t stream) {
    auto bytes = nvcomp_zstd_decompress(compressed, stream);
    size_t count = bytes.size() / sizeof(int32_t);
    std::vector<int32_t> result(count);
    std::memcpy(result.data(), bytes.data(), bytes.size());
    return result;
}

// ============================================================================
// Device-resident compression (avoids unnecessary H<->D copies)
// ============================================================================

NvcompCompressedBlock nvcomp_zstd_compress_int32_device(
    const int32_t* d_data, size_t count, cudaStream_t stream) {
    // Reinterpret int32_t* as uint8_t* and call the low-level device function
    return nvcomp_zstd_compress_device(
        reinterpret_cast<const uint8_t*>(d_data),
        count * sizeof(int32_t),
        stream
    );
}

NvcompCompressedBlock nvcomp_zstd_compress_uint32_device(
    const uint32_t* d_data, size_t count, cudaStream_t stream) {
    // Reinterpret uint32_t* as uint8_t* and call the low-level device function
    return nvcomp_zstd_compress_device(
        reinterpret_cast<const uint8_t*>(d_data),
        count * sizeof(uint32_t),
        stream
    );
}

// ============================================================================
// Device-resident decompression (avoids unnecessary D->H->D copies)
// ============================================================================

#ifdef ENABLE_NVCOMP

void nvcomp_zstd_decompress_to_device(
    const NvcompCompressedBlock& compressed,
    uint8_t** d_output,
    size_t* output_size,
    cudaStream_t stream) {

    if (compressed.payload.empty() || compressed.original_size == 0) {
        *d_output = nullptr;
        *output_size = 0;
        return;
    }

    const size_t chunk_size = 1 * 1024 * 1024;
    
    nvcomp::ZstdManager manager(
        chunk_size,
        nvcompBatchedZstdCompressDefaultOpts,
        nvcompBatchedZstdDecompressDefaultOpts,
        stream,
        nvcomp::NoComputeNoVerify,
        nvcomp::BitstreamKind::NVCOMP_NATIVE
    );

    // Allocate device memory for compressed input
    uint8_t* d_comp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_comp, compressed.payload.size()));
    CUDA_CHECK(cudaMemcpyAsync(d_comp, compressed.payload.data(),
                                compressed.payload.size(),
                                cudaMemcpyHostToDevice, stream));

    // Configure decompression
    auto decomp_config = manager.configure_decompression(d_comp);
    size_t decomp_size = decomp_config.decomp_data_size;

    // Allocate device memory for decompressed output
    CUDA_CHECK(cudaMalloc(d_output, decomp_size));

    // Decompress
    manager.decompress(d_output, &d_comp, std::vector<nvcomp::DecompressionConfig>{decomp_config});
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Cleanup compressed buffer
    CUDA_CHECK(cudaFree(d_comp));

    *output_size = decomp_size;
}

#else // ENABLE_NVCOMP not defined

void nvcomp_zstd_decompress_to_device(
    const NvcompCompressedBlock&,
    uint8_t**,
    size_t*,
    cudaStream_t) {
    throw std::runtime_error("nvComp not available. Rebuild with nvComp support.");
}

#endif // ENABLE_NVCOMP

void nvcomp_zstd_decompress_int32_to_device(
    const NvcompCompressedBlock& compressed,
    int32_t** d_output,
    size_t* count,
    cudaStream_t stream) {

    uint8_t* d_bytes = nullptr;
    size_t byte_size = 0;

    nvcomp_zstd_decompress_to_device(compressed, &d_bytes, &byte_size, stream);

    *d_output = reinterpret_cast<int32_t*>(d_bytes);
    *count = byte_size / sizeof(int32_t);
}

void nvcomp_zstd_decompress_uint32_to_device(
    const NvcompCompressedBlock& compressed,
    uint32_t** d_output,
    size_t* count,
    cudaStream_t stream) {

    uint8_t* d_bytes = nullptr;
    size_t byte_size = 0;

    nvcomp_zstd_decompress_to_device(compressed, &d_bytes, &byte_size, stream);

    *d_output = reinterpret_cast<uint32_t*>(d_bytes);
    *count = byte_size / sizeof(uint32_t);
}

} // namespace gpu_codec
