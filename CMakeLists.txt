cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
project(GfaCompression LANGUAGES C CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Language standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Feature toggles
option(ENABLE_CUDA "Enable GPU acceleration with CUDA" OFF)
option(ENABLE_PROFILING "Enable gperftools CPU profiling" OFF)
option(GFAZ_USE_SYSTEM_ZSTD "Use system-installed Zstd instead of vendored copy" ON)
option(BUILD_PYTHON_BINDINGS "Build Pybind11 Python bindings" ON)
option(BUILD_CLI "Build the gfaz CLI executable" ON)
set(CUDA_PATH "" CACHE PATH "Root directory of the CUDA Toolkit (e.g., /usr/local/cuda)")

# Python is only required for bindings.
if(BUILD_PYTHON_BINDINGS)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
  message(STATUS "Python3 found: ${Python3_EXECUTABLE}")
  message(STATUS "Python3 include: ${Python3_INCLUDE_DIRS}")

  # Tell pybind11 to use the Python we found
  set(PYBIND11_FINDPYTHON ON)
endif()

# Optional CUDA configuration
if(ENABLE_CUDA)
  if(CMAKE_VERSION VERSION_LESS 3.23.1)
    message(FATAL_ERROR "ENABLE_CUDA=ON requires CMake 3.23.1 or newer because extern/cuCollections enforces that minimum.")
  endif()

  # If CUDA_PATH is provided, use it to find nvcc
  if(CUDA_PATH)
    if(EXISTS "${CUDA_PATH}/bin/nvcc")
      set(CMAKE_CUDA_COMPILER "${CUDA_PATH}/bin/nvcc")
      message(STATUS "Using CUDA compiler from CUDA_PATH: ${CMAKE_CUDA_COMPILER}")
    else()
      message(WARNING "CUDA_PATH specified ('${CUDA_PATH}'), but bin/nvcc not found there.")
    endif()
  elseif(NOT CMAKE_CUDA_COMPILER)
    # Try standard nvcc locations
    find_program(
      CUDA_COMPILER_AUTO
      NAMES nvcc
      PATHS /usr/local/cuda /usr/local/cuda-13.0 /usr/local/cuda-12.8 /usr/local/cuda-12.0
      PATH_SUFFIXES bin
    )
    if(CUDA_COMPILER_AUTO)
      set(CMAKE_CUDA_COMPILER "${CUDA_COMPILER_AUTO}")
      message(STATUS "Auto-detected CUDA compiler: ${CMAKE_CUDA_COMPILER}")
    endif()
  endif()

  enable_language(CUDA)
  set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_ARCHITECTURES "native")

  find_package(CUDAToolkit REQUIRED)
  message(STATUS "CUDA enabled - version ${CUDAToolkit_VERSION}")
  message(STATUS "CUDA include dirs: ${CUDAToolkit_INCLUDE_DIRS}")

  # Add cuCollections
  set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
  set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  add_subdirectory(extern/cuCollections)
else()
  message(STATUS "CUDA disabled - building CPU-only version")
  message(STATUS "To enable GPU support, use: cmake -DENABLE_CUDA=ON ..")
endif()

# Third-party dependencies
if(BUILD_PYTHON_BINDINGS)
  add_subdirectory(extern/pybind11)
endif()

# Configure ZSTD dependency
if(GFAZ_USE_SYSTEM_ZSTD)
  find_package(ZSTD QUIET)
  add_library(gfaz_zstd INTERFACE)

  if(ZSTD_FOUND AND TARGET ZSTD::ZSTD)
    message(STATUS "Using system ZSTD via find_package(ZSTD)")
    target_link_libraries(gfaz_zstd INTERFACE ZSTD::ZSTD)
  else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(LIBZSTD REQUIRED IMPORTED_TARGET libzstd)
    message(STATUS "Using system ZSTD via pkg-config")
    target_link_libraries(gfaz_zstd INTERFACE PkgConfig::LIBZSTD)
  endif()
else()
  message(STATUS "Using vendored ZSTD")
  file(GLOB ZSTD_COMMON_SRC extern/zstd/lib/common/*.c)
  file(GLOB ZSTD_COMPRESS_SRC extern/zstd/lib/compress/*.c)
  file(GLOB ZSTD_DECOMPRESS_SRC extern/zstd/lib/decompress/*.c)
  set(ZSTD_THREADING_SRC extern/zstd/lib/common/pool.c
                         extern/zstd/lib/common/threading.c)

  add_library(zstd_static STATIC
    ${ZSTD_COMMON_SRC}
    ${ZSTD_COMPRESS_SRC}
    ${ZSTD_DECOMPRESS_SRC}
    ${ZSTD_THREADING_SRC}
  )
  target_include_directories(zstd_static PUBLIC extern/zstd/lib)
  target_compile_definitions(zstd_static PRIVATE ZSTD_DISABLE_ASM ZSTD_MULTITHREAD)
  find_package(Threads REQUIRED)
  target_link_libraries(zstd_static PRIVATE Threads::Threads)

  add_library(gfaz_zstd INTERFACE)
  target_link_libraries(gfaz_zstd INTERFACE zstd_static)
endif()

# OpenMP is optional
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found - enabling parallel k-mer collection")
else()
  message(WARNING "OpenMP not found - k-mer collection will be single-threaded")
endif()

# Shared source lists
set(GFA_CORE_SOURCES
  src/io/gfa_parser.cpp
  src/grammar/rule_generator.cpp
  src/grammar/path_encoder.cpp
  src/grammar/rule_processor.cpp
  src/workflows/add_haplotypes_workflow.cpp
  src/workflows/extraction_workflow.cpp
  src/workflows/compression_workflow.cpp
  src/workflows/compression_debug.cpp
  src/workflows/decompression_workflow.cpp
  src/workflows/decompression_debug.cpp
  src/utils/runtime_utils.cpp
  src/codec/codec.cpp
  src/codec/serialization.cpp
  src/io/gfa_writer.cpp
  src/io/gfa_write_utils.cpp
  src/workflows/compression_utils.cpp
)

set(GFA_CUDA_SOURCES
  # GPU backend core + host-side helpers
  src/gpu/core/gfa_graph_gpu.cpp
  src/gpu/core/path_chunk_planner.cpp
  src/gpu/io/gfa_writer_gpu_direct.cpp

  # GPU backend core
  src/gpu/core/codec_gpu.cu
  src/gpu/core/serialization_gpu.cpp

  # GPU compression pipeline
  src/gpu/compression/compression_workflow_gpu.cu
  src/gpu/compression/traversal_compression_gpu.cu
  src/gpu/compression/path_compression_gpu_legacy.cu
  src/gpu/compression/path_compression_gpu_rolling.cu
  src/gpu/compression/path_rulebook_gpu.cu
  src/gpu/compression/path_rule_remap_gpu.cu
  src/gpu/compression/metadata_codec_gpu.cu

  # GPU decompression pipeline
  src/gpu/decompression/decompression_workflow_gpu.cu
  src/gpu/decompression/decompression_primitives_gpu.cu
  src/gpu/decompression/traversal_decode_gpu.cu
  src/gpu/decompression/path_decompression_gpu_plan.cu
  src/gpu/decompression/path_decompression_gpu_legacy.cu
  src/gpu/decompression/path_decompression_gpu_rolling.cu
  src/gpu/decompression/path_expand_gpu.cu

  # GPU direct writer
  src/gpu/io/gfa_writer_gpu.cu
)

function(configure_gfa_target target_name)
  target_include_directories(${target_name} PRIVATE
    include
    extern/robin_hood
  )

  target_link_libraries(${target_name} PRIVATE gfaz_zstd)

  if(OpenMP_CXX_FOUND)
    # In odgi BUILD_STATIC=ON builds, linking OpenMP::OpenMP_CXX can inject a
    # direct path to libgomp.so, which breaks fully static linking.
    # odgi already propagates OpenMP flags globally, so skip the imported target
    # in that mode and let the top-level static link resolve libgomp.a.
    if(BUILD_STATIC)
      message(STATUS "Skipping OpenMP imported target link for ${target_name} in BUILD_STATIC mode")
    else()
      target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)
    endif()
  endif()

  if(ENABLE_CUDA)
    target_compile_definitions(${target_name} PRIVATE ENABLE_CUDA)
    target_link_libraries(${target_name} PRIVATE CUDA::cudart cuco::cuco)

    target_compile_options(${target_name} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
      $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
    )
  endif()

  if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(${target_name} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
      $<$<COMPILE_LANGUAGE:CUDA>:-Wall -Wextra>
    )
  endif()
endfunction()

# Core Static Library
add_library(gfa_compression_core STATIC ${GFA_CORE_SOURCES})
set_target_properties(gfa_compression_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(ENABLE_CUDA)
  target_sources(gfa_compression_core PRIVATE ${GFA_CUDA_SOURCES})
endif()
configure_gfa_target(gfa_compression_core)

# Target include directories so consumers can use it
target_include_directories(gfa_compression_core PUBLIC include)

if(BUILD_PYTHON_BINDINGS)
  # Python extension module
  add_library(gfa_compression MODULE src/python/bindings.cpp)
  set_target_properties(gfa_compression PROPERTIES PREFIX "")
  target_link_libraries(gfa_compression PRIVATE pybind11::module gfa_compression_core)
  configure_gfa_target(gfa_compression)
endif()

if(BUILD_CLI)
  # CLI executable
  add_executable(gfaz
    src/cli/gfaz_cli.cpp
    src/cli/common.cpp
    src/cli/compress_command.cpp
    src/cli/decompress_command.cpp
    src/cli/extract_path_command.cpp
    src/cli/extract_walk_command.cpp
    src/cli/add_haplotypes_command.cpp
  )
  set_target_properties(gfaz PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build/bin"
  )
  target_link_libraries(gfaz PRIVATE gfa_compression_core)
  configure_gfa_target(gfaz)

  # Profiling support for CLI only
  if(ENABLE_PROFILING)
    message(STATUS "Profiling enabled - linking with gperftools profiler")

    find_library(PROFILER_LIB profiler PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)
    if(PROFILER_LIB)
      message(STATUS "Found profiler at: ${PROFILER_LIB}")
      target_compile_options(gfaz PRIVATE -g -fno-omit-frame-pointer)
      target_link_libraries(gfaz PRIVATE ${PROFILER_LIB})
    else()
      message(FATAL_ERROR "gperftools profiler library not found. Install with: sudo apt install libgoogle-perftools-dev")
    endif()
  endif()
endif()
