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

# Module source lists. GFAz is two modules on a shared foundation:
#   core     - shared model/codec/utils (no compress- or compute-specific code)
#   compress - the .gfa <-> .gfaz compressor (+ optional GPU backend)
#   compute  - the compute engine that runs directly on .gfaz
# compress and compute each depend only on core; they never depend on each other.
set(GFAZ_CORE_SOURCES
  src/core/codec/codec.cpp
  src/core/codec/serialization.cpp
  src/core/utils/runtime_utils.cpp
)

set(GFAZ_COMPRESS_SOURCES
  src/compress/io/gfa_parser.cpp
  src/compress/io/gfa_decoder.cpp
  src/compress/io/gfa_writer.cpp
  src/compress/io/gfa_write_utils.cpp
  src/compress/grammar/rule_generator.cpp
  src/compress/grammar/path_encoder.cpp
  src/compress/compression_workflow.cpp
  src/compress/compression_utils.cpp
  src/compress/compression_debug.cpp
  src/compress/decompression_workflow.cpp
  src/compress/decompression_debug.cpp
  src/compress/extraction_workflow.cpp
  src/compress/add_haplotypes_workflow.cpp
)

set(GFAZ_COMPUTE_SOURCES
  src/compute/growth_workflow.cpp
  src/compute/pav_workflow.cpp
  src/compute/similarity_workflow.cpp
  src/compute/stats_workflow.cpp
  src/compute/depth_workflow.cpp
  src/compute/deconstruct_workflow.cpp
  src/compute/snarl_finder.cpp
  src/compute/traversal_query.cpp
)

# GPU backend (part of the compress module; built only when ENABLE_CUDA).
set(GFAZ_COMPRESS_CUDA_SOURCES
  # GPU backend core + host-side helpers
  src/compress/gpu/core/gfa_graph_gpu.cpp
  src/compress/gpu/core/path_chunk_planner.cpp
  src/compress/gpu/io/gfa_writer_gpu_direct.cpp

  # GPU backend core
  src/compress/gpu/core/codec_gpu.cu
  src/compress/gpu/core/serialization_gpu.cpp

  # GPU compression pipeline
  src/compress/gpu/compression/compression_workflow_gpu.cu
  src/compress/gpu/compression/traversal_compression_gpu.cu
  src/compress/gpu/compression/path_compression_gpu_legacy.cu
  src/compress/gpu/compression/path_compression_gpu_rolling.cu
  src/compress/gpu/compression/path_rulebook_gpu.cu
  src/compress/gpu/compression/path_rule_remap_gpu.cu
  src/compress/gpu/compression/metadata_codec_gpu.cu

  # GPU decompression pipeline
  src/compress/gpu/decompression/decompression_workflow_gpu.cu
  src/compress/gpu/decompression/decompression_primitives_gpu.cu
  src/compress/gpu/decompression/traversal_decode_gpu.cu
  src/compress/gpu/decompression/path_decompression_gpu_plan.cu
  src/compress/gpu/decompression/path_decompression_gpu_legacy.cu
  src/compress/gpu/decompression/path_decompression_gpu_rolling.cu
  src/compress/gpu/decompression/path_expand_gpu.cu

  # GPU direct writer
  src/compress/gpu/io/gfa_writer_gpu.cu
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

# Static libraries — one per module. compress and compute both depend on core;
# the boundary is enforced at link time (compute does not link compress, so any
# accidental use of compressor symbols from compute fails to link).
add_library(gfaz_core STATIC ${GFAZ_CORE_SOURCES})
set_target_properties(gfaz_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
configure_gfa_target(gfaz_core)
# Public so every consumer of any module gets the include root.
target_include_directories(gfaz_core PUBLIC include)

add_library(gfaz_compress STATIC ${GFAZ_COMPRESS_SOURCES})
set_target_properties(gfaz_compress PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(ENABLE_CUDA)
  target_sources(gfaz_compress PRIVATE ${GFAZ_COMPRESS_CUDA_SOURCES})
endif()
configure_gfa_target(gfaz_compress)
target_link_libraries(gfaz_compress PUBLIC gfaz_core)

add_library(gfaz_compute STATIC ${GFAZ_COMPUTE_SOURCES})
set_target_properties(gfaz_compute PROPERTIES POSITION_INDEPENDENT_CODE ON)
configure_gfa_target(gfaz_compute)
target_link_libraries(gfaz_compute PUBLIC gfaz_core)

if(BUILD_PYTHON_BINDINGS)
  # Python extension module (import name `gfa_compression`).
  add_library(gfa_compression MODULE src/python/bindings.cpp)
  set_target_properties(gfa_compression PROPERTIES PREFIX "")
  target_link_libraries(gfa_compression PRIVATE pybind11::module
    gfaz_compress gfaz_compute gfaz_core)
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
    src/cli/growth_command.cpp
    src/cli/pav_command.cpp
    src/cli/similarity_command.cpp
    src/cli/stats_command.cpp
    src/cli/depth_command.cpp
    src/cli/deconstruct_command.cpp
  )
  set_target_properties(gfaz PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build/bin"
  )
  target_link_libraries(gfaz PRIVATE gfaz_compress gfaz_compute gfaz_core)
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

  add_executable(gfaz_front_door_api_test
    tests/cpp/test_gfa_decoder.cpp
  )
  target_link_libraries(gfaz_front_door_api_test PRIVATE
    gfaz_compress gfaz_core)
  configure_gfa_target(gfaz_front_door_api_test)

  enable_testing()
  add_test(
    NAME gfaz_front_door_api
    COMMAND gfaz_front_door_api_test
      ${CMAKE_SOURCE_DIR}/tests/fixtures/front_door_api_fixture.gfa
      ${CMAKE_BINARY_DIR}/front_door_api_test.gfaz
  )

  # Hermetic CPU-only test suite (CLI regressions + golden-file concordance).
  # A single CTest entry delegates to the Python harness so its PASS/SKIP/FAIL
  # logic stays the source of truth. Binding-dependent round-trip suites are run
  # separately. Concordance goldens are regenerated via scripts/gen_golden.py.
  find_program(PYTHON3_EXECUTABLE NAMES python3 python)
  if(PYTHON3_EXECUTABLE)
    enable_testing()
    add_test(
      NAME gfaz_tests
      COMMAND ${PYTHON3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/run_all.py
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
  endif()
endif()
