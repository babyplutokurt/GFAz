cmake_minimum_required(VERSION 3.30)
project(GfaCompression LANGUAGES C CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Language standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Feature toggles
option(ENABLE_CUDA "Enable GPU acceleration with CUDA" OFF)
option(ENABLE_PROFILING "Enable gperftools CPU profiling" OFF)
option(GFAZ_USE_SYSTEM_ZSTD "Use system-installed Zstd instead of vendored copy" OFF)
set(CUDA_PATH "" CACHE PATH "Root directory of the CUDA Toolkit (e.g., /usr/local/cuda)")

# Find Python with development headers (required for pybind11 + optional nvComp auto-detect)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Python3 found: ${Python3_EXECUTABLE}")
message(STATUS "Python3 include: ${Python3_INCLUDE_DIRS}")

# Tell pybind11 to use the Python we found
set(PYBIND11_FINDPYTHON ON)

# Optional CUDA configuration
set(NVCOMP_FOUND FALSE)
if(ENABLE_CUDA)
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

  # nvComp (optional): auto-detect root from Python package unless provided
  if(NOT DEFINED NVCOMP_ROOT)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "import nvidia.libnvcomp; import os; print(os.path.dirname(nvidia.libnvcomp.__file__))"
      OUTPUT_VARIABLE NVCOMP_ROOT_AUTO
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    if(NVCOMP_ROOT_AUTO)
      set(NVCOMP_ROOT "${NVCOMP_ROOT_AUTO}")
    endif()
  endif()

  find_path(
    NVCOMP_INCLUDE_DIR
    NAMES nvcomp.h nvcomp/nvcomp.h
    HINTS "${NVCOMP_ROOT}" $ENV{CONDA_PREFIX}
    PATH_SUFFIXES include
  )

  find_library(
    NVCOMP_LIB
    NAMES nvcomp
    HINTS "${NVCOMP_ROOT}" $ENV{CONDA_PREFIX}
    PATH_SUFFIXES lib lib64
  )

  if(NVCOMP_INCLUDE_DIR AND NVCOMP_LIB)
    message(STATUS "nvComp found:")
    message(STATUS "  Include: ${NVCOMP_INCLUDE_DIR}")
    message(STATUS "  Library: ${NVCOMP_LIB}")

    add_library(nvcomp SHARED IMPORTED)
    set_target_properties(
      nvcomp
      PROPERTIES
        IMPORTED_LOCATION "${NVCOMP_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${NVCOMP_INCLUDE_DIR}"
    )

    set(NVCOMP_FOUND TRUE)
  else()
    message(WARNING "nvComp not found (headers or library missing).")
    if(NVCOMP_LIB)
      message(STATUS "  Library found at: ${NVCOMP_LIB}")
      message(STATUS "  But headers (nvcomp.h) were NOT found.")
    endif()
    message(STATUS "GPU ZSTD compression will fall back to CPU ZSTD.")
    message(STATUS "Install via Pip (includes headers): pip install nvidia-libnvcomp-cu12")
    message(STATUS "Or ensure your Conda package includes development headers.")
  endif()
else()
  message(STATUS "CUDA disabled - building CPU-only version")
  message(STATUS "To enable GPU support, use: cmake -DENABLE_CUDA=ON ..")
endif()

option(BUILD_PYTHON_BINDINGS "Build Pybind11 Python bindings" ON)
option(BUILD_CLI "Build the gfaz CLI executable" ON)

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

  add_library(zstd_static STATIC
    ${ZSTD_COMMON_SRC}
    ${ZSTD_COMPRESS_SRC}
    ${ZSTD_DECOMPRESS_SRC}
  )
  target_include_directories(zstd_static PUBLIC extern/zstd/lib)
  target_compile_definitions(zstd_static PRIVATE ZSTD_DISABLE_ASM)

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
  src/gfa_parser.cpp
  src/rule_generator.cpp
  src/path_encoder.cpp
  src/rule_processor.cpp
  src/add_haplotypes_workflow.cpp
  src/extraction_workflow.cpp
  src/compression_workflow.cpp
  src/decompression_workflow.cpp
  src/codec.cpp
  src/serialization.cpp
  src/gfa_writer.cpp
  src/gpu/gfa_graph_gpu.cpp
)

set(GFA_CUDA_SOURCES
  src/gpu/codec_gpu.cu
  src/gpu/compression_workflow_gpu.cu
  src/gpu/decompression_workflow_gpu.cu
  src/gpu/codec_gpu_nvcomp.cu
  src/gpu/serialization_gpu.cpp
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

    if(NVCOMP_FOUND)
      target_compile_definitions(${target_name} PRIVATE ENABLE_NVCOMP)
      target_link_libraries(${target_name} PRIVATE nvcomp)
    endif()

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
  add_library(gfa_compression MODULE src/bindings.cpp)
  set_target_properties(gfa_compression PROPERTIES PREFIX "")
  target_link_libraries(gfa_compression PRIVATE pybind11::module gfa_compression_core)
  configure_gfa_target(gfa_compression)
endif()

if(BUILD_CLI)
  # CLI executable
  add_executable(gfaz src/gfaz_cli.cpp)
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
