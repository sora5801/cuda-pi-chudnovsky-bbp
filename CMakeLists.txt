# =============================================================================
#  CMakeLists.txt  --  Cross-platform build for the CUDA pi calculator
# =============================================================================
#
#  This is the recommended way to build the project, and the easiest way to get a
#  Visual Studio solution. CMake auto-detects the compilers, the CUDA toolkit, and
#  the right MSBuild integration, so you do not have to hand-edit project files.
#
#  QUICK START (Windows, Visual Studio 2026):
#    cmake -S . -B build -G "Visual Studio 18 2026" -A x64
#    cmake --build build --config Release
#    ctest --test-dir build -C Release        # run the self-test
#  ...or simply open this folder in Visual Studio 2026 (File > Open > Folder),
#  which reads CMakePresets.json and configures automatically.
#
#  The build produces three executables:
#    pi          -- the command-line calculator (src/main.cpp)
#    pi_selftest -- the correctness test suite   (tests/self_test.cu)
#    pi_demo     -- the narrated guided tour      (demo/demo.cu)
#  all linked against one static library, `pidigits`, that holds the math.
# =============================================================================

cmake_minimum_required(VERSION 3.24)   # 3.24+ understands CUDA_ARCHITECTURES well

project(PiDigits
        VERSION 1.0.0
        DESCRIPTION "Compute pi via Chudnovsky + BBP, accelerated with CUDA"
        LANGUAGES CXX CUDA)

# --- Language standards -------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# --- GPU architecture ---------------------------------------------------------
# 75 = Turing (GeForce RTX 20-series, e.g. the RTX 2080 this was developed on).
# Override on the command line for other GPUs, e.g.:
#   -DCMAKE_CUDA_ARCHITECTURES=89   (Ada / RTX 40-series)
#   -DCMAKE_CUDA_ARCHITECTURES=native (auto-detect the local GPU; CMake 3.24+)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# The CUDA runtime library (cudart) -- linked into consumers of the static lib.
find_package(CUDAToolkit REQUIRED)

# --- The core math library ----------------------------------------------------
# Mixing .cpp (host) and .cu (host+device) sources in one library is fine: CMake
# compiles each with the right compiler (cl.exe vs nvcc) automatically.
add_library(pidigits STATIC
    src/bignum.cpp
    src/bignum_div.cpp
    src/ntt_cpu.cpp
    src/ntt_cuda.cu
    src/chudnovsky.cpp
    src/bbp_cpu.cpp
    src/bbp_cuda.cu)

target_include_directories(pidigits PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# Anything linking pidigits also needs the CUDA runtime.
target_link_libraries(pidigits PUBLIC CUDA::cudart)

# Enable a faster math path for the host compiler in Release builds.
if(MSVC)
    target_compile_options(pidigits PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/O2>)
endif()

# --- Executables --------------------------------------------------------------
add_executable(pi          src/main.cpp)
add_executable(pi_selftest tests/self_test.cu)
add_executable(pi_demo     demo/demo.cu)

foreach(exe pi pi_selftest pi_demo)
    target_link_libraries(${exe} PRIVATE pidigits)
endforeach()

# --- Testing ------------------------------------------------------------------
enable_testing()
add_test(NAME selftest COMMAND pi_selftest)

# --- Friendly summary ---------------------------------------------------------
message(STATUS "PiDigits: CUDA architectures = ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "PiDigits: CUDA toolkit       = ${CUDAToolkit_VERSION}")
