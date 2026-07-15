cmake_minimum_required(VERSION 3.10)
project(TranformerNetwork VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Ensure Release mode is default for performance
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# 2. Add Compiler Flags for Optimization and SIMD
if(MSVC)
    # Windows/MSVC specific optimizations
    add_compile_options(/O2 /arch:AVX2 /fp:fast)
else()
    # GCC/Clang specific optimizations
    # -O3: Aggressive optimization
    # -march=native: Use all instructions supported by your current CPU (critical for Eigen SIMD)
    # -ffast-math: Allows Eigen to use faster floating point math
    add_compile_options(-O3 -march=native -ffast-math)
endif()

add_executable(Network 
    src/main.cpp 
    src/MLPLayer.cpp 
    src/MLP.cpp
    src/HelperFunctions.cpp
)

target_include_directories(Network PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/libraries
)

# 3. Optional: Add a Precompiled Header (PCH) for faster compile times
# If you include <Eigen/Dense> in many files, this will speed up compilation significantly
target_precompile_headers(Network PRIVATE <Eigen/Dense>)