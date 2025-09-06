cmake_minimum_required(VERSION 3.16)

# Check for C++ compiler first
if(NOT CMAKE_CXX_COMPILER)
    find_program(CMAKE_CXX_COMPILER
        NAMES g++ clang++ c++
        PATHS /usr/bin /usr/local/bin
        DOC "C++ compiler"
    )
endif()

# Initialize project with C++ only first
project(UFRa VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optional CUDA support
option(ENABLE_CUDA "Enable CUDA support" OFF)
if(ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit QUIET)
        if(CUDAToolkit_FOUND)
            set(CUDA_ENABLED TRUE)
            message(STATUS "CUDA support enabled")
        else()
            message(WARNING "CUDA toolkit not found, disabling CUDA support")
            set(CUDA_ENABLED FALSE)
        endif()
    else()
        message(WARNING "CUDA compiler not found, disabling CUDA support")
        set(CUDA_ENABLED FALSE)
    endif()
else()
    set(CUDA_ENABLED FALSE)
    message(STATUS "CUDA support disabled")
endif()

# Find optional packages
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
else()
    message(WARNING "OpenCV not found, some features will be limited")
endif()

find_package(PkgConfig QUIET)

# Add subdirectories conditionally
add_subdirectory(core)

if(BUILD_PLUGINS)
    add_subdirectory(plugins)
endif()

if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(python_bindings)
endif()

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Compiler-specific options
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")
endif()

# CUDA flags (only if CUDA is enabled)
if(CUDA_ENABLED)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
endif()

# Build options
option(BUILD_TESTING "Build tests" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_PLUGINS "Build plugins" ON)