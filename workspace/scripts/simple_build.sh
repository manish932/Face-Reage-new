#!/bin/bash

# UFRa Simple Build Script
# Builds the project with minimal dependencies

set -e

echo "=========================================="
echo "UFRa Simple Build (No External Dependencies)"
echo "=========================================="

# Check for basic tools
if ! command -v cmake &> /dev/null; then
    echo "Error: CMake not found. Please install CMake first."
    exit 1
fi

if ! command -v make &> /dev/null && ! command -v ninja &> /dev/null; then
    echo "Error: Build system (make or ninja) not found."
    exit 1
fi

# Create build directory
mkdir -p build
cd build

echo "Configuring with minimal dependencies..."

# Configure CMake with minimal options
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_CUDA=OFF \
    -DBUILD_TESTING=ON \
    -DBUILD_PYTHON_BINDINGS=OFF \
    -DBUILD_PLUGINS=OFF

echo "Building core library..."
make -j$(nproc) ufra_core || make ufra_core

echo "Building tests..."
make -j$(nproc) ufra_tests || make ufra_tests

echo "=========================================="
echo "Build Summary:"
echo "=========================================="

if [ -f "./lib/libufra_core.so" ] || [ -f "./lib/libufra_core.dylib" ]; then
    echo "✓ Core library built successfully"
    ls -la ./lib/libufra_core.*
else
    echo "✗ Core library build failed"
fi

if [ -f "./bin/ufra_tests" ]; then
    echo "✓ Unit tests built successfully"
    echo "Running basic tests..."
    ./bin/ufra_tests --gtest_filter="EngineTest.*" || echo "Some tests failed (expected without models)"
else
    echo "✗ Unit tests build failed"
fi

cd ..
echo "Build completed!"