#!/bin/bash

# UFRa Build and Test Script
# This script builds the project and runs comprehensive tests

set -e  # Exit on any error

echo "=========================================="
echo "UFRa Build and Test Script"
echo "=========================================="

# Configuration
BUILD_TYPE=${1:-Release}
JOBS=${2:-$(nproc)}
ENABLE_TESTS=${3:-ON}
ENABLE_PYTHON=${4:-ON}

echo "Build Type: $BUILD_TYPE"
echo "Parallel Jobs: $JOBS"
echo "Enable Tests: $ENABLE_TESTS"
echo "Enable Python: $ENABLE_PYTHON"

# Create build directory
mkdir -p build
cd build

# Configure CMake
echo "Configuring CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DBUILD_TESTING=$ENABLE_TESTS \
    -DBUILD_PYTHON_BINDINGS=$ENABLE_PYTHON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Build project
echo "Building project..."
make -j$JOBS

# Run tests if enabled
if [ "$ENABLE_TESTS" = "ON" ]; then
    echo "Running unit tests..."
    ctest --output-on-failure --parallel $JOBS
    
    echo "Running integration tests..."
    ./bin/ufra_tests --gtest_filter="IntegrationTest.*" || true
    
    echo "Testing CLI tool..."
    ./bin/ufra_cli --help || true
fi

# Build Python bindings if enabled
if [ "$ENABLE_PYTHON" = "ON" ]; then
    echo "Building Python bindings..."
    cd ../
    python setup.py build_ext --inplace || true
    cd build
fi

# Generate coverage report if available
if command -v gcov &> /dev/null && [ "$BUILD_TYPE" = "Debug" ]; then
    echo "Generating coverage report..."
    gcov ../core/src/*.cpp || true
fi

# Summary
echo "=========================================="
echo "Build Summary:"
echo "=========================================="

if [ -f "./lib/libufra_core.so" ] || [ -f "./lib/libufra_core.dylib" ] || [ -f "./lib/ufra_core.dll" ]; then
    echo "✓ Core library built successfully"
else
    echo "✗ Core library build failed"
fi

if [ -f "./lib/UFRa.ofx" ]; then
    echo "✓ OpenFX plugin built successfully"
else
    echo "✗ OpenFX plugin build failed"
fi

if [ -f "./bin/ufra_cli" ]; then
    echo "✓ CLI tool built successfully"
else
    echo "✗ CLI tool build failed"
fi

if [ "$ENABLE_TESTS" = "ON" ]; then
    if [ -f "./bin/ufra_tests" ]; then
        echo "✓ Unit tests built successfully"
    else
        echo "✗ Unit tests build failed"
    fi
fi

echo "=========================================="
echo "Build completed!"
echo "=========================================="

# Return to original directory
cd ..