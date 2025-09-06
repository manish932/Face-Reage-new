#!/bin/bash

# UFRa Dependencies Setup Script
# Installs necessary build dependencies

set -e

echo "=========================================="
echo "UFRa Dependencies Setup"
echo "=========================================="

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi

echo "Detected OS: $OS"

# Install dependencies based on OS
if [ "$OS" = "linux" ]; then
    echo "Installing Linux dependencies..."
    
    # Update package manager
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y \
            build-essential \
            cmake \
            g++ \
            libopencv-dev \
            pkg-config \
            python3 \
            python3-pip \
            python3-dev \
            git
            
        # Optional CUDA (if available)
        if command -v nvcc &> /dev/null; then
            echo "CUDA found, installing CUDA development packages..."
            sudo apt-get install -y cuda-toolkit-dev || echo "CUDA dev packages not available"
        fi
        
    elif command -v yum &> /dev/null; then
        sudo yum groupinstall -y "Development Tools"
        sudo yum install -y cmake opencv-devel pkgconfig python3 python3-pip python3-devel
        
    elif command -v pacman &> /dev/null; then
        sudo pacman -S --noconfirm base-devel cmake opencv pkgconf python python-pip
    fi
    
elif [ "$OS" = "macos" ]; then
    echo "Installing macOS dependencies..."
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    # Install dependencies
    brew install cmake opencv pkg-config python3
    
    # Install Xcode command line tools if not present
    xcode-select --install 2>/dev/null || echo "Xcode command line tools already installed"
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install numpy opencv-python pybind11

# Verify installations
echo "=========================================="
echo "Verifying installations..."
echo "=========================================="

# Check CMake
if command -v cmake &> /dev/null; then
    echo "✓ CMake: $(cmake --version | head -n1)"
else
    echo "✗ CMake not found"
fi

# Check C++ compiler
if command -v g++ &> /dev/null; then
    echo "✓ G++: $(g++ --version | head -n1)"
elif command -v clang++ &> /dev/null; then
    echo "✓ Clang++: $(clang++ --version | head -n1)"
else
    echo "✗ C++ compiler not found"
fi

# Check Python
if command -v python3 &> /dev/null; then
    echo "✓ Python: $(python3 --version)"
else
    echo "✗ Python3 not found"
fi

# Check OpenCV
python3 -c "import cv2; print('✓ OpenCV Python:', cv2.__version__)" 2>/dev/null || echo "✗ OpenCV Python not found"

# Check CUDA (optional)
if command -v nvcc &> /dev/null; then
    echo "✓ CUDA: $(nvcc --version | grep 'release' | head -n1)"
else
    echo "○ CUDA not found (optional)"
fi

echo "=========================================="
echo "Dependencies setup complete!"
echo "=========================================="