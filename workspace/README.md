# Universal Face Re-Aging (UFRa) VFX Plugin

A cross-platform VFX plugin that performs AI-powered face re-aging (both aging and de-aging) with production reliability, identity preservation, temporal stability, and precise artist controls.

## Features

### Core Capabilities
- **Identity-preserving face re-aging** with minimal visual artifacts
- **Temporal stability** across video sequences to prevent flickering
- **Regional control** over different facial areas (eyes, forehead, cheeks, mouth, jaw, neck, hair, eyebrows)
- **Multiple processing modes**: Feed-forward (fast), Diffusion (high-quality), Hybrid, and Auto
- **Real-time preview** on modern GPUs
- **Batch processing** for efficient video handling

### Technical Highlights
- Cross-platform support (Windows, Linux, macOS)
- Multiple GPU backends (CUDA, Metal, DirectML, CPU fallback)
- ACES/OCIO color pipeline integration
- Self-learning capabilities with artist feedback
- Quick face adaptation for new performers

### Host Integration
- **OpenFX**: Nuke, DaVinci Resolve/Fusion, Flame
- **After Effects**: Native AE SDK plugin
- **Blender**: Python add-on
- **CLI**: Batch processing tool

## Installation

### Prerequisites
- CMake 3.16+
- C++17 compatible compiler
- CUDA Toolkit 11.0+ (for NVIDIA GPU support)
- OpenCV 4.5+
- Python 3.8+ (for Python bindings)

### Building from Source

```bash
# Clone the repository
git clone https://github.com/your-org/ufra.git
cd ufra

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)

# Install
sudo make install
```

### Python Bindings

```bash
# Install Python package
pip install ./python_bindings

# Or build manually
cd python_bindings
python setup.py install
```

## Quick Start

### CLI Usage

```bash
# Process a video with face aging
ufra_cli -i input.mp4 -o output.mp4 -a 65 -m feedforward

# Process an image sequence
ufra_cli -i frame_%04d.jpg -o aged_%04d.jpg -a 25 --identity-lock 0.8

# Batch process with diffusion mode
ufra_cli -i input.mp4 -o output.mp4 -a 30 -m diffusion --temporal-stability
```

### Python API

```python
import pyufra
import cv2

# Initialize engine
engine = pyufra.create_engine()
config = pyufra.ModelConfig()
config.backend = pyufra.GPUBackend.CUDA
engine.initialize(config)
engine.load_models("./models")

# Setup age controls
controls = pyufra.AgeControls()
controls.target_age = 65.0
controls.identity_lock_strength = 0.8

# Process image
image = cv2.imread("/images/portrait.jpg")
context = pyufra.FrameContext()
context.set_input_frame(image)
context.controls = controls

result = engine.process_frame(context)
if result.success:
    output = result.get_output_frame()
    cv2.imwrite("output.jpg", output)
```

### OpenFX Plugin (Nuke)

1. Install the UFRa.ofx plugin to your OFX plugins directory
2. Load your footage in Nuke
3. Add the UFRa node from the MetaGPT/FaceReaging menu
4. Adjust parameters:
   - **Target Age**: Set desired age (0-100)
   - **Processing Mode**: Choose quality vs speed
   - **Identity Lock**: Preserve facial identity
   - **Temporal Stability**: Reduce flicker

## Architecture

### Core Engine (C++)
- **Face Detection**: Multi-face detection and tracking
- **Age Estimation**: Automated age assessment
- **Face Parsing**: Semantic segmentation of facial regions
- **Feed-forward Generator**: Fast U-Net based aging
- **Diffusion Editor**: High-quality diffusion-based aging
- **Compositor**: Seamless face blending and integration

### Processing Pipeline
1. **Pre-processing**: Face detection, alignment, parsing
2. **Age Transformation**: Neural network-based re-aging
3. **Post-processing**: Detail reinjection, color correction
4. **Compositing**: Blend processed face back to original

### GPU Acceleration
- CUDA: Primary backend for NVIDIA GPUs
- Metal: Apple Silicon and AMD GPUs on macOS
- DirectML: Windows GPU acceleration
- CPU Fallback: Cross-platform compatibility

## Model Architecture

### Feed-forward Network
- U-Net generator with skip connections
- Blur-pooling for anti-aliasing
- Patch-style discriminator
- Identity preservation losses

### Diffusion Model
- Image-conditioned identity control
- Spatial control networks
- Temporal attention modules
- Motion-aware processing

### Training Data
- Age-paired identity datasets
- Multi-ethnic and age-diverse samples
- Temporal video sequences
- High-quality facial parsing labels

## Configuration

### Model Configuration
```cpp
ufra::ModelConfig config;
config.backend = ufra::GPUBackend::CUDA;
config.batch_size = 4;
config.use_half_precision = true;
config.max_resolution = 1024;
```

### Age Controls
```cpp
ufra::AgeControls controls;
controls.target_age = 45.0f;
controls.identity_lock_strength = 0.7f;
controls.temporal_stability = 0.8f;
controls.texture_keep = 0.6f;
controls.enable_hair_aging = true;
controls.gray_density = 0.5f;
```

## Performance

### Throughput Targets
- **1080p**: ≥10 fps preview, ≥5 fps render
- **4K**: ≥2-3 fps preview, tiled rendering
- **Memory**: Automatic tiling for VRAM management
- **Batch**: Multi-GPU support for background renders

### Quality Metrics
- Identity similarity ≥ 0.85
- Temporal stability (low LPIPS variance)
- Age accuracy within ±3 years
- Perceptual quality scores

## Development

### Building Tests
```bash
cd build
make test
ctest --verbose
```

### Adding New Faces
```cpp
// Register a new performer
std::vector<cv::Mat> reference_frames = loadReferenceFrames("actor_name");
engine->registerNewFace("actor_name", reference_frames);
```

### Custom Training
```python
# Train face-specific adapter
from ufra.training import FaceAdapter

adapter = FaceAdapter()
adapter.train(face_name="new_actor", 
              reference_images=images,
              age_pairs=age_data)
adapter.save("new_actor_adapter.pth")
```

## Ethics and Safety

### Consent and Approval
- Explicit consent required for each identity
- Audit logs for all processing operations
- Watermarking and metadata tracking
- Usage restrictions and guidelines

### Technical Safeguards
- Local processing by default
- Encrypted model storage
- Identity verification systems
- Tamper detection mechanisms

## Troubleshooting

### Common Issues

**GPU Memory Errors**
```bash
# Reduce batch size or resolution
ufra_cli --batch-size 1 --max-resolution 512
```

**Model Loading Failures**
```bash
# Verify model path and permissions
ls -la /usr/local/share/ufra/models/
```

**Poor Quality Results**
- Increase identity lock strength
- Use diffusion mode for hero shots
- Ensure sufficient lighting in source material

### Debug Mode
```bash
export UFRA_LOG_LEVEL=DEBUG
ufra_cli --verbose -i input.mp4 -o output.mp4
```

## API Reference

### Core Classes
- `Engine`: Main processing engine
- `FaceDetector`: Face detection and tracking
- `AgeEstimator`: Age prediction
- `FeedforwardGenerator`: Fast aging network
- `DiffusionEditor`: High-quality aging

### Key Methods
- `Engine::initialize()`: Setup engine with configuration
- `Engine::processFrame()`: Process single frame
- `Engine::processBatch()`: Batch processing
- `Engine::registerNewFace()`: Add new performer

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

### Code Standards
- C++17 features encouraged
- Google C++ style guide
- Comprehensive unit tests
- Documentation for public APIs

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: GitHub Issues tracker
- **Documentation**: [docs.ufra.ai](https://docs.ufra.ai)
- **Community**: Discord server
- **Commercial**: Enterprise support available

## Roadmap

### Version 1.1
- Real-time streaming support
- Advanced hair aging
- Emotion preservation
- Mobile GPU support

### Version 2.0
- Multi-person scenes
- Full body aging
- Expression transfer
- AR/VR integration

---

**Universal Face Re-Aging (UFRa)** - Bringing AI-powered face aging to professional VFX workflows.