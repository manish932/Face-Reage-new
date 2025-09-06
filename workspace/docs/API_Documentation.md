# UFRa API Documentation

## Table of Contents
1. [Core Engine](#core-engine)
2. [Face Detection](#face-detection)
3. [Age Estimation](#age-estimation)
4. [Face Parsing](#face-parsing)
5. [Feedforward Generator](#feedforward-generator)
6. [Diffusion Editor](#diffusion-editor)
7. [Compositor](#compositor)
8. [Python Bindings](#python-bindings)
9. [Error Handling](#error-handling)
10. [Performance Optimization](#performance-optimization)

## Core Engine

### Engine Class

The main processing engine that coordinates all UFRa components.

```cpp
#include "ufra/engine.h"

// Create engine instance
auto engine = ufra::createEngine();

// Configure engine
ufra::ModelConfig config;
config.backend = ufra::GPUBackend::CUDA;
config.batch_size = 4;
config.use_half_precision = true;
config.max_resolution = 1024;

// Initialize
bool success = engine->initialize(config);

// Load models
bool loaded = engine->loadModels("/path/to/models");
```

### Key Methods

#### `initialize(ModelConfig config)`
Initializes the engine with the specified configuration.

**Parameters:**
- `config`: Model configuration including GPU backend, batch size, precision settings

**Returns:** `bool` - Success status

#### `processFrame(FrameContext context)`
Processes a single frame with face re-aging.

**Parameters:**
- `context`: Frame processing context containing input image, age controls, and processing mode

**Returns:** `ProcessingResult` - Contains output image, processed faces, and metrics

## Python Bindings

### Installation

```bash
pip install pyufra
```

### Basic Usage

```python
import pyufra
import cv2
import numpy as np

# Create engine
engine = pyufra.create_engine()

# Configure
config = pyufra.ModelConfig()
config.backend = pyufra.GPUBackend.CUDA
config.batch_size = 1
config.use_half_precision = True

# Initialize
engine.initialize(config)
engine.load_models("./models")

# Load image
image = cv2.imread("/images/photo1756159210.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Setup controls
controls = pyufra.AgeControls()
controls.target_age = 65.0
controls.identity_lock_strength = 0.8

# Process
context = pyufra.FrameContext()
context.frame_number = 0
context.set_input_frame(image_rgb)
context.controls = controls
context.mode = pyufra.ProcessingMode.FEEDFORWARD

result = engine.process_frame(context)

if result.success:
    output_rgb = result.get_output_frame()
    output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite("/images/processedimage.jpg", output_bgr)
```

## Error Handling

### Exception Safety
All UFRa components are designed with exception safety in mind:

```cpp
try {
    auto result = engine->processFrame(context);
    if (!result.success) {
        std::cerr << "Processing failed: " << result.error_message << std::endl;
    }
} catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
}
```

### Common Error Scenarios

| Error Type | Cause | Solution |
|------------|-------|----------|
| Model Loading Failed | Invalid path or corrupted model | Check file path and model integrity |
| GPU Memory Error | Insufficient VRAM | Reduce batch size or resolution |
| Invalid Input | Empty or malformed image | Validate input dimensions and format |
| Backend Not Available | GPU drivers or CUDA not installed | Install proper drivers or use CPU fallback |

## Performance Optimization

### GPU Memory Management
```cpp
// Configure for optimal memory usage
config.batch_size = 1;              // Start with 1, increase if memory allows
config.use_half_precision = true;   // Reduces memory by ~50%
config.max_resolution = 512;        // Lower for real-time, higher for quality
```

### Threading Considerations
```cpp
// UFRa is thread-safe for read operations
// Use separate engine instances for concurrent processing
std::vector<std::unique_ptr<ufra::Engine>> engines;
for (int i = 0; i < num_threads; ++i) {
    engines.push_back(ufra::createEngine());
    engines[i]->initialize(config);
    engines[i]->loadModels(model_path);
}
```

### Batch Processing
```cpp
// Process multiple frames efficiently
std::vector<ufra::FrameContext> contexts;
// ... populate contexts ...

auto results = engine->processBatch(contexts);
```

## Integration Examples

### OpenFX Plugin (Nuke)
```cpp
// In plugin render function
ufra::FrameContext context;
context.input_frame = convertOFXToMat(src_image);
context.controls = getControlsFromUI(args.time);

auto result = engine->processFrame(context);
if (result.success) {
    copyMatToOFX(result.output_frame, dst_image);
}
```

### After Effects Plugin
```cpp
// In render function
PF_EffectWorld* input = &params[0]->u.ld;
PF_EffectWorld* output = &params[1]->u.ld;

cv::Mat input_mat = convertAEToMat(input);
// ... process with UFRa ...
copyMatToAE(output_mat, output);
```

### Blender Add-on
```python
import bpy
import pyufra

class UFRaOperator(bpy.types.Operator):
    bl_idname = "image.ufra_process"
    bl_label = "UFRa Face Aging"
    
    def execute(self, context):
        # Get active image
        img = bpy.data.images.get("///images/photo1756159210.jpg")
        
        # Process with UFRa
        engine = pyufra.create_engine()
        # ... setup and process ...
        
        # Update Blender image
        img.pixels[:] = processed_pixels
        img.update()
        
        return {'FINISHED'}
```

## Model Training and Adaptation

### Face Adapter Training
```python
from ufra.training import FaceAdapter

# Train custom adapter for specific actor
adapter = FaceAdapter()
adapter.train(
    face_name="actor_name",
    reference_images=reference_imgs,
    age_pairs=training_data,
    epochs=100,
    learning_rate=1e-4
)

# Save adapter
adapter.save("actor_adapter.pth")

# Load in engine
engine.loadFaceAdapter("actor_adapter.pth")
```

### Quality Metrics
```cpp
// Access processing metrics
auto metrics = result.metrics;
float processing_time = metrics["processing_time_ms"];
float identity_similarity = metrics["identity_similarity"];
float temporal_consistency = metrics["temporal_consistency"];
float age_accuracy = metrics["age_accuracy"];
```

## Deployment Guidelines

### Production Checklist
- [ ] Models downloaded and validated
- [ ] GPU drivers installed and tested
- [ ] Memory requirements verified
- [ ] Error handling implemented
- [ ] Logging configured
- [ ] Performance benchmarked
- [ ] Security measures in place

### Security Considerations
```cpp
// Enable audit logging
engine->enableAuditLog("/var/log/ufra/audit.log");

// Verify model integrity
bool model_valid = engine->verifyModelIntegrity(model_path);

// Enable watermarking
engine->enableWatermarking(true);
```

### Monitoring and Telemetry
```cpp
// Setup performance monitoring
engine->setMetricsCallback([](const auto& metrics) {
    // Log to monitoring system
    logger->info("Processing time: {}ms", metrics["processing_time_ms"]);
    logger->info("GPU utilization: {}%", metrics["gpu_utilization"]);
});
```