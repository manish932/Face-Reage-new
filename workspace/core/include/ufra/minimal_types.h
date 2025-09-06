#pragma once

#include <vector>
#include <memory>
#include <string>
#include <map>
#include <functional>

namespace ufra {

// Minimal image representation
struct ImageData {
    int width = 0;
    int height = 0;
    int channels = 3;
    std::vector<unsigned char> data;
    
    ImageData() = default;
    ImageData(int w, int h, int c = 3) : width(w), height(h), channels(c) {
        data.resize(w * h * c);
    }
    
    bool empty() const { return width == 0 || height == 0; }
    size_t size() const { return data.size(); }
};

// Basic geometric types
struct Point {
    float x = 0, y = 0;
    Point() = default;
    Point(float x_, float y_) : x(x_), y(y_) {}
};

struct Rect {
    float x = 0, y = 0, width = 0, height = 0;
    Rect() = default;
    Rect(float x_, float y_, float w, float h) : x(x_), y(y_), width(w), height(h) {}
};

// Face detection structures
struct FaceBox {
    float x, y, width, height;
    float confidence;
    int face_id;
};

struct Face {
    FaceBox box;
    std::vector<Point> landmarks;
    ImageData aligned_crop;
    int track_id = -1;
    int frame_number = 0;
};

// Age controls
struct AgeControls {
    float target_age = 25.0f;
    float identity_lock_strength = 0.8f;
    float temporal_stability = 0.9f;
    float texture_keep = 0.6f;
    bool enable_hair_aging = true;
    float gray_density = 0.5f;
};

// Processing modes
enum class ProcessingMode {
    FEEDFORWARD,
    DIFFUSION,
    HYBRID,
    AUTO
};

// GPU backend types
enum class GPUBackend {
    CUDA,
    METAL,
    DIRECTML,
    CPU_FALLBACK
};

// Model configuration
struct ModelConfig {
    std::string model_path;
    GPUBackend backend = GPUBackend::CPU_FALLBACK;
    int batch_size = 1;
    bool use_half_precision = false;
    int max_resolution = 512;
};

// Frame processing context
struct FrameContext {
    int frame_number = 0;
    ImageData input_frame;
    std::vector<Face> detected_faces;
    AgeControls controls;
    ProcessingMode mode = ProcessingMode::FEEDFORWARD;
};

// Result structures
struct ProcessingResult {
    ImageData output_frame;
    std::vector<Face> processed_faces;
    std::map<std::string, float> metrics;
    bool success = false;
    std::string error_message;
};

// Callback types
using ProgressCallback = std::function<void(float progress, const std::string& status)>;
using ErrorCallback = std::function<void(const std::string& error)>;

} // namespace ufra