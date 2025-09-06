#pragma once

#include <vector>
#include <memory>
#include <string>
#include <map>

// Minimal OpenCV-like types for dependency-free build
namespace cv {
    class Mat {
    public:
        Mat() : rows(0), cols(0), channels_(3) {}
        Mat(int h, int w, int type) : rows(h), cols(w), channels_(3) {}
        
        bool empty() const { return rows == 0 || cols == 0; }
        int size() const { return rows * cols; }
        Mat clone() const { return *this; }
        
        int rows, cols;
        int channels_;
        
        // Basic operators
        Mat& operator=(const Mat& other) {
            rows = other.rows;
            cols = other.cols;
            channels_ = other.channels_;
            return *this;
        }
    };
    
    class Rect {
    public:
        Rect() : x(0), y(0), width(0), height(0) {}
        Rect(int x_, int y_, int w, int h) : x(x_), y(y_), width(w), height(h) {}
        
        int x, y, width, height;
    };
    
    class Point {
    public:
        Point() : x(0), y(0) {}
        Point(int x_, int y_) : x(x_), y(y_) {}
        
        int x, y;
    };
    
    class Size {
    public:
        Size() : width(0), height(0) {}
        Size(int w, int h) : width(w), height(h) {}
        
        int width, height;
    };
}

namespace ufra {

// Forward declarations
class Engine;
class FaceDetector;
class FaceTracker;
class AgeEstimator;
class FaceParser;
class FeedforwardGenerator;
class DiffusionEditor;
class OpticalFlow;
class Compositor;

// Basic types
using ImageData = cv::Mat;
using FloatImage = cv::Mat;
using MaskImage = cv::Mat;

// Face detection structures
struct FaceBox {
    float x, y, width, height;
    float confidence;
    int face_id;
};

struct FaceLandmarks {
    std::vector<cv::Point2f> points;
    float confidence;
};

struct Face {
    FaceBox box;
    FaceLandmarks landmarks;
    cv::Mat aligned_crop;
    cv::Mat transform_matrix;
    int track_id;
    int frame_number;
};

// Age control structures
struct AgeMap {
    cv::Mat global_age_map;      // Per-pixel target age
    cv::Mat region_masks[8];     // Eyes, forehead, cheeks, mouth, jaw, neck, hair, eyebrows
    float global_strength;
    float region_strengths[8];
};

struct AgeControls {
    float target_age;
    AgeMap age_map;
    float identity_lock_strength;
    float temporal_stability;
    float texture_keep;
    float skin_clean;
    bool enable_hair_aging;
    bool enable_beard_aging;
    bool enable_neck_aging;
    float gray_density;
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
    GPUBackend backend;
    int batch_size;
    bool use_half_precision;
    int max_resolution;
};

// Frame processing context
struct FrameContext {
    int frame_number;
    ImageData input_frame;
    std::vector<Face> detected_faces;
    cv::Mat optical_flow;
    AgeControls controls;
    ProcessingMode mode;
};

// Result structures
struct ProcessingResult {
    ImageData output_frame;
    std::vector<Face> processed_faces;
    std::map<std::string, float> metrics;
    bool success;
    std::string error_message;
};

// Callback types
using ProgressCallback = std::function<void(float progress, const std::string& status)>;
using ErrorCallback = std::function<void(const std::string& error)>;

} // namespace ufra