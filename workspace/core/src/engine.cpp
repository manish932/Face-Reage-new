#include "ufra/engine.h"
#include "ufra/face_detector.h"
#include "ufra/face_tracker.h"
#include "ufra/age_estimator.h"
#include "ufra/face_parser.h"
#include "ufra/feedforward_generator.h"
#include "ufra/diffusion_editor.h"
#include "ufra/optical_flow.h"
#include "ufra/compositor.h"
#include "ufra/gpu_memory_manager.h"
#include "ufra/model_loader.h"
#include <iostream>
#include <chrono>

namespace ufra {

class Engine::Impl {
public:
    Impl() : initialized_(false), processing_mode_(ProcessingMode::FEEDFORWARD),
             gpu_backend_(GPUBackend::CUDA) {}

    bool initialize(const ModelConfig& config) {
        try {
            config_ = config;
            
            // Initialize GPU memory manager
            gpu_manager_ = std::make_unique<GPUMemoryManager>();
            if (!gpu_manager_->initialize(config.backend)) {
                error_callback_("Failed to initialize GPU memory manager");
                return false;
            }

            // Initialize model loader
            model_loader_ = std::make_unique<ModelLoader>();
            
            // Initialize core components
            face_detector_ = std::make_unique<FaceDetector>();
            face_tracker_ = std::make_unique<FaceTracker>();
            age_estimator_ = std::make_unique<AgeEstimator>();
            face_parser_ = std::make_unique<FaceParser>();
            feedforward_generator_ = std::make_unique<FeedforwardGenerator>();
            diffusion_editor_ = std::make_unique<DiffusionEditor>();
            optical_flow_ = std::make_unique<OpticalFlow>();
            compositor_ = std::make_unique<Compositor>();

            initialized_ = true;
            return true;
        }
        catch (const std::exception& e) {
            if (error_callback_) {
                error_callback_("Engine initialization failed: " + std::string(e.what()));
            }
            return false;
        }
    }

    bool loadModels(const std::string& model_dir) {
        if (!initialized_) {
            error_callback_("Engine not initialized");
            return false;
        }

        try {
            // Load face detection model
            if (!face_detector_->loadModel(model_dir + "/face_detector.onnx")) {
                error_callback_("Failed to load face detection model");
                return false;
            }

            // Load age estimation model
            if (!age_estimator_->loadModel(model_dir + "/age_estimator.onnx")) {
                error_callback_("Failed to load age estimation model");
                return false;
            }

            // Load face parsing model
            if (!face_parser_->loadModel(model_dir + "/face_parser.onnx")) {
                error_callback_("Failed to load face parsing model");
                return false;
            }

            // Load feedforward generator
            if (!feedforward_generator_->loadModel(model_dir + "/feedforward_generator.onnx")) {
                error_callback_("Failed to load feedforward generator model");
                return false;
            }

            // Load diffusion model (optional for basic functionality)
            diffusion_editor_->loadModel(model_dir + "/diffusion_editor");

            return true;
        }
        catch (const std::exception& e) {
            error_callback_("Model loading failed: " + std::string(e.what()));
            return false;
        }
    }

    ProcessingResult processFrame(const FrameContext& context) {
        if (!initialized_) {
            ProcessingResult result;
            result.success = false;
            result.error_message = "Engine not initialized";
            return result;
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        try {
            ProcessingResult result;
            
            // Detect faces if not provided
            std::vector<Face> faces = context.detected_faces;
            if (faces.empty()) {
                faces = face_detector_->detectFaces(context.input_frame);
            }

            if (faces.empty()) {
                result.output_frame = context.input_frame.clone();
                result.success = true;
                return result;
            }

            // Process each face
            ImageData output_frame = context.input_frame.clone();
            for (auto& face : faces) {
                // Generate face parsing mask
                MaskImage parsing_mask = face_parser_->parseFace(face.aligned_crop);
                
                // Apply age transformation based on processing mode
                ImageData processed_face;
                if (context.mode == ProcessingMode::FEEDFORWARD || 
                    context.mode == ProcessingMode::AUTO) {
                    processed_face = feedforward_generator_->generateAgedFace(
                        face.aligned_crop, context.controls, parsing_mask);
                } else if (context.mode == ProcessingMode::DIFFUSION) {
                    processed_face = diffusion_editor_->generateAgedFace(
                        face.aligned_crop, context.controls, parsing_mask);
                }

                // Composite back to original frame
                compositor_->compositeFace(output_frame, processed_face, face);
            }

            result.output_frame = output_frame;
            result.processed_faces = faces;
            result.success = true;

            // Calculate performance metrics
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time).count();
            result.metrics["processing_time_ms"] = static_cast<float>(duration);
            result.metrics["faces_processed"] = static_cast<float>(faces.size());

            return result;
        }
        catch (const std::exception& e) {
            ProcessingResult result;
            result.success = false;
            result.error_message = "Processing failed: " + std::string(e.what());
            return result;
        }
    }

    std::vector<Face> detectFaces(const ImageData& image) {
        if (!initialized_ || !face_detector_) {
            return {};
        }
        return face_detector_->detectFaces(image);
    }

    float estimateAge(const Face& face) {
        if (!initialized_ || !age_estimator_) {
            return 0.0f;
        }
        return age_estimator_->estimateAge(face.aligned_crop);
    }

    bool initialized_;
    ModelConfig config_;
    ProcessingMode processing_mode_;
    GPUBackend gpu_backend_;

    // Core components
    std::unique_ptr<GPUMemoryManager> gpu_manager_;
    std::unique_ptr<ModelLoader> model_loader_;
    std::unique_ptr<FaceDetector> face_detector_;
    std::unique_ptr<FaceTracker> face_tracker_;
    std::unique_ptr<AgeEstimator> age_estimator_;
    std::unique_ptr<FaceParser> face_parser_;
    std::unique_ptr<FeedforwardGenerator> feedforward_generator_;
    std::unique_ptr<DiffusionEditor> diffusion_editor_;
    std::unique_ptr<OpticalFlow> optical_flow_;
    std::unique_ptr<Compositor> compositor_;

    // Callbacks
    ProgressCallback progress_callback_;
    ErrorCallback error_callback_;

    // Performance metrics
    std::map<std::string, float> performance_metrics_;
};

// Engine implementation
Engine::Engine() : pImpl(std::make_unique<Impl>()) {}

Engine::~Engine() = default;

bool Engine::initialize(const ModelConfig& config) {
    return pImpl->initialize(config);
}

bool Engine::isInitialized() const {
    return pImpl->initialized_;
}

ProcessingResult Engine::processFrame(const FrameContext& context) {
    return pImpl->processFrame(context);
}

std::vector<Face> Engine::detectFaces(const ImageData& image) {
    return pImpl->detectFaces(image);
}

float Engine::estimateAge(const Face& face) {
    return pImpl->estimateAge(face);
}

void Engine::setProcessingMode(ProcessingMode mode) {
    pImpl->processing_mode_ = mode;
}

ProcessingMode Engine::getProcessingMode() const {
    return pImpl->processing_mode_;
}

void Engine::setErrorCallback(ErrorCallback callback) {
    pImpl->error_callback_ = callback;
}

std::string Engine::getVersionInfo() const {
    return "UFRa Engine v1.0.0";
}

// Factory functions
std::unique_ptr<Engine> createEngine() {
    return std::make_unique<Engine>();
}

std::string getLibraryVersion() {
    return "1.0.0";
}

std::vector<GPUBackend> getAvailableBackends() {
    std::vector<GPUBackend> backends;
    backends.push_back(GPUBackend::CPU_FALLBACK);
    
#ifdef CUDA_FOUND
    backends.push_back(GPUBackend::CUDA);
#endif
    
#ifdef METAL_FOUND
    backends.push_back(GPUBackend::METAL);
#endif
    
#ifdef DIRECTML_FOUND
    backends.push_back(GPUBackend::DIRECTML);
#endif
    
    return backends;
}

} // namespace ufra