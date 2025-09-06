#include "ufra/minimal_engine.h"
#include "ufra/utils.h"
#include <iostream>

namespace ufra {

class Engine::Impl {
public:
    ModelConfig config_;
    ProcessingMode mode_ = ProcessingMode::FEEDFORWARD;
    bool initialized_ = false;

    bool initialize(const ModelConfig& config) {
        config_ = config;
        initialized_ = true;
        std::cout << "UFRa Engine initialized (minimal build)" << std::endl;
        return true;
    }

    bool loadModels(const std::string& model_path) {
        if (!initialized_) {
            return false;
        }
        
        std::cout << "Loading models from: " << model_path << std::endl;
        
        // In minimal build, we simulate model loading
        if (!fileExists(model_path)) {
            std::cout << "Warning: Model path does not exist, using fallback" << std::endl;
        }
        
        return true;
    }

    ProcessingResult processFrame(const FrameContext& context) {
        ProcessingResult result;
        
        if (!initialized_) {
            result.error_message = "Engine not initialized";
            return result;
        }
        
        if (context.input_frame.empty()) {
            result.error_message = "Empty input frame";
            return result;
        }
        
        // Simulate processing
        result.output_frame = context.input_frame;  // Copy input to output
        
        // Create mock face detection
        Face mockFace;
        mockFace.box.x = context.input_frame.width * 0.25f;
        mockFace.box.y = context.input_frame.height * 0.25f;
        mockFace.box.width = context.input_frame.width * 0.5f;
        mockFace.box.height = context.input_frame.height * 0.5f;
        mockFace.box.confidence = 0.9f;
        mockFace.box.face_id = 1;
        mockFace.track_id = 1;
        mockFace.frame_number = context.frame_number;
        
        result.processed_faces.push_back(mockFace);
        
        // Add metrics
        result.metrics["processing_time_ms"] = 50.0f;
        result.metrics["face_count"] = 1.0f;
        result.metrics["confidence"] = 0.9f;
        
        result.success = true;
        
        return result;
    }
};

Engine::Engine() : impl_(std::make_unique<Impl>()) {}

Engine::~Engine() = default;

bool Engine::initialize(const ModelConfig& config) {
    return impl_->initialize(config);
}

bool Engine::loadModels(const std::string& model_path) {
    return impl_->loadModels(model_path);
}

ProcessingResult Engine::processFrame(const FrameContext& context) {
    return impl_->processFrame(context);
}

void Engine::setProcessingMode(ProcessingMode mode) {
    impl_->mode_ = mode;
}

ProcessingMode Engine::getProcessingMode() const {
    return impl_->mode_;
}

std::string Engine::getVersion() const {
    return getLibraryVersion();
}

void Engine::shutdown() {
    impl_->initialized_ = false;
}

std::unique_ptr<Engine> createEngine() {
    return std::make_unique<Engine>();
}

std::vector<GPUBackend> getAvailableBackends() {
    return {GPUBackend::CPU_FALLBACK};
}

} // namespace ufra