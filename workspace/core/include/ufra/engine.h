#pragma once

#include "types.h"
#include <memory>
#include <string>

namespace ufra {

class Engine {
public:
    Engine();
    ~Engine();

    // Initialization
    bool initialize(const ModelConfig& config);
    bool isInitialized() const;
    void shutdown();

    // Model management
    bool loadModels(const std::string& model_dir);
    bool registerNewFace(const std::string& face_name, const std::vector<ImageData>& reference_frames);
    bool loadFaceAdapter(const std::string& face_name);
    std::vector<std::string> getAvailableFaces() const;

    // Frame processing
    ProcessingResult processFrame(const FrameContext& context);
    ProcessingResult processBatch(const std::vector<FrameContext>& contexts);

    // Interactive preview
    bool startPreview(int width, int height);
    ProcessingResult previewFrame(const ImageData& input, const AgeControls& controls);
    void stopPreview();

    // Utility functions
    std::vector<Face> detectFaces(const ImageData& image);
    float estimateAge(const Face& face);
    MaskImage generateFaceParsing(const Face& face);
    cv::Mat computeOpticalFlow(const ImageData& frame1, const ImageData& frame2);

    // Configuration
    void setProcessingMode(ProcessingMode mode);
    ProcessingMode getProcessingMode() const;
    void setGPUBackend(GPUBackend backend);
    GPUBackend getGPUBackend() const;

    // Callbacks
    void setProgressCallback(ProgressCallback callback);
    void setErrorCallback(ErrorCallback callback);

    // Metrics and monitoring
    std::map<std::string, float> getPerformanceMetrics() const;
    std::string getVersionInfo() const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

// Factory functions
std::unique_ptr<Engine> createEngine();
std::string getLibraryVersion();
std::vector<GPUBackend> getAvailableBackends();

} // namespace ufra