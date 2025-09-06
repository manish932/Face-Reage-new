#pragma once

#include "minimal_types.h"

namespace ufra {

class Engine {
public:
    Engine();
    ~Engine();
    
    // Core functionality
    bool initialize(const ModelConfig& config);
    bool loadModels(const std::string& model_path);
    ProcessingResult processFrame(const FrameContext& context);
    
    // Configuration
    void setProcessingMode(ProcessingMode mode);
    ProcessingMode getProcessingMode() const;
    
    // Utility
    std::string getVersion() const;
    void shutdown();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// Factory function
std::unique_ptr<Engine> createEngine();

// Utility functions
std::vector<GPUBackend> getAvailableBackends();

} // namespace ufra