#pragma once

#include "types.h"
#include <memory>
#include <string>
#include <map>

namespace ufra {

class ModelLoader {
public:
    ModelLoader();
    ~ModelLoader();

    bool loadONNXModel(const std::string& model_path, const std::string& model_name);
    bool loadPyTorchModel(const std::string& model_path, const std::string& model_name);
    bool loadTensorRTModel(const std::string& model_path, const std::string& model_name);
    
    void* getModelHandle(const std::string& model_name);
    bool isModelLoaded(const std::string& model_name) const;
    
    void unloadModel(const std::string& model_name);
    void unloadAllModels();
    
    std::vector<std::string> getLoadedModels() const;
    std::map<std::string, std::string> getModelInfo(const std::string& model_name) const;
    
    void setPreferredBackend(GPUBackend backend);
    void enableModelOptimization(bool enable);

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace ufra