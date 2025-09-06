#pragma once

#include "types.h"
#include <string>
#include <memory>

namespace ufra {

class FaceDetector {
public:
    FaceDetector();
    ~FaceDetector();

    bool loadModel(const std::string& model_path);
    std::vector<Face> detectFaces(const ImageData& image);
    
    void setConfidenceThreshold(float threshold);
    void setNMSThreshold(float threshold);
    void setMaxFaces(int max_faces);

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace ufra