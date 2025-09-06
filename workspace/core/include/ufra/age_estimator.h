#pragma once

#include "types.h"
#include <string>
#include <memory>

namespace ufra {

class AgeEstimator {
public:
    AgeEstimator();
    ~AgeEstimator();

    bool loadModel(const std::string& model_path);
    float estimateAge(const ImageData& face_crop);
    std::vector<float> estimateAgeBatch(const std::vector<ImageData>& face_crops);

    void setInputSize(int width, int height);
    void setNormalization(float mean, float std);

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace ufra