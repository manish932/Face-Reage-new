#pragma once

#include "types.h"
#include <memory>

namespace ufra {

class OpticalFlow {
public:
    OpticalFlow();
    ~OpticalFlow();

    cv::Mat computeFlow(const ImageData& frame1, const ImageData& frame2);
    cv::Mat warpImage(const ImageData& source, const cv::Mat& flow);
    
    void setFlowAlgorithm(const std::string& algorithm); // "lucas_kanade", "farneback", "tvl1"
    void setQualityLevel(float quality);
    void enableGPUAcceleration(bool enable);

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace ufra