#pragma once

#include "types.h"
#include <string>
#include <memory>

namespace ufra {

class FeedforwardGenerator {
public:
    FeedforwardGenerator();
    ~FeedforwardGenerator();

    bool loadModel(const std::string& model_path);
    ImageData generateAgedFace(const ImageData& face_crop, 
                              const AgeControls& controls,
                              const MaskImage& parsing_mask);
    
    std::vector<ImageData> generateAgedFacesBatch(
        const std::vector<ImageData>& face_crops,
        const std::vector<AgeControls>& controls,
        const std::vector<MaskImage>& parsing_masks);

    void setInputResolution(int width, int height);
    void enableTemporalStabilization(bool enable);
    void setIdentityPreservationStrength(float strength);

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace ufra