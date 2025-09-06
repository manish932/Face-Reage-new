#pragma once

#include "types.h"
#include <string>
#include <memory>

namespace ufra {

class FaceParser {
public:
    FaceParser();
    ~FaceParser();

    bool loadModel(const std::string& model_path);
    MaskImage parseFace(const ImageData& face_crop);
    std::vector<MaskImage> parseFacesBatch(const std::vector<ImageData>& face_crops);
    
    // Get individual region masks
    MaskImage getEyesMask(const MaskImage& full_mask);
    MaskImage getForeheadMask(const MaskImage& full_mask);
    MaskImage getCheeksMask(const MaskImage& full_mask);
    MaskImage getMouthMask(const MaskImage& full_mask);
    MaskImage getJawMask(const MaskImage& full_mask);
    MaskImage getNeckMask(const MaskImage& full_mask);
    MaskImage getHairMask(const MaskImage& full_mask);
    MaskImage getEyebrowsMask(const MaskImage& full_mask);

    void setInputSize(int width, int height);

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace ufra