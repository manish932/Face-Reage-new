#pragma once

#include "types.h"
#include <string>
#include <memory>

namespace ufra {

class DiffusionEditor {
public:
    DiffusionEditor();
    ~DiffusionEditor();

    bool loadModel(const std::string& model_dir);
    ImageData generateAgedFace(const ImageData& face_crop,
                              const AgeControls& controls,
                              const MaskImage& parsing_mask);
    
    bool loadIdentityAdapter(const std::string& adapter_path);
    void setDiffusionSteps(int steps);
    void setGuidanceScale(float scale);
    void setSeed(unsigned int seed);
    void enableTemporalCoherence(bool enable);

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace ufra