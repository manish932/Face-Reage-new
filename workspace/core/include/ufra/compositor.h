#pragma once

#include "types.h"
#include <memory>

namespace ufra {

class Compositor {
public:
    Compositor();
    ~Compositor();

    void compositeFace(ImageData& target_frame, 
                      const ImageData& processed_face,
                      const Face& face_info);
    
    void setBlendingMode(const std::string& mode); // "linear", "poisson", "multiband"
    void setFeatherRadius(int radius);
    void enableColorCorrection(bool enable);
    void setDetailReinjectionStrength(float strength);

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace ufra