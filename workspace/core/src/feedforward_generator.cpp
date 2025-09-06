#include "ufra/feedforward_generator.h"
#include <opencv2/dnn.hpp>
#include <iostream>

namespace ufra {

class FeedforwardGenerator::Impl {
public:
    Impl() : model_loaded_(false), input_width_(512), input_height_(512),
             temporal_stabilization_(false), identity_strength_(0.5f) {}

    bool loadModel(const std::string& model_path) {
        try {
            net_ = cv::dnn::readNet(model_path);
            if (net_.empty()) {
                std::cerr << "Failed to load feedforward generator model: " << model_path << std::endl;
                return false;
            }
            
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            
            model_loaded_ = true;
            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "Error loading feedforward generator model: " << e.what() << std::endl;
            return false;
        }
    }

    ImageData generateAgedFace(const ImageData& face_crop, 
                              const AgeControls& controls,
                              const MaskImage& parsing_mask) {
        if (!model_loaded_ || face_crop.empty()) {
            return face_crop.clone();
        }

        try {
            // Preprocess input
            cv::Mat resized;
            cv::resize(face_crop, resized, cv::Size(input_width_, input_height_));
            
            // Normalize to [-1, 1]
            cv::Mat normalized;
            resized.convertTo(normalized, CV_32F, 2.0/255.0, -1.0);
            
            // Create input blob with age conditioning
            cv::Mat blob;
            cv::dnn::blobFromImage(normalized, blob, 1.0, 
                                   cv::Size(input_width_, input_height_), 
                                   cv::Scalar(0, 0, 0), true, false);
            
            // Add age conditioning vector
            cv::Mat age_vector = cv::Mat::zeros(1, 1, CV_32F);
            age_vector.at<float>(0, 0) = controls.target_age / 100.0f; // Normalize age
            
            // Set inputs
            net_.setInput(blob, "face_input");
            net_.setInput(age_vector, "age_input");
            
            // Forward pass
            cv::Mat output = net_.forward();
            
            // Post-process output
            cv::Mat result_float;
            cv::dnn::imagesFromBlob(output, result_float);
            
            // Denormalize from [-1, 1] to [0, 255]
            cv::Mat result_uint8;
            result_float = (result_float + 1.0) * 127.5;
            result_float.convertTo(result_uint8, CV_8UC3);
            
            // Apply identity preservation blending
            cv::Mat final_result;
            cv::addWeighted(face_crop, controls.identity_lock_strength,
                           result_uint8, 1.0f - controls.identity_lock_strength,
                           0, final_result);
            
            // Apply regional masking if parsing mask is provided
            if (!parsing_mask.empty()) {
                applyRegionalBlending(face_crop, final_result, parsing_mask, controls);
            }
            
            return final_result;
        }
        catch (const std::exception& e) {
            std::cerr << "Error in feedforward generation: " << e.what() << std::endl;
            return face_crop.clone();
        }
    }

    std::vector<ImageData> generateAgedFacesBatch(
        const std::vector<ImageData>& face_crops,
        const std::vector<AgeControls>& controls,
        const std::vector<MaskImage>& parsing_masks) {
        
        std::vector<ImageData> results;
        results.reserve(face_crops.size());
        
        for (size_t i = 0; i < face_crops.size(); ++i) {
            const AgeControls& ctrl = i < controls.size() ? controls[i] : controls[0];
            const MaskImage& mask = i < parsing_masks.size() ? parsing_masks[i] : cv::Mat();
            
            results.push_back(generateAgedFace(face_crops[i], ctrl, mask));
        }
        
        return results;
    }

private:
    void applyRegionalBlending(const ImageData& original, ImageData& aged, 
                              const MaskImage& parsing_mask, const AgeControls& controls) {
        // Apply different blending strengths to different facial regions
        cv::Mat hair_mask, forehead_mask, eye_mask, mouth_mask;
        
        // Extract region masks (assuming parsing mask has different values for regions)
        cv::inRange(parsing_mask, cv::Scalar(1), cv::Scalar(1), hair_mask);     // Hair = 1
        cv::inRange(parsing_mask, cv::Scalar(2), cv::Scalar(2), forehead_mask); // Forehead = 2
        cv::inRange(parsing_mask, cv::Scalar(3), cv::Scalar(4), eye_mask);      // Eyes = 3,4
        cv::inRange(parsing_mask, cv::Scalar(5), cv::Scalar(6), mouth_mask);    // Mouth = 5,6
        
        // Apply region-specific blending
        if (controls.enable_hair_aging) {
            blendRegion(original, aged, hair_mask, 0.8f);
        } else {
            blendRegion(original, aged, hair_mask, 0.1f); // Preserve original hair
        }
        
        // Eyes and mouth are more sensitive - preserve more identity
        blendRegion(original, aged, eye_mask, 0.3f);
        blendRegion(original, aged, mouth_mask, 0.4f);
    }
    
    void blendRegion(const ImageData& original, ImageData& aged, 
                    const cv::Mat& mask, float aged_strength) {
        if (mask.empty()) return;
        
        cv::Mat mask_norm;
        mask.convertTo(mask_norm, CV_32F, 1.0/255.0);
        
        for (int y = 0; y < aged.rows; ++y) {
            for (int x = 0; x < aged.cols; ++x) {
                if (mask_norm.at<float>(y, x) > 0.5f) {
                    cv::Vec3b orig_pixel = original.at<cv::Vec3b>(y, x);
                    cv::Vec3b aged_pixel = aged.at<cv::Vec3b>(y, x);
                    
                    cv::Vec3b blended_pixel;
                    for (int c = 0; c < 3; ++c) {
                        blended_pixel[c] = static_cast<uchar>(
                            orig_pixel[c] * (1.0f - aged_strength) + 
                            aged_pixel[c] * aged_strength
                        );
                    }
                    aged.at<cv::Vec3b>(y, x) = blended_pixel;
                }
            }
        }
    }

    cv::dnn::Net net_;
    bool model_loaded_;
    int input_width_, input_height_;
    bool temporal_stabilization_;
    float identity_strength_;
};

FeedforwardGenerator::FeedforwardGenerator() : pImpl(std::make_unique<Impl>()) {}
FeedforwardGenerator::~FeedforwardGenerator() = default;

bool FeedforwardGenerator::loadModel(const std::string& model_path) {
    return pImpl->loadModel(model_path);
}

ImageData FeedforwardGenerator::generateAgedFace(const ImageData& face_crop,
                                                const AgeControls& controls,
                                                const MaskImage& parsing_mask) {
    return pImpl->generateAgedFace(face_crop, controls, parsing_mask);
}

std::vector<ImageData> FeedforwardGenerator::generateAgedFacesBatch(
    const std::vector<ImageData>& face_crops,
    const std::vector<AgeControls>& controls,
    const std::vector<MaskImage>& parsing_masks) {
    return pImpl->generateAgedFacesBatch(face_crops, controls, parsing_masks);
}

void FeedforwardGenerator::setInputResolution(int width, int height) {
    pImpl->input_width_ = width;
    pImpl->input_height_ = height;
}

void FeedforwardGenerator::enableTemporalStabilization(bool enable) {
    pImpl->temporal_stabilization_ = enable;
}

void FeedforwardGenerator::setIdentityPreservationStrength(float strength) {
    pImpl->identity_strength_ = std::max(0.0f, std::min(1.0f, strength));
}

} // namespace ufra