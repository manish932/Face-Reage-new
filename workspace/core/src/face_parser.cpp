#include "ufra/face_parser.h"
#include <opencv2/dnn.hpp>
#include <iostream>

namespace ufra {

class FaceParser::Impl {
public:
    Impl() : model_loaded_(false), input_width_(512), input_height_(512) {}

    bool loadModel(const std::string& model_path) {
        try {
            net_ = cv::dnn::readNet(model_path);
            if (net_.empty()) {
                std::cerr << "Failed to load face parsing model: " << model_path << std::endl;
                return false;
            }
            
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            
            model_loaded_ = true;
            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "Error loading face parsing model: " << e.what() << std::endl;
            return false;
        }
    }

    MaskImage parseFace(const ImageData& face_crop) {
        if (!model_loaded_ || face_crop.empty()) {
            return cv::Mat::zeros(face_crop.size(), CV_8UC1);
        }

        try {
            // Preprocess input
            cv::Mat resized;
            cv::resize(face_crop, resized, cv::Size(input_width_, input_height_));
            
            // Create blob
            cv::Mat blob;
            cv::dnn::blobFromImage(resized, blob, 1.0/255.0, 
                                   cv::Size(input_width_, input_height_), 
                                   cv::Scalar(0.485, 0.456, 0.406), true, false);
            
            net_.setInput(blob);
            
            // Forward pass
            cv::Mat output = net_.forward();
            
            // Convert output to parsing mask
            cv::Mat parsing_mask = convertToParseMask(output);
            
            // Resize back to original size
            cv::Mat final_mask;
            cv::resize(parsing_mask, final_mask, face_crop.size(), 0, 0, cv::INTER_NEAREST);
            
            return final_mask;
        }
        catch (const std::exception& e) {
            std::cerr << "Error in face parsing: " << e.what() << std::endl;
            return cv::Mat::zeros(face_crop.size(), CV_8UC1);
        }
    }

    std::vector<MaskImage> parseFacesBatch(const std::vector<ImageData>& face_crops) {
        std::vector<MaskImage> results;
        results.reserve(face_crops.size());
        
        for (const auto& crop : face_crops) {
            results.push_back(parseFace(crop));
        }
        
        return results;
    }

    MaskImage getRegionMask(const MaskImage& full_mask, const std::vector<int>& region_indices) {
        if (full_mask.empty()) {
            return cv::Mat();
        }

        cv::Mat region_mask = cv::Mat::zeros(full_mask.size(), CV_8UC1);
        
        for (int region_idx : region_indices) {
            cv::Mat temp_mask;
            cv::inRange(full_mask, cv::Scalar(region_idx), cv::Scalar(region_idx), temp_mask);
            cv::bitwise_or(region_mask, temp_mask, region_mask);
        }
        
        return region_mask;
    }

    // Face parsing label definitions (based on CelebAMask-HQ)
    // 0: background, 1: skin, 2: nose, 3: eye_g (eye glasses), 4: l_eye, 5: r_eye,
    // 6: l_brow, 7: r_brow, 8: l_ear, 9: r_ear, 10: mouth, 11: u_lip, 12: l_lip,
    // 13: hair, 14: hat, 15: ear_r (ear ring), 16: neck_l (necklace), 17: neck, 18: cloth

    MaskImage getEyesMask(const MaskImage& full_mask) {
        return getRegionMask(full_mask, {4, 5}); // l_eye, r_eye
    }

    MaskImage getForeheadMask(const MaskImage& full_mask) {
        // Forehead is typically skin region above eyebrows
        cv::Mat skin_mask, brow_mask, forehead_mask;
        cv::inRange(full_mask, cv::Scalar(1), cv::Scalar(1), skin_mask); // skin
        cv::inRange(full_mask, cv::Scalar(6), cv::Scalar(7), brow_mask); // eyebrows
        
        // Dilate eyebrow mask to include forehead region
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
        cv::Mat dilated_brow;
        cv::dilate(brow_mask, dilated_brow, kernel);
        
        // Forehead is skin area above eyebrows
        cv::bitwise_and(skin_mask, dilated_brow, forehead_mask);
        
        return forehead_mask;
    }

    MaskImage getCheeksMask(const MaskImage& full_mask) {
        // Cheeks are skin regions excluding central features
        cv::Mat skin_mask, central_mask, cheeks_mask;
        cv::inRange(full_mask, cv::Scalar(1), cv::Scalar(1), skin_mask); // skin
        
        // Create central face mask (nose + mouth area)
        cv::Mat nose_mask, mouth_mask;
        cv::inRange(full_mask, cv::Scalar(2), cv::Scalar(2), nose_mask); // nose
        cv::inRange(full_mask, cv::Scalar(10), cv::Scalar(12), mouth_mask); // mouth area
        cv::bitwise_or(nose_mask, mouth_mask, central_mask);
        
        // Dilate central mask
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(20, 20));
        cv::Mat dilated_central;
        cv::dilate(central_mask, dilated_central, kernel);
        
        // Cheeks are skin minus central area
        cv::bitwise_and(skin_mask, ~dilated_central, cheeks_mask);
        
        return cheeks_mask;
    }

    MaskImage getMouthMask(const MaskImage& full_mask) {
        return getRegionMask(full_mask, {10, 11, 12}); // mouth, u_lip, l_lip
    }

    MaskImage getJawMask(const MaskImage& full_mask) {
        // Jaw is lower skin region
        cv::Mat skin_mask;
        cv::inRange(full_mask, cv::Scalar(1), cv::Scalar(1), skin_mask);
        
        // Create lower half mask
        cv::Mat lower_mask = cv::Mat::zeros(full_mask.size(), CV_8UC1);
        int start_row = static_cast<int>(full_mask.rows * 0.6); // Lower 40%
        lower_mask(cv::Rect(0, start_row, full_mask.cols, full_mask.rows - start_row)) = 255;
        
        cv::Mat jaw_mask;
        cv::bitwise_and(skin_mask, lower_mask, jaw_mask);
        
        return jaw_mask;
    }

    MaskImage getNeckMask(const MaskImage& full_mask) {
        return getRegionMask(full_mask, {17}); // neck
    }

    MaskImage getHairMask(const MaskImage& full_mask) {
        return getRegionMask(full_mask, {13}); // hair
    }

    MaskImage getEyebrowsMask(const MaskImage& full_mask) {
        return getRegionMask(full_mask, {6, 7}); // l_brow, r_brow
    }

private:
    cv::Mat convertToParseMask(const cv::Mat& network_output) {
        // Assuming network output is (1, num_classes, H, W)
        cv::Mat reshaped = network_output.reshape(1, network_output.size[2] * network_output.size[3]);
        
        cv::Mat parsing_mask = cv::Mat::zeros(network_output.size[2], network_output.size[3], CV_8UC1);
        
        // Find argmax for each pixel
        for (int i = 0; i < reshaped.rows; ++i) {
            cv::Point max_loc;
            cv::minMaxLoc(reshaped.row(i), nullptr, nullptr, nullptr, &max_loc);
            
            int y = i / network_output.size[3];
            int x = i % network_output.size[3];
            parsing_mask.at<uchar>(y, x) = static_cast<uchar>(max_loc.x);
        }
        
        return parsing_mask;
    }

    cv::dnn::Net net_;
    bool model_loaded_;
    int input_width_, input_height_;
};

FaceParser::FaceParser() : pImpl(std::make_unique<Impl>()) {}
FaceParser::~FaceParser() = default;

bool FaceParser::loadModel(const std::string& model_path) {
    return pImpl->loadModel(model_path);
}

MaskImage FaceParser::parseFace(const ImageData& face_crop) {
    return pImpl->parseFace(face_crop);
}

std::vector<MaskImage> FaceParser::parseFacesBatch(const std::vector<ImageData>& face_crops) {
    return pImpl->parseFacesBatch(face_crops);
}

MaskImage FaceParser::getEyesMask(const MaskImage& full_mask) {
    return pImpl->getEyesMask(full_mask);
}

MaskImage FaceParser::getForeheadMask(const MaskImage& full_mask) {
    return pImpl->getForeheadMask(full_mask);
}

MaskImage FaceParser::getCheeksMask(const MaskImage& full_mask) {
    return pImpl->getCheeksMask(full_mask);
}

MaskImage FaceParser::getMouthMask(const MaskImage& full_mask) {
    return pImpl->getMouthMask(full_mask);
}

MaskImage FaceParser::getJawMask(const MaskImage& full_mask) {
    return pImpl->getJawMask(full_mask);
}

MaskImage FaceParser::getNeckMask(const MaskImage& full_mask) {
    return pImpl->getNeckMask(full_mask);
}

MaskImage FaceParser::getHairMask(const MaskImage& full_mask) {
    return pImpl->getHairMask(full_mask);
}

MaskImage FaceParser::getEyebrowsMask(const MaskImage& full_mask) {
    return pImpl->getEyebrowsMask(full_mask);
}

void FaceParser::setInputSize(int width, int height) {
    pImpl->input_width_ = width;
    pImpl->input_height_ = height;
}

} // namespace ufra