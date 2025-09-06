#include "ufra/age_estimator.h"
#include <opencv2/dnn.hpp>
#include <iostream>

namespace ufra {

class AgeEstimator::Impl {
public:
    Impl() : model_loaded_(false), input_width_(224), input_height_(224),
             mean_(0.485f), std_(0.229f) {}

    bool loadModel(const std::string& model_path) {
        try {
            net_ = cv::dnn::readNet(model_path);
            if (net_.empty()) {
                std::cerr << "Failed to load age estimation model: " << model_path << std::endl;
                return false;
            }
            
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            
            model_loaded_ = true;
            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "Error loading age estimation model: " << e.what() << std::endl;
            return false;
        }
    }

    float estimateAge(const ImageData& face_crop) {
        if (!model_loaded_ || face_crop.empty()) {
            return 0.0f;
        }

        try {
            // Preprocess image
            cv::Mat resized;
            cv::resize(face_crop, resized, cv::Size(input_width_, input_height_));
            
            // Create blob
            cv::Mat blob;
            cv::dnn::blobFromImage(resized, blob, 1.0/255.0, 
                                   cv::Size(input_width_, input_height_), 
                                   cv::Scalar(mean_, mean_, mean_), true, false);
            
            net_.setInput(blob);
            
            // Run inference
            cv::Mat output = net_.forward();
            
            // Extract age prediction (assuming regression output)
            float predicted_age = output.at<float>(0, 0);
            
            // Clamp to reasonable range
            return std::max(0.0f, std::min(100.0f, predicted_age));
        }
        catch (const std::exception& e) {
            std::cerr << "Error in age estimation: " << e.what() << std::endl;
            return 0.0f;
        }
    }

    std::vector<float> estimateAgeBatch(const std::vector<ImageData>& face_crops) {
        std::vector<float> ages;
        ages.reserve(face_crops.size());
        
        for (const auto& crop : face_crops) {
            ages.push_back(estimateAge(crop));
        }
        
        return ages;
    }

    cv::dnn::Net net_;
    bool model_loaded_;
    int input_width_, input_height_;
    float mean_, std_;
};

AgeEstimator::AgeEstimator() : pImpl(std::make_unique<Impl>()) {}
AgeEstimator::~AgeEstimator() = default;

bool AgeEstimator::loadModel(const std::string& model_path) {
    return pImpl->loadModel(model_path);
}

float AgeEstimator::estimateAge(const ImageData& face_crop) {
    return pImpl->estimateAge(face_crop);
}

std::vector<float> AgeEstimator::estimateAgeBatch(const std::vector<ImageData>& face_crops) {
    return pImpl->estimateAgeBatch(face_crops);
}

void AgeEstimator::setInputSize(int width, int height) {
    pImpl->input_width_ = width;
    pImpl->input_height_ = height;
}

void AgeEstimator::setNormalization(float mean, float std) {
    pImpl->mean_ = mean;
    pImpl->std_ = std;
}

} // namespace ufra