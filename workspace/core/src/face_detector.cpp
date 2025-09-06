#include "ufra/face_detector.h"
#include <iostream>

#ifdef OPENCV_FOUND
#include <opencv2/dnn.hpp>
#else
// Mock OpenCV functionality when not available
namespace cv {
    class Mat {
    public:
        Mat() {}
        Mat(int rows, int cols, int type) : rows(rows), cols(cols) {}
        bool empty() const { return rows == 0 || cols == 0; }
        int rows = 0, cols = 0;
    };
    
    class Rect {
    public:
        Rect(int x, int y, int w, int h) : x(x), y(y), width(w), height(h) {}
        int x, y, width, height;
    };
    
    namespace dnn {
        class Net {
        public:
            bool empty() const { return true; }
            void setPreferableBackend(int) {}
            void setPreferableTarget(int) {}
            void setInput(const Mat&) {}
            Mat forward() { return Mat(); }
        };
        
        Net readNet(const std::string&) { return Net(); }
        void blobFromImage(const Mat&, Mat&, double, const cv::Size&, const cv::Scalar&, bool, bool) {}
        void NMSBoxes(const std::vector<Rect>&, const std::vector<float>&, float, float, std::vector<int>&) {}
        
        const int DNN_BACKEND_CUDA = 0;
        const int DNN_TARGET_CUDA = 0;
    }
    
    class Size {
    public:
        Size(int w, int h) : width(w), height(h) {}
        int width, height;
    };
    
    class Scalar {
    public:
        Scalar(double v0, double v1, double v2) {}
    };
}
#endif

namespace ufra {

class FaceDetector::Impl {
public:
    Impl() : confidence_threshold_(0.7f), nms_threshold_(0.4f), max_faces_(10) {}

    bool loadModel(const std::string& model_path) {
        try {
            net_ = cv::dnn::readNet(model_path);
            if (net_.empty()) {
                std::cerr << "Failed to load face detection model: " << model_path << std::endl;
                return false;
            }
            
            // Set backend and target
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            
            model_loaded_ = true;
            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "Error loading face detection model: " << e.what() << std::endl;
            return false;
        }
    }

    std::vector<Face> detectFaces(const ImageData& image) {
        std::vector<Face> faces;
        
        if (!model_loaded_ || image.empty()) {
            return faces;
        }

        try {
            // Prepare input blob
            cv::Mat blob;
            cv::dnn::blobFromImage(image, blob, 1.0, cv::Size(640, 640), 
                                   cv::Scalar(104, 117, 123), false, false);
            
            net_.setInput(blob);
            
            // Run inference
            std::vector<cv::Mat> outputs;
            net_.forward(outputs, net_.getUnconnectedOutLayersNames());
            
            // Parse detections
            if (!outputs.empty()) {
                cv::Mat detection = outputs[0];
                
                std::vector<cv::Rect> boxes;
                std::vector<float> confidences;
                
                for (int i = 0; i < detection.rows; i++) {
                    float confidence = detection.at<float>(i, 2);
                    
                    if (confidence > confidence_threshold_) {
                        int x1 = static_cast<int>(detection.at<float>(i, 3) * image.cols);
                        int y1 = static_cast<int>(detection.at<float>(i, 4) * image.rows);
                        int x2 = static_cast<int>(detection.at<float>(i, 5) * image.cols);
                        int y2 = static_cast<int>(detection.at<float>(i, 6) * image.rows);
                        
                        boxes.emplace_back(x1, y1, x2 - x1, y2 - y1);
                        confidences.push_back(confidence);
                    }
                }
                
                // Apply NMS
                std::vector<int> indices;
                cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold_, 
                                  nms_threshold_, indices);
                
                // Create Face objects
                for (size_t i = 0; i < indices.size() && i < max_faces_; i++) {
                    int idx = indices[i];
                    Face face;
                    
                    face.box.x = static_cast<float>(boxes[idx].x);
                    face.box.y = static_cast<float>(boxes[idx].y);
                    face.box.width = static_cast<float>(boxes[idx].width);
                    face.box.height = static_cast<float>(boxes[idx].height);
                    face.box.confidence = confidences[idx];
                    face.box.face_id = static_cast<int>(i);
                    
                    // Extract face crop with padding
                    int padding = 50;
                    cv::Rect crop_rect(
                        std::max(0, boxes[idx].x - padding),
                        std::max(0, boxes[idx].y - padding),
                        std::min(image.cols - std::max(0, boxes[idx].x - padding), 
                                boxes[idx].width + 2 * padding),
                        std::min(image.rows - std::max(0, boxes[idx].y - padding), 
                                boxes[idx].height + 2 * padding)
                    );
                    
                    face.aligned_crop = image(crop_rect).clone();
                    
                    // Create identity transform matrix for now
                    face.transform_matrix = cv::Mat::eye(2, 3, CV_32F);
                    
                    faces.push_back(face);
                }
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Error in face detection: " << e.what() << std::endl;
        }
        
        return faces;
    }

    cv::dnn::Net net_;
    bool model_loaded_ = false;
    float confidence_threshold_;
    float nms_threshold_;
    int max_faces_;
};

FaceDetector::FaceDetector() : pImpl(std::make_unique<Impl>()) {}
FaceDetector::~FaceDetector() = default;

bool FaceDetector::loadModel(const std::string& model_path) {
    return pImpl->loadModel(model_path);
}

std::vector<Face> FaceDetector::detectFaces(const ImageData& image) {
    return pImpl->detectFaces(image);
}

void FaceDetector::setConfidenceThreshold(float threshold) {
    pImpl->confidence_threshold_ = threshold;
}

void FaceDetector::setNMSThreshold(float threshold) {
    pImpl->nms_threshold_ = threshold;
}

void FaceDetector::setMaxFaces(int max_faces) {
    pImpl->max_faces_ = max_faces;
}

} // namespace ufra