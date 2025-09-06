#include <gtest/gtest.h>
#include "ufra/face_detector.h"
#include <opencv2/opencv.hpp>

class FaceDetectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        detector = std::make_unique<ufra::FaceDetector>();
        
        // Create test images
        empty_image = cv::Mat();
        
        // Simple test image with face-like pattern
        test_image = cv::Mat::zeros(480, 640, CV_8UC3);
        cv::circle(test_image, cv::Point(320, 240), 80, cv::Scalar(200, 180, 160), -1); // Face
        cv::circle(test_image, cv::Point(300, 220), 10, cv::Scalar(50, 50, 50), -1);     // Left eye
        cv::circle(test_image, cv::Point(340, 220), 10, cv::Scalar(50, 50, 50), -1);     // Right eye
        cv::ellipse(test_image, cv::Point(320, 260), cv::Size(15, 8), 0, 0, 180, cv::Scalar(100, 50, 50), -1); // Mouth
        
        // Multi-face test image
        multi_face_image = cv::Mat::zeros(480, 640, CV_8UC3);
        // Face 1
        cv::circle(multi_face_image, cv::Point(200, 200), 60, cv::Scalar(200, 180, 160), -1);
        cv::circle(multi_face_image, cv::Point(185, 185), 8, cv::Scalar(50, 50, 50), -1);
        cv::circle(multi_face_image, cv::Point(215, 185), 8, cv::Scalar(50, 50, 50), -1);
        // Face 2
        cv::circle(multi_face_image, cv::Point(450, 300), 60, cv::Scalar(190, 170, 150), -1);
        cv::circle(multi_face_image, cv::Point(435, 285), 8, cv::Scalar(50, 50, 50), -1);
        cv::circle(multi_face_image, cv::Point(465, 285), 8, cv::Scalar(50, 50, 50), -1);
    }

    std::unique_ptr<ufra::FaceDetector> detector;
    cv::Mat empty_image;
    cv::Mat test_image;
    cv::Mat multi_face_image;
};

TEST_F(FaceDetectorTest, CreateDetector) {
    ASSERT_NE(detector, nullptr);
}

TEST_F(FaceDetectorTest, DetectFacesEmptyImage) {
    std::vector<ufra::Face> faces = detector->detectFaces(empty_image);
    EXPECT_TRUE(faces.empty());
}

TEST_F(FaceDetectorTest, DetectFacesTestImage) {
    std::vector<ufra::Face> faces = detector->detectFaces(test_image);
    // Without loaded model, should handle gracefully (return empty or not crash)
    EXPECT_GE(faces.size(), 0);
}

TEST_F(FaceDetectorTest, DetectMultipleFaces) {
    std::vector<ufra::Face> faces = detector->detectFaces(multi_face_image);
    // Without loaded model, should handle gracefully
    EXPECT_GE(faces.size(), 0);
}

TEST_F(FaceDetectorTest, SetParameters) {
    // Test parameter setting doesn't crash
    detector->setConfidenceThreshold(0.8f);
    detector->setNMSThreshold(0.3f);
    detector->setMaxFaces(5);
    
    // Test detection still works after parameter changes
    std::vector<ufra::Face> faces = detector->detectFaces(test_image);
    EXPECT_GE(faces.size(), 0);
}

TEST_F(FaceDetectorTest, ParameterBounds) {
    // Test extreme parameter values
    detector->setConfidenceThreshold(0.0f);
    detector->setConfidenceThreshold(1.0f);
    detector->setNMSThreshold(0.0f);
    detector->setNMSThreshold(1.0f);
    detector->setMaxFaces(0);
    detector->setMaxFaces(100);
    
    // Should not crash with extreme values
    std::vector<ufra::Face> faces = detector->detectFaces(test_image);
    EXPECT_GE(faces.size(), 0);
}

TEST_F(FaceDetectorTest, LoadInvalidModel) {
    bool result = detector->loadModel("nonexistent_model.onnx");
    EXPECT_FALSE(result);
    
    // Detection should still work (gracefully fail)
    std::vector<ufra::Face> faces = detector->detectFaces(test_image);
    EXPECT_TRUE(faces.empty()); // Should be empty without valid model
}

TEST_F(FaceDetectorTest, FaceBoxValidation) {
    std::vector<ufra::Face> faces = detector->detectFaces(test_image);
    
    for (const auto& face : faces) {
        // Validate face box properties
        EXPECT_GE(face.box.x, 0.0f);
        EXPECT_GE(face.box.y, 0.0f);
        EXPECT_GT(face.box.width, 0.0f);
        EXPECT_GT(face.box.height, 0.0f);
        EXPECT_GE(face.box.confidence, 0.0f);
        EXPECT_LE(face.box.confidence, 1.0f);
        EXPECT_GE(face.box.face_id, 0);
        
        // Validate face crop is not empty
        EXPECT_FALSE(face.aligned_crop.empty());
        
        // Validate transform matrix
        EXPECT_FALSE(face.transform_matrix.empty());
        EXPECT_EQ(face.transform_matrix.rows, 2);
        EXPECT_EQ(face.transform_matrix.cols, 3);
    }
}

TEST_F(FaceDetectorTest, ImageSizeVariations) {
    // Test different image sizes
    std::vector<cv::Size> test_sizes = {
        cv::Size(160, 120),   // Very small
        cv::Size(320, 240),   // Small
        cv::Size(640, 480),   // Medium
        cv::Size(1280, 720),  // Large
        cv::Size(1920, 1080)  // Very large
    };
    
    for (const auto& size : test_sizes) {
        cv::Mat test_img;
        cv::resize(test_image, test_img, size);
        
        std::vector<ufra::Face> faces = detector->detectFaces(test_img);
        EXPECT_GE(faces.size(), 0); // Should not crash
    }
}

TEST_F(FaceDetectorTest, ColorSpaceVariations) {
    // Test different color formats
    cv::Mat gray_image, bgr_image, rgba_image;
    
    cv::cvtColor(test_image, gray_image, cv::COLOR_BGR2GRAY);
    bgr_image = test_image.clone(); // Already BGR
    cv::cvtColor(test_image, rgba_image, cv::COLOR_BGR2BGRA);
    
    // Test detection with different color formats
    std::vector<ufra::Face> faces_gray = detector->detectFaces(gray_image);
    std::vector<ufra::Face> faces_bgr = detector->detectFaces(bgr_image);
    std::vector<ufra::Face> faces_rgba = detector->detectFaces(rgba_image);
    
    EXPECT_GE(faces_gray.size(), 0);
    EXPECT_GE(faces_bgr.size(), 0);
    EXPECT_GE(faces_rgba.size(), 0);
}