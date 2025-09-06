#include <gtest/gtest.h>
#include "ufra/compositor.h"
#include "ufra/types.h"
#include <opencv2/opencv.hpp>

class CompositorTest : public ::testing::Test {
protected:
    void SetUp() override {
        compositor = std::make_unique<ufra::Compositor>();
        
        // Create test frame (640x480)
        target_frame = cv::Mat::zeros(480, 640, CV_8UC3);
        cv::rectangle(target_frame, cv::Point(0, 0), cv::Point(640, 480), cv::Scalar(100, 100, 100), -1);
        
        // Create processed face crop (128x128)
        processed_face = cv::Mat::zeros(128, 128, CV_8UC3);
        cv::circle(processed_face, cv::Point(64, 64), 50, cv::Scalar(200, 180, 160), -1);
        
        // Create face info
        face_info.box.x = 256.0f;
        face_info.box.y = 176.0f;
        face_info.box.width = 128.0f;
        face_info.box.height = 128.0f;
        face_info.box.confidence = 0.9f;
        face_info.box.face_id = 1;
        
        // Create identity transform matrix
        face_info.transform_matrix = cv::Mat::eye(2, 3, CV_32F);
        face_info.aligned_crop = processed_face.clone();
    }

    std::unique_ptr<ufra::Compositor> compositor;
    cv::Mat target_frame;
    cv::Mat processed_face;
    ufra::Face face_info;
};

TEST_F(CompositorTest, CreateCompositor) {
    ASSERT_NE(compositor, nullptr);
}

TEST_F(CompositorTest, CompositeFaceBasic) {
    cv::Mat original_frame = target_frame.clone();
    
    // Composite face onto frame
    compositor->compositeFace(target_frame, processed_face, face_info);
    
    // Frame should be modified
    cv::Mat diff;
    cv::absdiff(original_frame, target_frame, diff);
    cv::Scalar total_diff = cv::sum(diff);
    
    // Should have some difference after compositing
    EXPECT_GT(total_diff[0] + total_diff[1] + total_diff[2], 0);
}

TEST_F(CompositorTest, CompositeEmptyFace) {
    cv::Mat empty_face;
    cv::Mat original_frame = target_frame.clone();
    
    // Should handle empty face gracefully
    compositor->compositeFace(target_frame, empty_face, face_info);
    
    // Frame should remain unchanged with empty input
    cv::Mat diff;
    cv::absdiff(original_frame, target_frame, diff);
    cv::Scalar total_diff = cv::sum(diff);
    
    EXPECT_EQ(total_diff[0] + total_diff[1] + total_diff[2], 0);
}

TEST_F(CompositorTest, CompositeOutOfBounds) {
    // Set face position outside frame bounds
    face_info.box.x = 700.0f;  // Outside 640 width
    face_info.box.y = 500.0f;  // Outside 480 height
    
    cv::Mat original_frame = target_frame.clone();
    
    // Should handle out-of-bounds gracefully
    compositor->compositeFace(target_frame, processed_face, face_info);
    
    // Should not crash and frame might be unchanged
    EXPECT_EQ(target_frame.size(), original_frame.size());
}

TEST_F(CompositorTest, SetBlendingModes) {
    // Test different blending modes don't crash
    compositor->setBlendingMode("linear");
    compositor->compositeFace(target_frame, processed_face, face_info);
    
    compositor->setBlendingMode("poisson");
    compositor->compositeFace(target_frame, processed_face, face_info);
    
    compositor->setBlendingMode("multiband");
    compositor->compositeFace(target_frame, processed_face, face_info);
    
    compositor->setBlendingMode("invalid_mode");
    compositor->compositeFace(target_frame, processed_face, face_info);
}

TEST_F(CompositorTest, SetFeatherRadius) {
    // Test different feather radius values
    std::vector<int> radii = {0, 5, 10, 20, 50};
    
    for (int radius : radii) {
        compositor->setFeatherRadius(radius);
        
        cv::Mat test_frame = target_frame.clone();
        compositor->compositeFace(test_frame, processed_face, face_info);
        
        // Should not crash with different radii
        EXPECT_EQ(test_frame.size(), target_frame.size());
    }
}

TEST_F(CompositorTest, EnableColorCorrection) {
    compositor->enableColorCorrection(true);
    cv::Mat frame_with_correction = target_frame.clone();
    compositor->compositeFace(frame_with_correction, processed_face, face_info);
    
    compositor->enableColorCorrection(false);
    cv::Mat frame_without_correction = target_frame.clone();
    compositor->compositeFace(frame_without_correction, processed_face, face_info);
    
    // Results might be different with/without color correction
    EXPECT_EQ(frame_with_correction.size(), frame_without_correction.size());
}

TEST_F(CompositorTest, SetDetailReinjectionStrength) {
    std::vector<float> strengths = {0.0f, 0.5f, 1.0f, 1.5f, -0.5f};
    
    for (float strength : strengths) {
        compositor->setDetailReinjectionStrength(strength);
        
        cv::Mat test_frame = target_frame.clone();
        compositor->compositeFace(test_frame, processed_face, face_info);
        
        // Should handle all strength values gracefully
        EXPECT_EQ(test_frame.size(), target_frame.size());
    }
}

TEST_F(CompositorTest, MultipleFaceCompositing) {
    // Create multiple faces
    std::vector<ufra::Face> faces;
    std::vector<cv::Mat> processed_faces;
    
    for (int i = 0; i < 3; ++i) {
        ufra::Face face = face_info;
        face.box.x = 100.0f + i * 150.0f;
        face.box.y = 100.0f + i * 50.0f;
        face.box.face_id = i;
        
        cv::Mat proc_face = processed_face.clone();
        cv::circle(proc_face, cv::Point(64, 64), 40, cv::Scalar(180 + i*20, 160, 140), -1);
        
        faces.push_back(face);
        processed_faces.push_back(proc_face);
    }
    
    cv::Mat original_frame = target_frame.clone();
    
    // Composite all faces
    for (size_t i = 0; i < faces.size(); ++i) {
        compositor->compositeFace(target_frame, processed_faces[i], faces[i]);
    }
    
    // Should have changes from multiple faces
    cv::Mat diff;
    cv::absdiff(original_frame, target_frame, diff);
    cv::Scalar total_diff = cv::sum(diff);
    
    EXPECT_GT(total_diff[0] + total_diff[1] + total_diff[2], 0);
}

TEST_F(CompositorTest, SizeConsistency) {
    cv::Size original_size = target_frame.size();
    
    compositor->compositeFace(target_frame, processed_face, face_info);
    
    // Frame size should remain the same
    EXPECT_EQ(target_frame.size(), original_size);
    EXPECT_EQ(target_frame.type(), CV_8UC3);
}

TEST_F(CompositorTest, DifferentFaceSizes) {
    std::vector<cv::Size> face_sizes = {
        cv::Size(64, 64),
        cv::Size(128, 128),
        cv::Size(256, 256),
        cv::Size(100, 150),  // Non-square
        cv::Size(200, 100)   // Wide
    };
    
    for (const auto& size : face_sizes) {
        cv::Mat test_face;
        cv::resize(processed_face, test_face, size);
        
        // Update face info for new size
        ufra::Face test_face_info = face_info;
        test_face_info.box.width = static_cast<float>(size.width);
        test_face_info.box.height = static_cast<float>(size.height);
        
        cv::Mat test_frame = target_frame.clone();
        compositor->compositeFace(test_frame, test_face, test_face_info);
        
        // Should handle different face sizes
        EXPECT_EQ(test_frame.size(), target_frame.size());
    }
}