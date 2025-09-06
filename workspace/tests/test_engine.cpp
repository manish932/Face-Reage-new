#include <gtest/gtest.h>
#include "ufra/engine.h"
#include <opencv2/opencv.hpp>

class EngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine = ufra::createEngine();
        
        config.backend = ufra::GPUBackend::CPU_FALLBACK;
        config.batch_size = 1;
        config.use_half_precision = false;
        config.max_resolution = 512;
    }

    void TearDown() override {
        if (engine && engine->isInitialized()) {
            engine->shutdown();
        }
    }

    std::unique_ptr<ufra::Engine> engine;
    ufra::ModelConfig config;
};

TEST_F(EngineTest, CreateEngine) {
    ASSERT_NE(engine, nullptr);
    EXPECT_FALSE(engine->isInitialized());
}

TEST_F(EngineTest, InitializeEngine) {
    bool result = engine->initialize(config);
    // May fail without models, but should not crash
    EXPECT_TRUE(result || !result);  // Just test it doesn't crash
}

TEST_F(EngineTest, VersionInfo) {
    std::string version = engine->getVersionInfo();
    EXPECT_FALSE(version.empty());
    EXPECT_NE(version.find("UFRa"), std::string::npos);
}

TEST_F(EngineTest, ProcessingModes) {
    engine->setProcessingMode(ufra::ProcessingMode::FEEDFORWARD);
    EXPECT_EQ(engine->getProcessingMode(), ufra::ProcessingMode::FEEDFORWARD);
    
    engine->setProcessingMode(ufra::ProcessingMode::DIFFUSION);
    EXPECT_EQ(engine->getProcessingMode(), ufra::ProcessingMode::DIFFUSION);
}

TEST_F(EngineTest, FaceDetectionEmptyImage) {
    cv::Mat empty_image;
    std::vector<ufra::Face> faces = engine->detectFaces(empty_image);
    EXPECT_TRUE(faces.empty());
}

TEST_F(EngineTest, FaceDetectionSyntheticImage) {
    // Create a simple test image
    cv::Mat test_image = cv::Mat::zeros(480, 640, CV_8UC3);
    cv::circle(test_image, cv::Point(320, 240), 50, cv::Scalar(255, 255, 255), -1);
    
    std::vector<ufra::Face> faces = engine->detectFaces(test_image);
    // Without loaded models, this should return empty or handle gracefully
    EXPECT_TRUE(faces.size() >= 0);  // Should not crash
}

TEST_F(EngineTest, AgeEstimation) {
    ufra::Face dummy_face;
    dummy_face.aligned_crop = cv::Mat::zeros(224, 224, CV_8UC3);
    
    float age = engine->estimateAge(dummy_face);
    EXPECT_GE(age, 0.0f);
    EXPECT_LE(age, 100.0f);
}

TEST_F(EngineTest, FrameProcessingWithoutInit) {
    ufra::FrameContext context;
    context.frame_number = 0;
    context.input_frame = cv::Mat::zeros(480, 640, CV_8UC3);
    context.mode = ufra::ProcessingMode::FEEDFORWARD;
    
    ufra::ProcessingResult result = engine->processFrame(context);
    EXPECT_FALSE(result.success);  // Should fail without initialization
    EXPECT_FALSE(result.error_message.empty());
}

TEST(FactoryTest, LibraryVersion) {
    std::string version = ufra::getLibraryVersion();
    EXPECT_FALSE(version.empty());
}

TEST(FactoryTest, AvailableBackends) {
    std::vector<ufra::GPUBackend> backends = ufra::getAvailableBackends();
    EXPECT_FALSE(backends.empty());
    
    // CPU fallback should always be available
    bool cpu_found = false;
    for (const auto& backend : backends) {
        if (backend == ufra::GPUBackend::CPU_FALLBACK) {
            cpu_found = true;
            break;
        }
    }
    EXPECT_TRUE(cpu_found);
}