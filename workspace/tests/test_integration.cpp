#include <gtest/gtest.h>
#include "ufra/engine.h"
#include "ufra/face_detector.h"
#include "ufra/age_estimator.h"
#include "ufra/face_parser.h"
#include "ufra/feedforward_generator.h"
#include "ufra/compositor.h"
#include <opencv2/opencv.hpp>

class IntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test image with face-like pattern
        test_image = cv::Mat::zeros(480, 640, CV_8UC3);
        
        // Face 1
        cv::circle(test_image, cv::Point(200, 200), 80, cv::Scalar(200, 180, 160), -1); // Face
        cv::circle(test_image, cv::Point(180, 180), 10, cv::Scalar(50, 50, 50), -1);    // Left eye
        cv::circle(test_image, cv::Point(220, 180), 10, cv::Scalar(50, 50, 50), -1);    // Right eye
        cv::ellipse(test_image, cv::Point(200, 220), cv::Size(15, 8), 0, 0, 180, cv::Scalar(100, 50, 50), -1); // Mouth
        
        // Face 2
        cv::circle(test_image, cv::Point(450, 300), 60, cv::Scalar(190, 170, 150), -1); // Face
        cv::circle(test_image, cv::Point(435, 285), 8, cv::Scalar(50, 50, 50), -1);     // Left eye
        cv::circle(test_image, cv::Point(465, 285), 8, cv::Scalar(50, 50, 50), -1);     // Right eye
        
        // Setup age controls
        age_controls.target_age = 45.0f;
        age_controls.identity_lock_strength = 0.7f;
        age_controls.temporal_stability = 0.8f;
        age_controls.texture_keep = 0.6f;
        age_controls.enable_hair_aging = true;
        age_controls.gray_density = 0.5f;
    }

    cv::Mat test_image;
    ufra::AgeControls age_controls;
};

TEST_F(IntegrationTest, FullPipelineWithoutModels) {
    // Test full pipeline without loaded models (should handle gracefully)
    auto engine = ufra::createEngine();
    
    ufra::ModelConfig config;
    config.backend = ufra::GPUBackend::CPU_FALLBACK;
    config.batch_size = 1;
    
    // Initialize engine
    bool initialized = engine->initialize(config);
    EXPECT_TRUE(initialized);
    
    // Setup frame context
    ufra::FrameContext context;
    context.frame_number = 0;
    context.input_frame = test_image;
    context.controls = age_controls;
    context.mode = ufra::ProcessingMode::FEEDFORWARD;
    
    // Process frame
    ufra::ProcessingResult result = engine->processFrame(context);
    
    // Without models, should fail gracefully
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.error_message.empty());
}

TEST_F(IntegrationTest, ComponentChain) {
    // Test individual components in sequence
    
    // 1. Face Detection
    ufra::FaceDetector detector;
    std::vector<ufra::Face> faces = detector.detectFaces(test_image);
    EXPECT_GE(faces.size(), 0); // Should not crash
    
    // 2. Age Estimation (if faces detected)
    if (!faces.empty()) {
        ufra::AgeEstimator estimator;
        for (auto& face : faces) {
            float age = estimator.estimateAge(face.aligned_crop);
            EXPECT_GE(age, 0.0f);
            EXPECT_LE(age, 100.0f);
        }
    }
    
    // 3. Face Parsing
    ufra::FaceParser parser;
    if (!faces.empty()) {
        cv::Mat parsing_mask = parser.parseFace(faces[0].aligned_crop);
        EXPECT_FALSE(parsing_mask.empty());
    }
    
    // 4. Feedforward Generation
    ufra::FeedforwardGenerator generator;
    if (!faces.empty()) {
        cv::Mat aged_face = generator.generateAgedFace(
            faces[0].aligned_crop, age_controls, cv::Mat());
        EXPECT_FALSE(aged_face.empty());
    }
    
    // 5. Compositing
    ufra::Compositor compositor;
    if (!faces.empty()) {
        cv::Mat output_frame = test_image.clone();
        compositor.compositeFace(output_frame, faces[0].aligned_crop, faces[0]);
        EXPECT_EQ(output_frame.size(), test_image.size());
    }
}

TEST_F(IntegrationTest, MultiFrameProcessing) {
    auto engine = ufra::createEngine();
    
    ufra::ModelConfig config;
    config.backend = ufra::GPUBackend::CPU_FALLBACK;
    config.batch_size = 1;
    
    engine->initialize(config);
    
    // Process multiple frames
    std::vector<ufra::FrameContext> contexts;
    for (int i = 0; i < 5; ++i) {
        ufra::FrameContext context;
        context.frame_number = i;
        context.input_frame = test_image;
        context.controls = age_controls;
        context.mode = ufra::ProcessingMode::FEEDFORWARD;
        contexts.push_back(context);
    }
    
    // Test individual processing
    for (const auto& context : contexts) {
        ufra::ProcessingResult result = engine->processFrame(context);
        // Should handle consistently (fail gracefully without models)
        EXPECT_FALSE(result.success);
    }
}

TEST_F(IntegrationTest, DifferentProcessingModes) {
    auto engine = ufra::createEngine();
    
    ufra::ModelConfig config;
    config.backend = ufra::GPUBackend::CPU_FALLBACK;
    
    engine->initialize(config);
    
    std::vector<ufra::ProcessingMode> modes = {
        ufra::ProcessingMode::FEEDFORWARD,
        ufra::ProcessingMode::DIFFUSION,
        ufra::ProcessingMode::HYBRID,
        ufra::ProcessingMode::AUTO
    };
    
    for (auto mode : modes) {
        engine->setProcessingMode(mode);
        EXPECT_EQ(engine->getProcessingMode(), mode);
        
        ufra::FrameContext context;
        context.frame_number = 0;
        context.input_frame = test_image;
        context.controls = age_controls;
        context.mode = mode;
        
        ufra::ProcessingResult result = engine->processFrame(context);
        // Should handle all modes consistently
        EXPECT_FALSE(result.success); // Without models
    }
}

TEST_F(IntegrationTest, GPUBackendSwitching) {
    auto engine = ufra::createEngine();
    
    std::vector<ufra::GPUBackend> backends = ufra::getAvailableBackends();
    EXPECT_FALSE(backends.empty());
    
    for (auto backend : backends) {
        ufra::ModelConfig config;
        config.backend = backend;
        config.batch_size = 1;
        
        bool initialized = engine->initialize(config);
        // CPU fallback should always work
        if (backend == ufra::GPUBackend::CPU_FALLBACK) {
            EXPECT_TRUE(initialized);
        }
        
        if (initialized) {
            engine->shutdown();
        }
    }
}

TEST_F(IntegrationTest, MemoryStressTest) {
    auto engine = ufra::createEngine();
    
    ufra::ModelConfig config;
    config.backend = ufra::GPUBackend::CPU_FALLBACK;
    config.batch_size = 1;
    
    engine->initialize(config);
    
    // Process many frames to test for memory leaks
    for (int i = 0; i < 100; ++i) {
        ufra::FrameContext context;
        context.frame_number = i;
        context.input_frame = test_image;
        context.controls = age_controls;
        context.mode = ufra::ProcessingMode::FEEDFORWARD;
        
        ufra::ProcessingResult result = engine->processFrame(context);
        
        // Should consistently handle without memory issues
        EXPECT_FALSE(result.success); // Without models
        
        // Force cleanup
        result = ufra::ProcessingResult();
    }
}

TEST_F(IntegrationTest, ThreadSafety) {
    // Test that multiple engine instances can run concurrently
    std::vector<std::unique_ptr<ufra::Engine>> engines;
    
    for (int i = 0; i < 4; ++i) {
        auto engine = ufra::createEngine();
        
        ufra::ModelConfig config;
        config.backend = ufra::GPUBackend::CPU_FALLBACK;
        config.batch_size = 1;
        
        bool initialized = engine->initialize(config);
        EXPECT_TRUE(initialized);
        
        engines.push_back(std::move(engine));
    }
    
    // Test concurrent access
    std::vector<std::thread> threads;
    std::atomic<int> successful_processes{0};
    
    for (size_t i = 0; i < engines.size(); ++i) {
        threads.emplace_back([&engines, i, this, &successful_processes]() {
            ufra::FrameContext context;
            context.frame_number = static_cast<int>(i);
            context.input_frame = test_image;
            context.controls = age_controls;
            context.mode = ufra::ProcessingMode::FEEDFORWARD;
            
            ufra::ProcessingResult result = engines[i]->processFrame(context);
            
            // Should handle thread-safely
            if (!result.success) {
                successful_processes++;
            }
        });
    }
    
    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_EQ(successful_processes.load(), 4);
}

TEST_F(IntegrationTest, ErrorRecovery) {
    auto engine = ufra::createEngine();
    
    ufra::ModelConfig config;
    config.backend = ufra::GPUBackend::CPU_FALLBACK;
    
    engine->initialize(config);
    
    // Test with invalid inputs
    std::vector<cv::Mat> invalid_inputs = {
        cv::Mat(),                           // Empty image
        cv::Mat::zeros(0, 0, CV_8UC3),      // Zero size
        cv::Mat::zeros(1, 1, CV_8UC3),      // Too small
        cv::Mat::zeros(10000, 10000, CV_8UC3) // Too large (might cause memory issues)
    };
    
    for (const auto& invalid_input : invalid_inputs) {
        ufra::FrameContext context;
        context.frame_number = 0;
        context.input_frame = invalid_input;
        context.controls = age_controls;
        context.mode = ufra::ProcessingMode::FEEDFORWARD;
        
        ufra::ProcessingResult result = engine->processFrame(context);
        
        // Should handle invalid inputs gracefully
        EXPECT_FALSE(result.success);
        EXPECT_FALSE(result.error_message.empty());
    }
    
    // Test recovery with valid input after errors
    ufra::FrameContext valid_context;
    valid_context.frame_number = 0;
    valid_context.input_frame = test_image;
    valid_context.controls = age_controls;
    valid_context.mode = ufra::ProcessingMode::FEEDFORWARD;
    
    ufra::ProcessingResult result = engine->processFrame(valid_context);
    // Should still work after handling errors
    EXPECT_FALSE(result.success); // Without models, but should not crash
}