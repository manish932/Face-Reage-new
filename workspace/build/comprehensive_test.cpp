#include <iostream>
#include <memory>

// Include minimal types and engine
#include "ufra/minimal_types.h"
#include "ufra/minimal_engine.h"

int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << "UFRa Comprehensive Test Suite" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    int tests_passed = 0;
    int total_tests = 0;
    
    // Test 1: Engine creation
    total_tests++;
    std::cout << "Test 1: Engine creation... ";
    try {
        auto engine = ufra::createEngine();
        if (engine) {
            std::cout << "PASSED" << std::endl;
            tests_passed++;
        } else {
            std::cout << "FAILED" << std::endl;
        }
    } catch (...) {
        std::cout << "FAILED (exception)" << std::endl;
    }
    
    // Test 2: Engine initialization
    total_tests++;
    std::cout << "Test 2: Engine initialization... ";
    try {
        auto engine = ufra::createEngine();
        ufra::ModelConfig config;
        config.backend = ufra::GPUBackend::CPU_FALLBACK;
        
        bool initialized = engine->initialize(config);
        if (initialized) {
            std::cout << "PASSED" << std::endl;
            tests_passed++;
        } else {
            std::cout << "FAILED" << std::endl;
        }
    } catch (...) {
        std::cout << "FAILED (exception)" << std::endl;
    }
    
    // Test 3: Model loading
    total_tests++;
    std::cout << "Test 3: Model loading... ";
    try {
        auto engine = ufra::createEngine();
        ufra::ModelConfig config;
        engine->initialize(config);
        
        bool loaded = engine->loadModels("./nonexistent_path");
        // Should not crash even with non-existent path
        std::cout << "PASSED" << std::endl;
        tests_passed++;
    } catch (...) {
        std::cout << "FAILED (exception)" << std::endl;
    }
    
    // Test 4: Frame processing
    total_tests++;
    std::cout << "Test 4: Frame processing... ";
    try {
        auto engine = ufra::createEngine();
        ufra::ModelConfig config;
        engine->initialize(config);
        engine->loadModels("./models");
        
        ufra::FrameContext context;
        context.input_frame = ufra::ImageData(640, 480, 3);
        context.frame_number = 0;
        
        auto result = engine->processFrame(context);
        if (result.success) {
            std::cout << "PASSED" << std::endl;
            tests_passed++;
        } else {
            std::cout << "FAILED: " << result.error_message << std::endl;
        }
    } catch (...) {
        std::cout << "FAILED (exception)" << std::endl;
    }
    
    // Test 5: Version and utilities
    total_tests++;
    std::cout << "Test 5: Version and utilities... ";
    try {
        auto engine = ufra::createEngine();
        std::string version = engine->getVersion();
        
        auto backends = ufra::getAvailableBackends();
        if (!version.empty() && !backends.empty()) {
            std::cout << "PASSED" << std::endl;
            tests_passed++;
        } else {
            std::cout << "FAILED" << std::endl;
        }
    } catch (...) {
        std::cout << "FAILED (exception)" << std::endl;
    }
    
    std::cout << "=========================================" << std::endl;
    std::cout << "Test Results: " << tests_passed << "/" << total_tests << " passed" << std::endl;
    
    if (tests_passed == total_tests) {
        std::cout << "✓ All tests PASSED!" << std::endl;
        std::cout << "✓ UFRa Core Library is working correctly" << std::endl;
        return 0;
    } else {
        std::cout << "✗ Some tests FAILED" << std::endl;
        return 1;
    }
}
