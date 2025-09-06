#include "ufra/engine.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

void printUsage() {
    std::cout << "UFRa CLI - Universal Face Re-Aging Command Line Interface\n";
    std::cout << "Usage: ufra_cli [options]\n";
    std::cout << "\nOptions:\n";
    std::cout << "  -i, --input <path>      Input video file or image sequence\n";
    std::cout << "  -o, --output <path>     Output video file or image sequence\n";
    std::cout << "  -a, --age <value>       Target age (0-100)\n";
    std::cout << "  -m, --mode <mode>       Processing mode (feedforward|diffusion|hybrid|auto)\n";
    std::cout << "  --models <path>         Path to model directory\n";
    std::cout << "  --gpu <backend>         GPU backend (cuda|metal|directml|cpu)\n";
    std::cout << "  --batch-size <size>     Batch size for processing\n";
    std::cout << "  --identity-lock <val>   Identity preservation strength (0.0-1.0)\n";
    std::cout << "  --temporal-stability    Enable temporal stability\n";
    std::cout << "  --help                  Show this help message\n";
    std::cout << "\nExamples:\n";
    std::cout << "  ufra_cli -i input.mp4 -o output.mp4 -a 25 -m feedforward\n";
    std::cout << "  ufra_cli -i frame_%04d.jpg -o aged_%04d.jpg -a 65 --identity-lock 0.8\n";
}

struct CLIConfig {
    std::string input_path;
    std::string output_path;
    std::string models_path = "/usr/local/share/ufra/models";
    float target_age = 30.0f;
    ufra::ProcessingMode mode = ufra::ProcessingMode::FEEDFORWARD;
    ufra::GPUBackend gpu_backend = ufra::GPUBackend::CUDA;
    int batch_size = 1;
    float identity_lock = 0.5f;
    bool temporal_stability = true;
    bool help = false;
};

CLIConfig parseArguments(int argc, char* argv[]) {
    CLIConfig config;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            config.help = true;
        } else if ((arg == "-i" || arg == "--input") && i + 1 < argc) {
            config.input_path = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            config.output_path = argv[++i];
        } else if ((arg == "-a" || arg == "--age") && i + 1 < argc) {
            config.target_age = std::stof(argv[++i]);
        } else if ((arg == "-m" || arg == "--mode") && i + 1 < argc) {
            std::string mode_str = argv[++i];
            if (mode_str == "feedforward") config.mode = ufra::ProcessingMode::FEEDFORWARD;
            else if (mode_str == "diffusion") config.mode = ufra::ProcessingMode::DIFFUSION;
            else if (mode_str == "hybrid") config.mode = ufra::ProcessingMode::HYBRID;
            else if (mode_str == "auto") config.mode = ufra::ProcessingMode::AUTO;
        } else if (arg == "--models" && i + 1 < argc) {
            config.models_path = argv[++i];
        } else if (arg == "--gpu" && i + 1 < argc) {
            std::string gpu_str = argv[++i];
            if (gpu_str == "cuda") config.gpu_backend = ufra::GPUBackend::CUDA;
            else if (gpu_str == "metal") config.gpu_backend = ufra::GPUBackend::METAL;
            else if (gpu_str == "directml") config.gpu_backend = ufra::GPUBackend::DIRECTML;
            else if (gpu_str == "cpu") config.gpu_backend = ufra::GPUBackend::CPU_FALLBACK;
        } else if (arg == "--batch-size" && i + 1 < argc) {
            config.batch_size = std::stoi(argv[++i]);
        } else if (arg == "--identity-lock" && i + 1 < argc) {
            config.identity_lock = std::stof(argv[++i]);
        } else if (arg == "--temporal-stability") {
            config.temporal_stability = true;
        }
    }
    
    return config;
}

int processVideo(const CLIConfig& config, ufra::Engine& engine) {
    cv::VideoCapture cap(config.input_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open input video: " << config.input_path << std::endl;
        return -1;
    }

    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    cv::VideoWriter writer(config.output_path, cv::VideoWriter::fourcc('m','p','4','v'), fps, cv::Size(width, height));
    if (!writer.isOpened()) {
        std::cerr << "Error: Could not open output video: " << config.output_path << std::endl;
        return -1;
    }

    std::cout << "Processing video: " << config.input_path << std::endl;
    std::cout << "Resolution: " << width << "x" << height << ", FPS: " << fps << ", Frames: " << total_frames << std::endl;

    ufra::AgeControls controls;
    controls.target_age = config.target_age;
    controls.identity_lock_strength = config.identity_lock;
    controls.temporal_stability = config.temporal_stability ? 1.0f : 0.0f;

    cv::Mat frame;
    int frame_number = 0;

    while (cap.read(frame)) {
        ufra::FrameContext context;
        context.frame_number = frame_number;
        context.input_frame = frame;
        context.controls = controls;
        context.mode = config.mode;

        ufra::ProcessingResult result = engine.processFrame(context);

        if (result.success) {
            writer.write(result.output_frame);
        } else {
            std::cerr << "Warning: Failed to process frame " << frame_number << ": " << result.error_message << std::endl;
            writer.write(frame); // Write original frame on failure
        }

        // Progress indicator
        if (frame_number % 30 == 0) {
            float progress = static_cast<float>(frame_number) / total_frames * 100.0f;
            std::cout << "Progress: " << std::fixed << std::setprecision(1) << progress << "% (" << frame_number << "/" << total_frames << ")" << std::endl;
        }

        frame_number++;
    }

    std::cout << "Processing complete. Output saved to: " << config.output_path << std::endl;
    return 0;
}

int main(int argc, char* argv[]) {
    CLIConfig config = parseArguments(argc, argv);

    if (config.help || config.input_path.empty() || config.output_path.empty()) {
        printUsage();
        return config.help ? 0 : -1;
    }

    // Initialize engine
    auto engine = ufra::createEngine();
    
    ufra::ModelConfig model_config;
    model_config.backend = config.gpu_backend;
    model_config.batch_size = config.batch_size;
    model_config.use_half_precision = true;
    model_config.max_resolution = 1024;

    if (!engine->initialize(model_config)) {
        std::cerr << "Error: Failed to initialize UFRa engine" << std::endl;
        return -1;
    }

    if (!engine->loadModels(config.models_path)) {
        std::cerr << "Error: Failed to load models from: " << config.models_path << std::endl;
        return -1;
    }

    engine->setProcessingMode(config.mode);

    std::cout << "UFRa CLI initialized successfully" << std::endl;
    std::cout << "Engine version: " << engine->getVersionInfo() << std::endl;

    return processVideo(config, *engine);
}