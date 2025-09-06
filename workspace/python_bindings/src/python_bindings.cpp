#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <opencv2/opencv.hpp>
#include "ufra/engine.h"
#include "ufra/types.h"

namespace py = pybind11;

// Helper functions for OpenCV Mat conversion
cv::Mat numpy_to_mat(py::array_t<uint8_t> input) {
    py::buffer_info buf_info = input.request();
    cv::Mat mat(buf_info.shape[0], buf_info.shape[1], CV_8UC3, (unsigned char*)buf_info.ptr);
    return mat.clone();
}

py::array_t<uint8_t> mat_to_numpy(cv::Mat mat) {
    return py::array_t<uint8_t>(
        {mat.rows, mat.cols, mat.channels()},
        {sizeof(uint8_t)*mat.cols*mat.channels(), sizeof(uint8_t)*mat.channels(), sizeof(uint8_t)},
        mat.data
    );
}

PYBIND11_MODULE(pyufra, m) {
    m.doc() = "Universal Face Re-Aging (UFRa) Python Bindings";

    // Enums
    py::enum_<ufra::ProcessingMode>(m, "ProcessingMode")
        .value("FEEDFORWARD", ufra::ProcessingMode::FEEDFORWARD)
        .value("DIFFUSION", ufra::ProcessingMode::DIFFUSION)
        .value("HYBRID", ufra::ProcessingMode::HYBRID)
        .value("AUTO", ufra::ProcessingMode::AUTO);

    py::enum_<ufra::GPUBackend>(m, "GPUBackend")
        .value("CUDA", ufra::GPUBackend::CUDA)
        .value("METAL", ufra::GPUBackend::METAL)
        .value("DIRECTML", ufra::GPUBackend::DIRECTML)
        .value("CPU_FALLBACK", ufra::GPUBackend::CPU_FALLBACK);

    // Structures
    py::class_<ufra::FaceBox>(m, "FaceBox")
        .def(py::init<>())
        .def_readwrite("x", &ufra::FaceBox::x)
        .def_readwrite("y", &ufra::FaceBox::y)
        .def_readwrite("width", &ufra::FaceBox::width)
        .def_readwrite("height", &ufra::FaceBox::height)
        .def_readwrite("confidence", &ufra::FaceBox::confidence)
        .def_readwrite("face_id", &ufra::FaceBox::face_id);

    py::class_<ufra::Face>(m, "Face")
        .def(py::init<>())
        .def_readwrite("box", &ufra::Face::box)
        .def_readwrite("track_id", &ufra::Face::track_id)
        .def_readwrite("frame_number", &ufra::Face::frame_number);

    py::class_<ufra::AgeControls>(m, "AgeControls")
        .def(py::init<>())
        .def_readwrite("target_age", &ufra::AgeControls::target_age)
        .def_readwrite("identity_lock_strength", &ufra::AgeControls::identity_lock_strength)
        .def_readwrite("temporal_stability", &ufra::AgeControls::temporal_stability)
        .def_readwrite("texture_keep", &ufra::AgeControls::texture_keep)
        .def_readwrite("skin_clean", &ufra::AgeControls::skin_clean)
        .def_readwrite("enable_hair_aging", &ufra::AgeControls::enable_hair_aging)
        .def_readwrite("gray_density", &ufra::AgeControls::gray_density);

    py::class_<ufra::ModelConfig>(m, "ModelConfig")
        .def(py::init<>())
        .def_readwrite("model_path", &ufra::ModelConfig::model_path)
        .def_readwrite("backend", &ufra::ModelConfig::backend)
        .def_readwrite("batch_size", &ufra::ModelConfig::batch_size)
        .def_readwrite("use_half_precision", &ufra::ModelConfig::use_half_precision)
        .def_readwrite("max_resolution", &ufra::ModelConfig::max_resolution);

    py::class_<ufra::ProcessingResult>(m, "ProcessingResult")
        .def(py::init<>())
        .def_readwrite("processed_faces", &ufra::ProcessingResult::processed_faces)
        .def_readwrite("metrics", &ufra::ProcessingResult::metrics)
        .def_readwrite("success", &ufra::ProcessingResult::success)
        .def_readwrite("error_message", &ufra::ProcessingResult::error_message)
        .def("get_output_frame", [](const ufra::ProcessingResult &result) {
            return mat_to_numpy(result.output_frame);
        });

    py::class_<ufra::FrameContext>(m, "FrameContext")
        .def(py::init<>())
        .def_readwrite("frame_number", &ufra::FrameContext::frame_number)
        .def_readwrite("detected_faces", &ufra::FrameContext::detected_faces)
        .def_readwrite("controls", &ufra::FrameContext::controls)
        .def_readwrite("mode", &ufra::FrameContext::mode)
        .def("set_input_frame", [](ufra::FrameContext &ctx, py::array_t<uint8_t> input) {
            ctx.input_frame = numpy_to_mat(input);
        });

    // Main Engine class
    py::class_<ufra::Engine>(m, "Engine")
        .def(py::init<>())
        .def("initialize", &ufra::Engine::initialize)
        .def("is_initialized", &ufra::Engine::isInitialized)
        .def("load_models", &ufra::Engine::loadModels)
        .def("process_frame", &ufra::Engine::processFrame)
        .def("detect_faces", [](ufra::Engine &engine, py::array_t<uint8_t> input) {
            cv::Mat image = numpy_to_mat(input);
            return engine.detectFaces(image);
        })
        .def("estimate_age", &ufra::Engine::estimateAge)
        .def("set_processing_mode", &ufra::Engine::setProcessingMode)
        .def("get_processing_mode", &ufra::Engine::getProcessingMode)
        .def("get_version_info", &ufra::Engine::getVersionInfo)
        .def("set_error_callback", &ufra::Engine::setErrorCallback);

    // Factory functions
    m.def("create_engine", &ufra::createEngine, "Create a new UFRa engine instance");
    m.def("get_library_version", &ufra::getLibraryVersion, "Get library version");
    m.def("get_available_backends", &ufra::getAvailableBackends, "Get available GPU backends");

    // Helper functions
    m.def("numpy_to_mat", &numpy_to_mat, "Convert numpy array to OpenCV Mat");
    m.def("mat_to_numpy", &mat_to_numpy, "Convert OpenCV Mat to numpy array");
}