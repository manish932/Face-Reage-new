# Core engine library
set(CORE_SOURCES
    src/engine.cpp
    src/face_detector.cpp
    src/face_tracker.cpp
    src/age_estimator.cpp
    src/face_parser.cpp
    src/feedforward_generator.cpp
    src/diffusion_editor.cpp
    src/optical_flow.cpp
    src/compositor.cpp
    src/texture_manager.cpp
    src/gpu_memory_manager.cpp
    src/model_loader.cpp
    src/utils.cpp
)

set(CORE_HEADERS
    include/ufra/engine.h
    include/ufra/face_detector.h
    include/ufra/face_tracker.h
    include/ufra/age_estimator.h
    include/ufra/face_parser.h
    include/ufra/feedforward_generator.h
    include/ufra/diffusion_editor.h
    include/ufra/optical_flow.h
    include/ufra/compositor.h
    include/ufra/texture_manager.h
    include/ufra/gpu_memory_manager.h
    include/ufra/model_loader.h
    include/ufra/types.h
    include/ufra/utils.h
)

# Create shared library
add_library(ufra_core SHARED ${CORE_SOURCES} ${CORE_HEADERS})

# Include directories
target_include_directories(ufra_core 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link libraries
target_link_libraries(ufra_core)

# Link OpenCV if available
if(OpenCV_FOUND)
    target_link_libraries(ufra_core ${OpenCV_LIBS})
    target_compile_definitions(ufra_core PRIVATE OPENCV_FOUND)
endif()

# Link CUDA libraries if available
if(CUDA_ENABLED AND CUDAToolkit_FOUND)
    target_link_libraries(ufra_core
        CUDA::cudart
        CUDA::cublas
        CUDA::curand
    )
    target_compile_definitions(ufra_core PRIVATE CUDA_FOUND)
endif()

# Set properties
set_target_properties(ufra_core PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    POSITION_INDEPENDENT_CODE ON
)

# Install targets
install(TARGETS ufra_core
    EXPORT UFRaTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/ DESTINATION include)