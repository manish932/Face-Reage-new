# Unit tests
find_package(GTest QUIET)

set(TEST_SOURCES
    test_engine.cpp
    test_face_detector.cpp
    test_face_parser.cpp
    test_compositor.cpp
    test_integration.cpp
)

# Create a simple test runner if GTest is not available
if(NOT GTest_FOUND)
    message(WARNING "Google Test not found, creating simple test runner")
    
    # Create simple test main
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/simple_test_main.cpp
        "#include <iostream>\n"
        "int main() {\n"
        "    std::cout << \"UFRa Tests - Simple Runner\" << std::endl;\n"
        "    std::cout << \"GTest not available, skipping detailed tests\" << std::endl;\n"
        "    return 0;\n"
        "}\n"
    )
    
    add_executable(ufra_tests ${CMAKE_CURRENT_BINARY_DIR}/simple_test_main.cpp)
    target_link_libraries(ufra_tests ufra_core)
else()
    add_executable(ufra_tests ${TEST_SOURCES})
    target_link_libraries(ufra_tests
        ufra_core
        GTest::GTest
        GTest::Main
    )
    
    if(OpenCV_FOUND)
        target_link_libraries(ufra_tests ${OpenCV_LIBS})
    endif()
endif()

target_include_directories(ufra_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/core/include
)

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(ufra_tests)