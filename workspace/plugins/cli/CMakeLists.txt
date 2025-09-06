# CLI executable
add_executable(ufra_cli
    ../cli/main.cpp
)

target_link_libraries(ufra_cli PRIVATE 
    ufra_core
    ${OpenCV_LIBS}
)

target_include_directories(ufra_cli PRIVATE
    ${CMAKE_SOURCE_DIR}/core/include
)

# Install CLI
install(TARGETS ufra_cli DESTINATION bin)