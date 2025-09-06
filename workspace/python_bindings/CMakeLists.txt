# Python bindings using pybind11
find_package(pybind11 REQUIRED)

pybind11_add_module(pyufra
    src/python_bindings.cpp
)

target_link_libraries(pyufra PRIVATE ufra_core)

target_compile_definitions(pyufra PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

# Set properties
set_target_properties(pyufra PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    INTERPROCEDURAL_OPTIMIZATION ON
)