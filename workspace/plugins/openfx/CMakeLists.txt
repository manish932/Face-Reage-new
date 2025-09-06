# OpenFX plugin for Nuke, DaVinci Resolve, Flame
find_package(PkgConfig REQUIRED)

# OpenFX headers (assuming they are in third_party/openfx)
set(OFX_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/openfx/include")

# Plugin sources
set(OFX_SOURCES
    src/UFRa_Plugin.cpp
    src/UFRa_Params.cpp
    src/UFRa_Processor.cpp
    src/UFRa_Interact.cpp
)

# Create the plugin
add_library(UFRa.ofx SHARED ${OFX_SOURCES})

target_include_directories(UFRa.ofx PRIVATE
    ${OFX_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(UFRa.ofx PRIVATE ufra_core)

# Platform-specific settings
if(WIN32)
    set_target_properties(UFRa.ofx PROPERTIES SUFFIX ".ofx")
elseif(APPLE)
    set_target_properties(UFRa.ofx PROPERTIES
        SUFFIX ".ofx"
        BUNDLE TRUE
        BUNDLE_EXTENSION "ofx"
    )
else()
    set_target_properties(UFRa.ofx PROPERTIES SUFFIX ".ofx")
endif()

# Install plugin
if(APPLE)
    install(TARGETS UFRa.ofx DESTINATION "/Library/OFX/Plugins")
elseif(WIN32)
    install(TARGETS UFRa.ofx DESTINATION "C:/Program Files/Common Files/OFX/Plugins")
else()
    install(TARGETS UFRa.ofx DESTINATION "/usr/OFX/Plugins")
endif()