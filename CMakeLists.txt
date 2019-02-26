# Basic cmake setup
cmake_minimum_required( VERSION 3.1 FATAL_ERROR )
project( sgdicp )

# Compiler options
set( CMAKE_CXX_STANDARD 11 )

# Set output directories for libraries and executables
set( BASE_DIR ${CMAKE_SOURCE_DIR} )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${BASE_DIR}/lib )
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BASE_DIR}/bin )

# Find libraries
find_package( PCL REQUIRED COMPONENTS common filters io )
find_package( Boost REQUIRED COMPONENTS program_options system )

include_directories(
    src
    ${Boost_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)
link_directories( ${PCL_LIBRARY_DIRS} )
add_definitions( ${PCL_DEFINITIONS} )

# Add src folder
add_subdirectory(src)
