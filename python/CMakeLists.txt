cmake_minimum_required(VERSION 3.16...3.26)
project(kiss_icp_pybind VERSION 1.2.3 LANGUAGES CXX)

# Set build type
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(PYBIND11_NEWPYTHON ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# --- SỬA: CHỈ DÙNG LOCAL SOURCE ---
# Trỏ thẳng vào thư mục cpp/kiss_icp nằm ngang hàng thư mục python
get_filename_component(KISS_ICP_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cpp/kiss_icp" ABSOLUTE)

if(NOT EXISTS ${KISS_ICP_CPP_DIR})
  message(FATAL_ERROR "Could not find local KISS-ICP C++ source at: ${KISS_ICP_CPP_DIR}")
endif()

add_subdirectory(${KISS_ICP_CPP_DIR} ${CMAKE_CURRENT_BINARY_DIR}/kiss_icp)
# ----------------------------------

add_subdirectory(kiss_icp/pybind)
