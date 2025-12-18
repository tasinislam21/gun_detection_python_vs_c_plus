cmake_minimum_required(VERSION 3.5)
project(Detect_firearm)
# -------------------------------

# Find OpenCV

# -------------------------------

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ../../c++_libraries/opencv/build)

# -------------------------------

# Find LibTorch

# -------------------------------

set(Torch_DIR "../../c++_libraries/libtorch_13/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# -------------------------------

# Build executable

# -------------------------------

add_executable(Detect_firearm detect_firearm.cpp)
target_link_libraries(Detect_firearm ${OpenCV_LIBS} "${TORCH_LIBRARIES}")