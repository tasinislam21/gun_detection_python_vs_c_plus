cmake_minimum_required(VERSION 3.5)
project(Train_MNIST)
# -------------------------------

# Find OpenCV

# -------------------------------

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ../../c++_libraries/opencv/build)

# -------------------------------

# Find LibTorch

# -------------------------------

set(Torch_DIR "../../c++_libraries/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# -------------------------------

# Build executable

# -------------------------------

add_executable(Train_MNIST train_mnist.cpp)
target_link_libraries(Train_MNIST ${OpenCV_LIBS} "${TORCH_LIBRARIES}")
