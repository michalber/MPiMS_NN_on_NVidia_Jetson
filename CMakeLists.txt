cmake_minimum_required(VERSION 3.0.0)
project(NeuralNetworkDetection VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

set(TENSORFLOW_LIB_DIR "/opt/tensorflow/lib") 

add_executable(NeuralNetworkDetection main.cpp TensorflowModelHandler.cpp)

target_include_directories(NeuralNetworkDetection PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(NeuralNetworkDetection PRIVATE ${TENSORFLOW_LIB_DIR}/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(NeuralNetworkDetection ${TENSORFLOW_LIB_DIR}/libtensorflow_cc.so ${TENSORFLOW_LIB_DIR}/libtensorflow_framework.so ${OpenCV_LIBRARIES})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)