project(TransformerTTS-CPP VERSION 0.1.0)

cmake_minimum_required(VERSION 3.5.1)
set (CMAKE_CXX_STANDARD 17)


#include(CTest)
#enable_testing()

# Transformer class
add_library(TransformerTTS-CPP src/transformer.cpp)
target_include_directories(TransformerTTS-CPP PUBLIC src)

# Include dir
add_library(Includes INTERFACE)
target_include_directories(Includes INTERFACE include)

# eSpeak
find_library(ESPEAK_LIB espeak-ng)

# Tensorflow
find_library(TENSORFLOW_LIB tensorflow HINT $ENV{HOME}/libtensorflow2/lib)

set(Torch_DIR libtorch/share/cmake/Torch)
set (CUDA_CUDART_LIBRARY /usr/local/cuda/lib64/libcudart.so)
#set (CMAKE_CUDA_COMPILER /usr/local/cuda-11.6/bin/nvcc)
#set(CUDA_HOME /usr/local/cuda)
#set(CUDA_PATH /usr/local/cuda)
#CUDNN_INCLUDE_DIR
#set(PATH /usr/local/cuda-11.6/bin:$PATH)
#set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
#et(CUDA_V "11.6")
set(CUDA_INCLUDE_DIRS /usr/local/cuda-11.5/include)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Link all
target_link_libraries(
    TransformerTTS-CPP
    Includes
    ${ESPEAK_LIB}
    ${TENSORFLOW_LIB}
    #${CUDA_LIBRARIES}
    ${TORCH_LIBRARIES}
)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
