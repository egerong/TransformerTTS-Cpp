cmake_minimum_required(VERSION 3.0.0)
project(TransformerTTS-Cpp VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(TransformerTTS-Cpp main.cpp)

ADD_LIBRARY(Transformer 
    transformer.cpp
)

find_library(TENSORFLOW_LIB tensorflow HINT $ENV{HOME}/libtensorflow2/lib)

target_link_libraries(
    TransformerTTS-Cpp
    Transformer
    -lespeak-ng
    "${TENSORFLOW_LIB}"
)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
