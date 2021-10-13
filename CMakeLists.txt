cmake_minimum_required(VERSION 3.0.0)
project(TransformerTTS-Cpp VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(TransformerTTS-Cpp main.cpp)

ADD_LIBRARY(LibsModule 
    preprocess.cpp
    model.cpp
)

target_link_libraries(
    TransformerTTS-Cpp
    LibsModule
    -lespeak-ng
)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
