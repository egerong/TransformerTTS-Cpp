cmake_minimum_required(VERSION 3.1.0)
set (CMAKE_CXX_STANDARD 17)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
project(TransformerTTS-CPP VERSION 0.1.0)

include(CTest)
enable_testing()

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

# NLopt
add_subdirectory(lib/nlopt)

# Link all
target_link_libraries(
    TransformerTTS-CPP
    Includes
    ${ESPEAK_LIB}
    ${TENSORFLOW_LIB}
    nlopt
)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
