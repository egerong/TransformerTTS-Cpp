cmake_minimum_required(VERSION 3.5.1)
set (CMAKE_CXX_STANDARD 17)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")

project(TransformerTTS-CPP VERSION 0.1.0)

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

# NLopt
add_subdirectory(lib/nlopt)

# NanoSnap
#add_library(TransformerTTS-CPP STATIC lib/nanosnap)
add_subdirectory(lib/nanosnap)
option(NANOSNAP_BUILD_EXAMPLES "Build NanoSNAP example codes" OFF)
option(NANOSNAP_BUILD_TESTS "Build NanoSNAP test codes" OFF)

    # Link all
target_link_libraries(
    TransformerTTS-CPP
    Includes
    ${ESPEAK_LIB}
    ${TENSORFLOW_LIB}
    nlopt
    nanosnap
)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
