cmake_minimum_required(VERSION 3.10)

project(TranformerNetwork VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(MyExecutable 
    src/main.cpp 
    src/MLPLayer.cpp 
    src/MLP.cpp
    src/activationFunction.cpp
)

target_include_directories(MyExecutable PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)