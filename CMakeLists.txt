cmake_minimum_required(VERSION 3.16)

project(Orion VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)

include_directories(include)

add_subdirectory(src)

add_subdirectory(examples)
