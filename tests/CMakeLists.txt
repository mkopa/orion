# tests/CMakeLists.txt
# Builds GoogleTest-based unit tests for orion

cmake_minimum_required(VERSION 3.14)

# Use FetchContent to get googletest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(orion_tests
    test_orion.cpp
)

target_include_directories(orion_tests PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(orion_tests PRIVATE orion GTest::gtest_main)

# make sure to link threads on linux
find_package(Threads REQUIRED)
target_link_libraries(orion_tests PRIVATE Threads::Threads)

include(GoogleTest)
gtest_discover_tests(orion_tests)
