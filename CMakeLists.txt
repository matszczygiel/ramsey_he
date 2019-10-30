cmake_minimum_required(VERSION 3.8)

project(he-korobov 
        VERSION 0.1.0 
        LANGUAGES CXX)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(CMAKE_CXX_EXTENSIONS OFF)

    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif()

file(GLOB_RECURSE sources CONFIGURE_DEPENDS
    src/*.cpp            src/*.h
    )


add_executable(he_kor ${sources})

target_compile_features(he_kor 
                        PRIVATE 
                        cxx_std_17)

target_compile_options(he_kor 
                       PRIVATE 
                       -Wall -Wextra -pedantic)

                       