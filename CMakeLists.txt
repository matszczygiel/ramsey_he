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


find_package(OpenMP REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

find_library(gmp NAMES libgmp.so)
find_library(mpfr NAMES libmpfr.so)


add_executable(he_state_s 
                src/definitions.h
                src/functions.h
                src/nelder_mead.h
                src/stateS_main.cpp
                )

target_compile_features(he_state_s 
                        PRIVATE 
                        cxx_std_17)

target_compile_options(he_state_s 
                       PRIVATE 
                       -Wall -Wextra -pedantic -march=native)


target_link_libraries(he_state_s 
                        PRIVATE 
                        OpenMP::OpenMP_CXX Eigen3::Eigen ${gmp} ${mpfr})

add_executable(nm_test 
                tests/nelder_mead.cpp
                src/nelder_mead.h)        


target_compile_features(nm_test 
                PRIVATE 
                cxx_std_17)                

target_include_directories(nm_test PRIVATE src)

target_compile_options(nm_test
                       PRIVATE 
                       -Wall -Wextra -pedantic)



add_executable(he_state_p 
                src/definitions.h
                src/functions.h
                src/nelder_mead.h
                src/stateP_main.cpp
                )
       
target_compile_features(he_state_p
                        PRIVATE 
                        cxx_std_17)
       
target_compile_options(he_state_p 
                       PRIVATE 
                       -Wall -Wextra -pedantic -march=native)
       
       
target_link_libraries(he_state_p
                        PRIVATE 
                        OpenMP::OpenMP_CXX Eigen3::Eigen ${gmp} ${mpfr})                       


add_executable(he_pol 
                src/definitions.h
                src/functions.h
                src/polarization_main.cpp
                )
               
target_compile_features(he_pol
                        PRIVATE 
                        cxx_std_17)
               
target_compile_options(he_pol 
                       PRIVATE 
                       -Wall -Wextra -pedantic -march=native)
               
               
target_link_libraries(he_pol
                        PRIVATE 
                        OpenMP::OpenMP_CXX Eigen3::Eigen ${gmp} ${mpfr})    