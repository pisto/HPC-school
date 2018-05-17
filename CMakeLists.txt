cmake_minimum_required(VERSION 3.10)
project(mpibologna)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -ggdb3 -Wall -DOMPI_SKIP_MPICXX -mtune=intel")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -fopt-info-vec -fopt-info-loop -fopt-info-omp")
set(CMAKE_CXX_FLAGS_DEBUG "")

find_package(MPI REQUIRED)
set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} "${MPI_LINK_FLAGS} -static-libgcc -static-libstdc++")
include_directories(${MPI_INCLUDE_PATH})

set(Boost_USE_STATIC_LIBS ON)
find_package(Boost REQUIRED mpi program_options)

add_executable(mpibologna main.cpp mpienv.hpp misc.hpp)
target_link_libraries(mpibologna "-Wl,--as-needed" ${Boost_LIBRARIES} ${MPI_LIBRARIES})
