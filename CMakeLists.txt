cmake_minimum_required(VERSION 3.8)
project(slvi_sam)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "-std=c++17 -Wall -lstdc++fs -lX11")
set(CMAKE_CXX_FLAGS_RELEASE  "-std=c++17 -O3 -fopenmp -pthread -lstdc++fs -lX11")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)


## Dependencies

# Eigen
include_directories("/usr/include/eigen3")

# OpenCV
find_package(OpenCV 3.4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# pangolin
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

# glog
find_package(Glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})

# pcl
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS}) 

# GTSAM
find_package(GTSAM REQUIRED QUIET)
include_directories(${GTSAM_INCLUDE_DIR}) 

# Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# Boost
find_package(Boost REQUIRED COMPONENTS timer)

# G2O
find_package(g2o REQUIRED
             COMPONENTS
             g2o::core
             g2o::stuff
             g2o::types_sba
             g2o::types_sim3
             g2o::solver_dense
             g2o::solver_eigen
             g2o::solver_csparse
             g2o::csparse_extension
             OPTIONAL_COMPONENTS
             g2o::csparse)

# Check first if CSparse is built from g2o
if(TARGET g2o::csparse)
    set(${CXSPARSE_LIBRARIES} g2o::csparse)
else()
    # CXSparse for g2o
    find_package(CXSparse)
    if(CXSPARSE_FOUND)
        include_directories(${CXSPARSE_INCLUDE_DIRS})
    endif()
    # SuiteSparse for g2o
    if(MSVC)
        find_package(SuiteSparse CONFIG)
    else()
        find_package(SuiteSparse)
    endif()
    if(SUITESPARSE_FOUND)
        include_directories(${SUITESPARSE_INCLUDE_DIRS})
    endif()
endif()

# #${Pangolin_LIBRARIES} GL GLU GLEW glut
set(THIRD_PARTY_LIBS
        ${OpenCV_LIBS}
        pthread
        ${GLOG_LIBRARIES}
        ${PCL_LIBRARIES}
        ${Pangolin_LIBRARIES}
        stdc++fs
        ${GTSAM_LIBRARY_DIRS}
        gtsam
        ${CERES_LIBRARY_DIRS}
        ceres
        Boost::timer
        g2o::core
        g2o::stuff
        g2o::types_sba
        g2o::types_sim3
        g2o::solver_dense
        g2o::solver_eigen
        g2o::solver_csparse
        g2o::csparse_extension
        )

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include/LIO)
include_directories(${PROJECT_SOURCE_DIR}/include/VIO)
add_subdirectory(src)
add_subdirectory(app)
