cmake_minimum_required(VERSION 3.10)
project(rknn_infer C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# RKNN API 头文件和库
set(RKNN_API_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/rknn_api" CACHE PATH "RKNN API path")

if(EXISTS "${RKNN_API_DIR}/include")
    include_directories(${RKNN_API_DIR}/include)
    link_directories(${RKNN_API_DIR}/lib)
else()
    # 系统安装的 RKNN API
    include_directories(/usr/include)
    link_directories(/usr/lib /usr/lib/aarch64-linux-gnu)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# 共享库
add_library(rknn_infer SHARED rknn_infer.cpp)
target_link_libraries(rknn_infer rknnrt pthread)

# 安装
install(TARGETS rknn_infer LIBRARY DESTINATION lib)
install(FILES rknn_infer.h DESTINATION include)
