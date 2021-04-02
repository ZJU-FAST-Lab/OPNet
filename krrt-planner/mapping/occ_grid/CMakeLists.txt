cmake_minimum_required(VERSION 2.8.3)
project(occ_grid)
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -O3 -Wall")

set(CUDA_INSTALL_DIR /usr/local/cuda-10.2)
set(CUDA_LIBRARIES ${CUDA_INSTALL_DIR}/lib64)
set(ONNX_INCLUDE_DIR models CACHE STRING "ONNX include directory")
set(DEPS_LIST "")


list(APPEND DEPS_LIST nvinfer_plugin)
# list(APPEND DEPS_LIST nvuffparser nvcaffeparser nvonnxparser)
list(APPEND DEPS_LIST nvonnxparser)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  visualization_msgs
  message_filters
  cv_bridge
  tf2_ros
  tf2
  message_generation
  ocp_msgs
)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL 1.7 REQUIRED)
find_package(CUDA)
find_library(CUBLAS_LIBRARIES cublas HINTS
    ${CUDA_INSTALL_DIR}/lib64)

message(STATUS "${OpenCV_INCLUDE_DIRS}")
message(STATUS "${PCL_INCLUDE_DIRS}")

add_message_files(FILES
  CenteredPointcloud.msg
)

generate_messages(DEPENDENCIES std_msgs)
catkin_package(
 INCLUDE_DIRS include
 LIBRARIES occ_grid
 CATKIN_DEPENDS roscpp std_msgs message_runtime ocp_msgs
#  DEPENDS system_lib
)

message("=================================================================")
message("=================================================================")
message("=================================================================")
message("${catkin_INCLUDE_DIRS}")
message("${CUDNN_LIB}")


# include_directories(common)

include_directories( 
    include 
    common
    # ${ONNX_INCLUDE_DIR}
    ${CUDA_INSTALL_DIR}/include

    ${catkin_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS} 
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)


link_directories(${PCL_LIBRARY_DIRS})
link_directories(${OpenCV_LIBRARIES})
# link_directories(${CUDA_LIBRARIES})

add_library( occ_grid 
    src/occ_map.cpp 
    src/raycast.cpp
    common/logger.cpp
)

# target_include_directories( occ_grid
#   ./common
# )

add_dependencies(occ_grid
  # ${DEPS_LIST}
  ${catkin_EXPORTED_TARGETS}
)

# target_include_directories(occ_grid
#     PUBLIC /include
#     PUBLIC ${ONNX_INCLUDE_DIR}
#     PUBLIC ${CUDA_INSTALL_DIR}/include
#     PRIVATE common
#     # PRIVATE ${TARGET_DIR}
# )

# target_compile_options(${TARGET_NAME} PUBLIC "-fno-rtti")

set(DEP_LIBS
    ${CUDA_LIBRARIES}
    ${CUDART_LIB}
    ${CUBLAS_LIBRARIES}
    #${CUBLAS_LIB}
    ${CUDNN_LIB}
    nvinfer
    ${RT_LIB}
    ${CMAKE_DL_LIBS}
    ${CMAKE_THREAD_LIBS_INIT}
)
list(APPEND DEP_LIBS nvinfer_plugin)
list(APPEND DEP_LIBS nvonnxparser)


target_link_libraries( occ_grid
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${OpenCV_LIBRARIES}
)  
target_link_libraries(occ_grid
    ${DEP_LIBS}
    # -Wl,--unresolved-symbols=ignore-in-shared-libs
)

# set_target_properties(occ_grid PROPERTIES LINK_FLAGS "-Wl,--exclude-libs,ALL")

# set_property(TARGET occ_grid PROPERTY CXX_STANDARD 11)
